"""Fit Erlang mixtures to the public duration intervals of Hu et al. (2021).

Read the released bounds, fit each candidate using interval likelihood,
compare an independent Weibull fit, and plot both CDFs against a
nonparametric interval-data estimate."""
from pathlib import Path
import csv
import json
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy.special import gammainc, gammaincc, gammaln
from scipy.optimize import minimize
from scipy.stats import weibull_min
from scipy.integrate import solve_ivp
from em_core import cdf, log_density, initializations
from study import plot_style

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/hunan_incubation'
OUT = ROOT/'results/hunan_incubation'
LS = (3, 5, 10, 20, 50, 100)


def read_intervals():
    # Read the two numeric columns in the archived workbook using standard
    # Python XML tools. The workbook itself is preserved without modification.
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(DATA/'data_Table_S3.xlsx') as archive:
        sheets = ET.fromstring(archive.read('xl/workbook.xml'))
        assert sheets.find('s:sheets/s:sheet', ns).attrib['name'] == 'Main analysis'
        xml = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    pairs = []
    for row in xml.findall('s:sheetData/s:row', ns)[1:]:
        cells = {cell.attrib['r'].rstrip('0123456789'):
                 float(cell.find('s:v', ns).text) for cell in row}
        pairs.append((cells['A'], cells['B']))
    x = np.asarray(pairs)
    assert len(x) == 268 and np.isfinite(x).all()
    assert np.all(x[:, 0] >= 0) and np.all(x[:, 1] >= x[:, 0])
    assert np.all(x[:, 1] > 0)
    np.savetxt(DATA/'intervals.csv', x, delimiter=',', header='left,right',
               comments='', fmt='%.17g')
    return x[:, 0], x[:, 1]


def interval_mass(lo, hi, k, rate):
    # Switch to survival differences in the upper tail to avoid cancellation.
    a, b = rate*lo[:, None], rate*hi[:, None]
    p = gammainc(k, b)-gammainc(k, a)
    use_survival = gammainc(k, a) > .5
    tail = gammaincc(k, a)-gammaincc(k, b)
    return np.maximum(np.where(use_survival, tail, p), 0.)


def component_terms(lo, hi, L, rate):
    k = np.arange(1, L+1)
    # p[i,k] is the probability of an observed interval; for an exact value it
    # is the component density. 
    all_mass = interval_mass(lo, hi, np.arange(1, L+2), rate)
    p = all_mass[:, :-1].copy()
    # Integral of t*f_k(t) over an interval is (k/r)*P_{k+1}(interval).
    moment = all_mass[:, 1:] * (k/rate)
    exact = lo == hi
    t = lo[exact, None]
    p[exact] = np.exp(k*np.log(rate)+(k-1)*np.log(t)-rate*t-gammaln(k))
    moment[exact] = t*p[exact]
    return p, moment


def fit_em(lo, hi, L, rate, weights_init=None, max_iter=30000,
           tol_per_sample=1e-8, rate_tol=1e-6, weight_tol=1e-5, patience=5):
    """EM with hidden route and hidden duration inside each observed interval."""
    n = len(lo)
    # Accept the same uniform or bin-based starting weights as Experiment 1.
    w = np.full(L, 1/L) if weights_init is None else np.array(weights_init, dtype=float, copy=True)
    if w.shape != (L,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Initial weights must be finite, nonnegative, have positive total mass and length L.")
    # Match Experiment 1, including normalization before a continued run.
    w /= w.sum()
    k = np.arange(1, L+1)
    p, moment = component_terms(lo, hi, L, rate)
    ll = np.log(p@w).sum()
    stable = 0
    for iteration in range(1, max_iter+1):
        prob = p@w
        q = p*w/prob[:, None]
        total_time = np.sum((moment@w)/prob)
        # Complete-data M-step: weights are average route probabilities;
        # rate is expected total phase count divided by expected duration.
        w_new = q.mean(axis=0)
        w_new /= w_new.sum()
        rate_new = float(np.dot(q.sum(axis=0), k)/total_time)
        p_new, moment_new = component_terms(lo, hi, L, rate_new)
        ll_new = float(np.log(p_new@w_new).sum())
        if ll_new < ll-1e-8:
            raise RuntimeError('EM likelihood decreased.')
        small = (abs(ll_new-ll)/n < tol_per_sample and
                 abs(rate_new-rate)/rate < rate_tol and
                 np.max(np.abs(w_new-w)) < weight_tol)
        stable = stable+1 if small else 0
        w, rate, ll, p, moment = w_new, rate_new, ll_new, p_new, moment_new
        if stable >= patience:
            break
    return dict(L=L, weights=w.tolist(), rate=rate, loglik=ll,
                BIC=L*np.log(n)-2*ll, iterations=iteration, converged=stable >= patience)


def weibull_fit(lo, hi):
    # Fit the Weibull comparator independently to the same interval likelihood.
    exact = lo == hi
    def objective(logpars):
        shape, scale = np.exp(logpars)
        dist = weibull_min(shape, scale=scale)
        p = dist.sf(lo)-dist.sf(hi)
        p[exact] = dist.pdf(lo[exact])
        return -np.log(np.maximum(p, 1e-300)).sum()
    fit = minimize(objective, np.log([1.5, 7.]), method='BFGS',
                   options={'gtol': 1e-6})
    shape, scale = np.exp(fit.x)
    return dict(shape=float(shape), scale=float(scale), loglik=float(-fit.fun),
                BIC=float(2*np.log(len(lo))+2*fit.fun),
                AIC=float(4+2*fit.fun))


def empirical_cdf(lo, hi):
    # An endpoint-supported interval-censored likelihood fit.
    support = np.unique(np.r_[lo[lo > 0], hi])
    exact = lo == hi
    a = ((support > lo[:, None]) & (support <= hi[:, None])).astype(float)
    a[exact] = support == lo[exact, None]
    mass = np.full(len(support), 1/len(support))
    for iteration in range(100000):
        p = a@mass
        new = mass*(a.T@(1/p))/len(lo)
        if np.max(np.abs(new-mass)) < 1e-10:
            mass = new
            break
        mass = new
    # KKT score checks numerical stationarity of this concave mass problem.
    score = a.T@(1/(a@mass))/len(lo)
    assert np.max(score) < 1+1e-6
    return support, mass


def plot_figure7(selected, reference, support, mass, maximum_upper):
    """Plot Figure 7"""
    w, rate = np.asarray(selected['weights']), selected['rate']
    grid = np.linspace(0, maximum_upper*1.2, 2001)
    empirical = np.cumsum(mass)
    wb = weibull_min(reference['shape'], scale=reference['scale'])

    times = np.unique(np.r_[grid, support, np.nextafter(support, -np.inf)])
    ref_cdf = np.r_[0., empirical][np.searchsorted(support, times, side='right')]
    fitted = cdf(times, w, rate)
    weibull = wb.cdf(times)
    difference = np.abs(fitted-ref_cdf)
    difference_w = np.abs(weibull-ref_cdf)

    left = np.r_[0., empirical[:-1]]
    D = max(np.max(abs(cdf(support,w,rate)-empirical)),
            np.max(abs(cdf(support,w,rate)-left)))
    Dw = max(np.max(abs(wb.cdf(support)-empirical)),
             np.max(abs(wb.cdf(support)-left)))
    assert np.isclose(difference.max(), D, atol=1e-12, rtol=0)
    assert np.isclose(difference_w.max(), Dw, atol=1e-12, rtol=0)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUT/'figure7_cdf_discrepancies.csv',
        np.c_[times, ref_cdf, fitted, weibull, difference, difference_w],
        delimiter=',',
        header='time,reference_cdf,mixture_cdf,weibull_cdf,mixture_abs_difference,weibull_abs_difference',
        comments='',
        fmt='%.17g'
    )

    (OUT/'figure7_metrics.json').write_text(json.dumps(dict(
        mixture_max_absolute_CDF_difference=float(D),
        weibull_max_absolute_CDF_difference=float(Dw),
        reference='Endpoint-supported interval-data NPMLE',
        interpretation='Descriptive CDF discrepancies relative to the endpoint-supported estimate',
        maximum_checks='Left and right limits at every reference-CDF jump'),
        indent=2), encoding='utf-8')

    plt = plot_style()
    (ROOT/'figures').mkdir(exist_ok=True)

    # Figure 7(a): CDF comparison
    fig, ax = plt.subplots(figsize=(5,4))
    ax.step(np.r_[0,support], np.r_[0,empirical], where='post',
            color='#e1b956', label='Data (interval NPMLE)')
    ax.plot(grid, cdf(grid,w,rate), '-', color='#c9777a', label='Mixture Erlang')
    ax.plot(grid, wb.cdf(grid), '--', color='#7099bf', label='Weibull')
    ax.set_xlabel('Incubation period (days)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, grid[-1])
    ax.legend()
    # ax.text(-.12, 1.03, '(a)', transform=ax.transAxes)
    fig.tight_layout()
    for extension in ('pdf','png','svg'):
        fig.savefig(ROOT/'figures'/f'Fig7a_Hunan_incubation.{extension}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 7(b): absolute CDF discrepancy
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(times, difference, '-', color='#c9777a',
            label=f'Mixture Erlang (max = {D:.4f})')
    ax.plot(times, difference_w, '--', color='#7099bf',
            label=f'Weibull (max = {Dw:.4f})')
    ax.set_xlabel('Incubation period (days)')
    ax.set_ylabel('Absolute CDF difference')
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(0, max(D,Dw)*1.28)
    ax.legend()
    # ax.text(-.12, 1.03, '(b)', transform=ax.transAxes)
    fig.tight_layout()
    for extension in ('pdf','png','svg'):
        fig.savefig(ROOT/'figures'/f'Fig7b_Hunan_incubation.{extension}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)

    return float(D), float(Dw)


def fit_candidate(lo, hi, L):
    """Match Experiment 1: six 30000-step runs, then continue only the best.

    Initial rates and weights come from the exact same generator. Bounds
    supply initialization information only; the observed-data likelihood is
    interval based. All records include their stage to distinguish restarts
    from continuation of a single chosen run.
    """
    candidates, records = [], []
    for index, (rate, weights) in enumerate(initializations(hi, L), 1):
        fit = fit_em(lo, hi, L, rate, weights_init=weights)
        candidates.append(fit)
        records.append(dict(L=L, start_index=index, stage='initial',
            max_iter=30000, initial_rate=rate, initial_weights=weights.tolist(),
            **{k:fit[k] for k in ['rate','loglik','iterations','converged']}))
        print(f"L={L}, start={index}: ll={fit['loglik']:.8f}, "
              f"iterations={fit['iterations']}, converged={fit['converged']}", flush=True)
    # Select BEFORE continuation, exactly as in em_core.fit_multistart.
    index = max(range(6), key=lambda i: candidates[i]['loglik'])
    best = candidates[index]
    if not best['converged']:
        rate, weights = best['rate'], np.asarray(best['weights'])
        best = fit_em(lo, hi, L, rate, weights_init=weights, max_iter=20000)
        records.append(dict(L=L, start_index=index+1, stage='continuation',
            max_iter=20000, initial_rate=rate, initial_weights=weights.tolist(),
            **{k:best[k] for k in ['rate','loglik','iterations','converged']}))
    if not best['converged']:
        raise RuntimeError(f'L={L}: increase the iteration allowance before reporting BIC.')
    best['selected_start'] = index+1
    best['total_selected_iterations'] = sum(r['iterations'] for r in records
                                           if r['start_index']==index+1)
    return best, records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = read_intervals()
    fits, runs = [], []
    start = time.process_time()
    for L in LS:
        best, records = fit_candidate(lo, hi, L)
        fits.append(best)
        runs.extend(records)
        (OUT/f'fit_L{L}.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    finish_results(lo, hi, fits, runs, time.process_time()-start)


def finish_results(lo, hi, fits, runs, fitting_time):
    """Save assessments and Figure 7 for the best fit at every candidate L."""
    selected = min(fits, key=lambda f: f['BIC'])
    reference = weibull_fit(lo, hi)
    support, mass = empirical_cdf(lo, hi)
    w, r, L = np.array(selected['weights']), selected['rate'], selected['L']
    grid = np.linspace(0, max(hi)*1.2, 2001)
    fitted = cdf(grid, w, r)
    wb = weibull_min(reference['shape'], scale=reference['scale'])
    empirical = np.cumsum(mass)
    at_jumps = cdf(support, w, r)
    # Check left and right limits at every jump of the reference CDF.
    D = max(np.max(np.abs(at_jumps-empirical)),
            np.max(np.abs(at_jumps-np.r_[0, empirical[:-1]])))
    Dw = max(np.max(np.abs(wb.cdf(support)-empirical)),
             np.max(np.abs(wb.cdf(support)-np.r_[0, empirical[:-1]])))
    # A unit cohort checks the fitted duration CDF through its ODE realization.
    # The accumulated exit probability corresponds to symptom onset.
    starts = np.r_[0, np.cumsum(np.arange(1, L))]
    ends = starts+np.arange(1, L+1)-1
    size = L*(L+1)//2
    inner = np.setdiff1d(np.arange(size), starts)
    initial = np.zeros(size+1)
    initial[starts] = w
    def rhs(t, y):
        dy = np.zeros_like(y)
        dy[:-1] = -r*y[:-1]
        dy[inner] += r*y[inner-1]
        dy[-1] = r*y[ends].sum()
        return dy
    sol = solve_ivp(rhs, (0, grid[-1]), initial, t_eval=grid,
                    method='DOP853', rtol=1e-10, atol=1e-12)
    assert sol.success
    summary = dict(n=len(lo), exact=int(np.sum(lo == hi)),
                   interval=int(np.sum(lo < hi)), left_zero=int(np.sum(lo == 0)),
                   maximum_upper=float(hi.max()), selected=selected, weibull=reference,
                   CDF_distance_to_nonparametric=float(D), weibull_CDF_distance=float(Dw),
                   ODE_CDF_max_error=float(np.max(np.abs(sol.y[-1]-fitted))),
                   ODE_mass_max_error=float(np.max(np.abs(sol.y.sum(axis=0)-1))),
                   fitting_cpu_seconds=fitting_time)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT/'all_starts.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
    with (OUT/'model_selection.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['L','rate','loglik','BIC','iterations','converged'])
        writer.writeheader()
        writer.writerows({k: f[k] for k in writer.fieldnames} for f in fits)
    np.savetxt(OUT/'nonparametric.csv', np.c_[support, mass, empirical],
               delimiter=',', header='duration,mass,cdf', comments='')
    plot_figure7(selected, reference, support, mass, float(hi.max()))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
