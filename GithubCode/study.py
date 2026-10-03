"""Shared distribution definitions and simple file/plot helpers."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.stats import norm, lognorm, weibull_min
from scipy.optimize import brentq
from em_core import fit_em, fit_multistart, save_csv

ROOT = Path(__file__).resolve().parent
N = 10000 # sample size
CANDIDATES = (3, 5, 10, 20, 50, 100) # candidates of L
MAIN = ('lognormal', 'weibull', 'normal_mixture', 'normal_lognormal')
SEEDS = dict(lognormal=9901, weibull=9902, normal_mixture=9903,
             normal_lognormal=9904, sensitivity_normal=999)

def parts(name):
    # Parameters below are four target distributions.
    # SciPy's lognormal scale is exp(mu), and its shape argument is sigma.
    return {'lognormal':[(1.,lognorm(s=1,scale=1))],
            'weibull':[(1.,weibull_min(c=1.5,scale=1))],
            'normal_mixture':[(.4,norm(2,.5)),(.6,norm(7,1))],
            'normal_lognormal':[(.5,norm(3,.8)),(.5,lognorm(s=.4,scale=np.exp(2)))],
            'sensitivity_normal':[(1.,norm(10,1.5))]}[name]

def target_cdf(name,x):
    # Compute the cdf
    x = np.asarray(x, float)
    components = parts(name)
    F = sum(w*d.cdf(x) for w,d in components)
    return np.where(x >= 0, np.clip(F, 0, 1), 0.)

def target_sf(name,x):
    # Compute the survival function
    # Evaluate survival directly to avoid subtracting nearly equal tail values.
    x = np.asarray(x, float)
    components = parts(name)
    Z = sum(w*d.sf(0) for w,d in components)
    return np.where(x >= 0, sum(w*d.sf(x) for w,d in components)/Z, 1.)

def target_ppf(name,p):
    # Numerically invert the target CDF for the Q-Q and P-P plots.
    def quantile(prob):
        upper=1.
        while target_cdf(name,upper)<prob: upper*=2
        return brentq(lambda t:float(target_cdf(name,t))-prob,0,upper)
    return np.array([quantile(float(prob)) for prob in np.atleast_1d(p)])

def sample(name,n=N):
    # Sample the data
    rng = np.random.default_rng(SEEDS[name])
    components = parts(name)
    chunks = []
    total = 0
    while total<n:
        size=n-total
        choices=rng.choice(len(components),size=size,p=[w for w,d in components])
        draw=np.empty(size)
        for i,(_,d) in enumerate(components):
            mask=choices==i
            draw[mask]=d.rvs(size=int(mask.sum()),random_state=rng)
        chunks.append(draw)
        total += len(draw)
    return np.concatenate(chunks)

def dump(path,obj):
    # Store parameters and numerical results as readable JSON lists.
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,default=encode),encoding='utf-8')

def load_fit(name, L=None):
    """Read a saved fit produced by the corresponding estimation script."""
    filename = 'selected.json' if L is None else f'fit_L{L}.json'
    fit = json.loads((ROOT/'results'/name/filename).read_text())
    fit['weights'] = np.asarray(fit['weights'])
    return fit

def fit_data(name, data):
    """Run the complete finite BIC search on the observations."""
    for folder in ('data', 'results', 'figures'):
        (ROOT/folder).mkdir(exist_ok=True)
    directory = ROOT/'results'/name
    directory.mkdir(exist_ok=True)
    np.savetxt(ROOT/'data'/f'{name}.csv', data, delimiter=',',
               header='duration', comments='', fmt='%.17g')
    fits = []
    # Accumulate CPU time for the six complete multistart searches.
    elapsed = 0.
    for L in CANDIDATES:
        start = time.process_time()
        fit = fit_multistart(data, L)
        elapsed += time.process_time() - start
        # Output the fitted parameters and selection information.
        fit = {key: fit[key] for key in ('L', 'weights', 'rate', 'loglik', 'BIC')}
        fits.append(fit)
        dump(directory/f'fit_L{L}.json', fit)
    selected = min(fits, key=lambda fit: fit['BIC']).copy()
    selected['fitting_cpu_seconds'] = elapsed
    dump(directory/'selected.json', selected)
    save_csv(directory/'model_selection.csv',
             [{key: fit[key] for key in ('L', 'rate', 'loglik', 'BIC')} for fit in fits])
    print(f"{name}: selected L={selected['L']}; fitting CPU seconds={elapsed:.3f}", flush=True)
    return selected


def plot_style():
    # Set the serif fonts, sizes and colors of the figures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'mathtext.fontset':'stix','font.size':14,'axes.titlesize':18,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'figure.titlesize':20,'axes.unicode_minus':False})
    return plt

def save_figure(fig,name):
    # Save the figures
    (ROOT/'figures').mkdir(exist_ok=True)
    padding=.6 if name.startswith('3d_') else .1
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=padding)
    fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight',pad_inches=padding)
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=padding)

