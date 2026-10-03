"""Fit common-rate Erlang mixtures with fixed shapes 1,...,L.

The E-step computes posterior route counts in log scale; the M-step updates
weights and the shared rate. Six starts are compared using observed likelihood."""
import csv

import numpy as np
from numba import njit
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma

# Validate the data
def validate_data(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError('Expected at least two finite, strictly positive observations in one dimension.')
    return x

# Compute log mixture Erlang density
def log_density(x, weights, rate):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = np.arange(1, len(weights) + 1)
    out = np.full(len(x), -np.inf)
    with np.errstate(divide='ignore'):
        lw = np.log(weights)
    positive = x > 0
    xp = x[positive, None]
    out[positive] = logsumexp(lw + k*np.log(rate) + (k-1)*np.log(xp)
                                 - rate*xp - gammaln(k), axis=1)
    if weights[0] > 0:
        out[x == 0] = np.log(weights[0]*rate)
    return out

# Compute mixture Erlang cdf
def cdf(x, weights, rate):
    return gamma.cdf(np.asarray(x)[:, None], a=np.arange(1, len(weights)+1),
                     scale=1/rate) @ weights

def fit_em(x, L, rate_init, weights_init=None, max_iter=6000,
           tol_per_sample=1e-8, rate_tol=1e-6, weight_tol=1e-5, patience=5):
    """EM algorithm; stop on observed likelihood AND parameter changes."""
    x = validate_data(x)
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError('L must be a positive integer.')
    if not np.isfinite(rate_init) or rate_init <= 0 or max_iter < 1:
        raise ValueError('Invalid initial rate or iteration limit.')
    n = len(x)

    k = np.arange(1, L+1)

    # Precompute the part of each log density that does not depend on weights/rate.
    base = np.log(x[:, None])*(k-1) - gammaln(k)
    w = np.full(L, 1/L) if weights_init is None else np.asarray(weights_init, float).copy()
    if w.shape != (L,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid initial weights.')
    w /= w.sum()
    r = float(rate_init)
    with np.errstate(divide='ignore'):
        lw = np.log(w)

    def evaluate(logw, rate):
        # E-step: normalize component densities to posterior probabilities.
        # -rate*x cancels out of the responsibility ratios, but not out of log L.
        return fast_e_step(base, x, logw, rate)

    ll, q = evaluate(lw, r)
    stable = 0
    converged = False
    for iteration in range(1, max_iter+1):
        # M-step: average the posterior weights, then update the common rate.
        w_new = q / n
        w_new /= w_new.sum()
        r_new = float(np.dot(k, w_new)/x.mean())
        with np.errstate(divide='ignore'):
            lw_new = np.log(w_new)
        ll_new, q_new = evaluate(lw_new, r_new)
        gain = ll_new - ll
        if gain < -1e-8:
            raise RuntimeError(f'Observed likelihood decreased by {gain}; inspect numerical arithmetic.')
        rel_rate_change = abs(r_new-r)/r
        max_weight_change = float(np.max(np.abs(w_new-w)))
        small = (abs(gain)/n < tol_per_sample and rel_rate_change < rate_tol
                 and max_weight_change < weight_tol)
        stable = stable+1 if small else 0
        w, r, lw, ll, q = w_new, r_new, lw_new, ll_new, q_new
        if stable >= patience:
            converged = True
            break
    # There are L-1 independent weights and one rate, hence L BIC parameters.
    return dict(L=int(L), weights=w, rate=r, loglik=ll,
                BIC=L*np.log(n)-2*ll, converged=converged, iterations=iteration)


def initializations(x, L):
    q95 = np.quantile(x, 0.95)
    # Start 1 uses the empirical 80th percentile and uniform weights.
    yield 8/np.quantile(x, 0.8), np.full(L, 1/L)
    # Scale the grid to the data: k/r is the mean of the k-th component.
    # Starts 2--5 span four common rates with uniform weights.
    for factor in (0.5, 1.0, 1.5, 2.0):
        r = factor*L/q95
        yield r, np.full(L, 1/L)
    r = L/q95
    # Start 6 allocates observations to rate-scaled shape bins.
    # Mix 95% empirical bin weights with 5% uniform mass to start every branch.
    k_init = np.clip(np.ceil(r*x).astype(int), 1, L)
    counts = np.bincount(k_init, minlength=L+1)[1:].astype(float)
    w = counts/counts.sum()
    w = 0.95*w + 0.05/L  
    yield r, w


def fit_multistart(x, L):
    """Try six deterministic starts and retain the largest observed likelihood."""
    best = None
    for index, (rate, weights) in enumerate(initializations(x, L), 1):
        fit = fit_em(x, L, rate, weights)
        print(f"L={L}, start={index}: iterations={fit['iterations']}, converged={fit['converged']}", flush=True)
        if best is None or fit['loglik'] > best['loglik']:
            best = fit
    # Continue the best run if the initial iteration allowance was insufficient.
    if not best['converged']:
        best = fit_em(x, L, best['rate'], best['weights'], max_iter=20000)
    if not best['converged']:
        raise RuntimeError(f'L={L}: increase the iteration allowance before reporting BIC.')
    print(f"L={L}: log-likelihood={best['loglik']:.6f}, BIC={best['BIC']:.6f}", flush=True)
    return best


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)



@njit(cache=True)
def fast_e_step(base, x, logw, rate):
    # Accumulate the observed log-likelihood and posterior component counts.
    # One temporary row stores the L responsibilities for each observation.
    # Subtract the row maximum before exponentiation to prevent overflow.
    n, L = base.shape
    counts = np.zeros(L)
    row = np.empty(L)
    add = np.empty(L)
    for k in range(L):
        add[k] = (k+1)*np.log(rate) + logw[k]
    ll = 0.0
    for j in range(n):
        maximum = -np.inf
        for k in range(L):
            row[k] = base[j,k]+add[k]
            maximum = max(maximum,row[k])
        total = 0.0
        for k in range(L):
            row[k] = np.exp(row[k]-maximum)
            total += row[k]
        ll += maximum+np.log(total)-rate*x[j]
        for k in range(L):
            counts[k] += row[k]/total
    return ll, counts
