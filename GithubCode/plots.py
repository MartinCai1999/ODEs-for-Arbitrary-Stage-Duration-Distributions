"""Small numerical helpers shared by the original figure scripts."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc
from study import ROOT, CANDIDATES, load_fit, target_sf
from dynamics import epidemic_ode, epidemic_integral

def km_data():
    """Compute all curves directly from the selected Weibull parameters."""
    fit = load_fit('weibull')
    t = np.linspace(0, 5, 2001)
    I0 = np.array([15., 10., 5.])
    S0 = 20-I0
    target, ode = [], []
    for s, i in zip(S0, I0):
        reference = epidemic_integral(lambda z: target_sf('weibull', z), t, s, i)[0]
        approximation = epidemic_ode(fit, t, s, i)[0]
        target.append(reference)
        ode.append(approximation)
        np.savetxt(ROOT/'results'/f'KM_S0_{s:g}_I0_{i:g}.csv',
                   np.column_stack((t, reference, approximation)), delimiter=',',
                   header='t,S_target,I_target,S_ODE,I_ODE', comments='')
    return dict(t=t, S0=S0, I0=I0, target=np.array(target), ode=np.array(ode)), fit


def sensitivity_fits():
    # Read the six actual fits, in increasing order of L.
    return [load_fit('sensitivity_normal', L) for L in CANDIDATES]


def mixture_ppf(fit,p):
    """Invert the complete fitted CDF at fixed probabilities.

    Expand the upper bracket until it contains all requested probabilities.
    """
    k=np.arange(1,fit['L']+1);w=fit['weights'];r=fit['rate']
    def F(x):return float(np.dot(gammainc(k,r*x),w))
    upper=max(1.,2*fit['L']/r)
    while F(upper)<max(p):upper*=2
    return np.array([brentq(lambda x:F(x)-prob,0,upper) for prob in p])
