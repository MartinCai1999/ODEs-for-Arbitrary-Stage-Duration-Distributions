"""Forward solvers for the population-model efficiency experiment.

The upwind and Legendre--Galerkin transport formulations adapt Peterson and
Adhikari (2021). 
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from scipy.sparse import coo_matrix
from numpy.polynomial.legendre import legvander
from dynamics import POPULATION, mixture_sf, mixture_cdf, population_integral
from Volterra_methods import kernels


def cohort_flux(t, weights, rate, mortality):
    """Evaluate the mixture density by the Erlang recurrence."""
    term = rate*np.exp(-(rate+mortality)*t)
    total = weights[0]*term
    for k in range(1,len(weights)):
        term *= rate*t/k
        total += weights[k]*term
    return total


def upwind(fit, times, h):
    """First-order upwind transport with right-rectangle age quadrature."""
    p = POPULATION
    age = np.arange(1,round(times[-1]/h)+1)*h
    K,q = kernels(fit,age)
    w,r = np.asarray(fit['weights']),fit['rate']
    def rhs(t,y):
        out=np.empty_like(y)
        birth=p['a']*y[-1]/(1+y[-1])
        out[0]=(birth-y[0])/h
        out[1:-1]=(y[:-2]-y[1:-1])/h
        out[-1]=p['I0']*cohort_flux(t,w,r,p['mu_I'])+h*(q@y[:-1])-p['mu_M']*y[-1]
        return out
    n=len(age)
    # Fine upwind grids are stiff; BDF uses this sparse analytic Jacobian.
    rows=np.r_[np.arange(n),np.arange(1,n),0,np.full(n+1,n)]
    cols=np.r_[np.arange(n),np.arange(n-1),n,np.arange(n+1)]
    data=np.r_[np.full(n,-1/h),np.full(n-1,1/h),1.,h*q,-p['mu_M']]
    base=coo_matrix((data,(rows,cols)),shape=(n+1,n+1)).tocsc()
    def jac(t,y):
        matrix=base.copy()
        matrix[0,n]=p['a']/(1+y[-1])**2/h
        return matrix
    initial=np.zeros(n+1);initial[-1]=p['M0']
    sol=solve_ivp(rhs,(0,times[-1]),initial,t_eval=times,method='BDF',jac=jac,rtol=1e-7,atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)
    K0=mixture_sf(fit,times)*np.exp(-p['mu_I']*times)
    return np.column_stack((p['I0']*K0+h*(K@sol.y[:-1]),sol.y[-1]))


def branch_ode(fit, times, rtol=1e-6, method='DOP853'):
    """Use precisely the manuscript's branching equations, with a chosen tolerance."""
    p = POPULATION
    w, r, L = np.asarray(fit['weights']), fit['rate'], fit['L']
    start = np.r_[0, np.cumsum(np.arange(1, L))]
    end = start + np.arange(1, L+1)-1
    size = L*(L+1)//2
    interior = np.setdiff1d(np.arange(size), start)
    def rhs(t, y):
        dy = np.zeros(size+1)
        dy[:-1] = -(r+p['mu_I'])*y[:-1]
        dy[start] += w*p['a']*y[-1]/(1+y[-1])
        dy[interior] += r*y[interior-1]
        dy[-1] = r*y[end].sum()-p['mu_M']*y[-1]
        return dy
    initial = np.zeros(size+1)
    initial[start], initial[-1] = p['I0']*w, p['M0']
    sol = solve_ivp(rhs, (0, times[-1]), initial, t_eval=times,
                    method=method, rtol=rtol, atol=rtol*.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.column_stack((sol.y[:-1].sum(axis=0), sol.y[-1]))


def galerkin(fit, times, degree, quadrature_factor=1):
    """Legendre tau/Galerkin transport approximation on ages [0,T].

    The highest coefficient enforces u(t,0)=B(M(t)), as in the source's boundary elimination.
    Initially its interior coefficients are zero and its boundary is B(M0).
    """
    p, T, N = POPULATION, times[-1], int(degree)
    # Resolve kernel projection integrals independently of the ODE tolerances.
    x, weights = roots_legendre(quadrature_factor*max(256, 4*(N+1)))
    ages = T*(x+1)/2
    basis = legvander(x, N)
    K, q = kernels(fit, ages)
    Kcoef = (weights*K*T/2)@basis
    qcoef = (weights*q*T/2)@basis
    signs = (-1.)**np.arange(N+1)
    row = np.arange(N)[:,None]
    col = np.arange(N+1)[None,:]
    derivative = 2/T*(2*row+1)*((col>row)&((col-row)%2==1))
    w,r = np.asarray(fit['weights']),fit['rate']
    def rhs(t, y):
        c = np.empty(N+1)
        c[:N] = y[:N]
        birth = p['a']*y[-1]/(1+y[-1])
        c[N] = signs[N]*(birth-signs[:N]@c[:N])
        dy = np.empty(N+1)
        # Apply the precomputed Legendre derivative matrix to the coefficients.
        dy[:N] = -derivative@c
        dy[-1] = p['I0']*cohort_flux(t,w,r,p['mu_I'])+qcoef@c-p['mu_M']*y[-1]
        return dy
    initial = np.zeros(N+1)
    initial[-1] = p['M0']
    sol = solve_ivp(rhs, (0,T), initial, t_eval=times,
                    method='DOP853', rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    coeff = np.empty((N+1, len(times)))
    coeff[:N] = sol.y[:N]
    birth = p['a']*sol.y[-1]/(1+sol.y[-1])
    coeff[N] = signs[N]*(birth-signs[:N]@coeff[:N])
    K0 = mixture_sf(fit, times)*np.exp(-p['mu_I']*times)
    return np.column_stack((p['I0']*K0+Kcoef@coeff, sol.y[-1]))


def direct_trapezoid(fit, times, h):
    """Direct composite-trapezoid solution of the population Volterra equations.

    The existing dynamics.py implementation uses a CDF-based adult kernel.
    Kernel construction and interpolation to common output times are included.
    """
    grid = np.linspace(0, times[-1], round(times[-1]/h)+1)
    values, _ = population_integral(lambda t: mixture_cdf(fit, t),
                                    lambda t: mixture_sf(fit, t), grid)
    return np.column_stack([np.interp(times, grid, values[:, j]) for j in range(2)])


def solve(method, fit, times, setting):
    """Return I and M at common output times, including all numerical setup."""
    if method == 'Trapezoid':
        return direct_trapezoid(fit, times, setting)
    if method == 'ODE':
        return branch_ode(fit, times, setting)
    if method == 'MOL':
        return upwind(fit, times, setting)
    if method == 'Galerkin':
        return galerkin(fit, times, setting)
    raise ValueError(method)
