"""Original branch ODEs and causal trapezoid rules for the integral models."""
import time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.special import gammainc, gammaincc

POPULATION=dict(mu_I=.05,mu_M=.15,a=5.,I0=10.,M0=5.)
SENSITIVITY=dict(mu_I=.3,mu_M=.1,a=5.,I0=.1,M0=.05)

def mixture_cdf(fit,t):
    t=np.atleast_1d(t)
    return gammainc(np.arange(1,fit['L']+1),fit['rate']*t[:,None]) @ np.asarray(fit['weights'])

def mixture_sf(fit,t):
    t=np.atleast_1d(t)
    return gammaincc(np.arange(1,fit['L']+1),fit['rate']*t[:,None]) @ np.asarray(fit['weights'])

def population_ode(fit, t, parameters=POPULATION):
    """Integrate the paper's I_{k,j} branches and adult population M."""
    p = parameters
    w, r, L = np.asarray(fit['weights']), fit['rate'], fit['L']
    # Branch k contains k successive phases; flatten the branches into one array.
    start = np.r_[0, np.cumsum(np.arange(1, L))]
    end = start + np.arange(1, L+1) - 1
    size = L*(L+1)//2
    interior = np.setdiff1d(np.arange(size), start)
    def rhs(_, y):
        d = np.zeros(size+1)
        d[:-1] = -(r+p['mu_I'])*y[:-1]
        d[start] += w*p['a']*y[-1]/(1+y[-1])
        d[interior] += r*y[interior-1]
        # Every branch contributes maturation, including the shape-one branch.
        d[-1] = r*y[end].sum()-p['mu_M']*y[-1]
        return d
    initial = np.zeros(size+1)
    initial[start], initial[-1] = p['I0']*w, p['M0']
    tick = time.process_time()
    sol = solve_ivp(rhs, (t[0],t[-1]), initial, t_eval=t,
                    method='DOP853', rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.column_stack((sol.y[:-1].sum(axis=0),sol.y[-1])), time.process_time()-tick


def _population_recurrence(t,H,K,mu_M,a,I0,M0):
    # Advance in time, using only previously computed births in each convolution.
    n=len(t);h=t[1]-t[0]
    I=np.empty(n);M=np.empty(n);birth=np.empty(n)
    I[0]=I0;M[0]=M0;birth[0]=a*M0/(1+M0)
    for j in range(1,n):
        adult=.5*birth[0]*H[j]
        juvenile=.5*birth[0]*K[j]
        # Compute the interior trapezoid sum over past births.
        # Reverse the kernel lags so birth[i] is paired with kernel[j-i].
        adult += np.dot(birth[1:j], H[j-1:0:-1])
        juvenile += np.dot(birth[1:j], K[j-1:0:-1])
        # H(0)=0: the adult convolution has no unknown endpoint term.
        M[j]=M0*np.exp(-mu_M*t[j])+I0*H[j]+h*adult
        birth[j]=a*M[j]/(1+M[j])
        # K(0)=1 supplies the final endpoint of the juvenile trapezoid sum.
        I[j]=I0*K[j]+h*(juvenile+.5*birth[j]*K[0])
    return np.column_stack((I,M))

def population_integral(F,survival,t,parameters=POPULATION):
    """Advance the population integral equations by causal trapezoid quadrature.
    """
    p = parameters
    tic = time.process_time()
    d = p['mu_I']-p['mu_M']
    c = np.asarray(F(t))
    # K gives surviving immatures; H gives those matured and still alive.
    K = np.asarray(survival(t))*np.exp(-p['mu_I']*t)
    H = (np.exp(-p['mu_I']*t)*c
         + d*np.exp(-p['mu_M']*t)
         * cumulative_trapezoid(np.exp(-d*t)*c,t,initial=0))
    H[0]=0.
    if np.min(H)<-1e-7:raise RuntimeError('Negative mature kernel: refine numerical quadrature.')
    result=_population_recurrence(t,H,K,p['mu_M'],p['a'],p['I0'],p['M0'])
    return result,time.process_time()-tic

def epidemic_ode(fit, t, S0, I0, beta=.1):
    """Use the original infectious branches and susceptible population S."""
    w, r, L = np.asarray(fit['weights']), fit['rate'], fit['L']
    start = np.r_[0, np.cumsum(np.arange(1, L))]
    size = L*(L+1)//2
    interior = np.setdiff1d(np.arange(size), start)
    def rhs(_, y):
        incidence = beta*y[-1]*y[:-1].sum()
        d = np.zeros(size+1)
        d[:-1] = -r*y[:-1]
        d[start] += w*incidence
        d[interior] += r*y[interior-1]
        d[-1] = -incidence
        return d
    initial = np.zeros(size+1)
    initial[start], initial[-1] = I0*w, S0
    tick = time.process_time()
    sol = solve_ivp(rhs, (t[0],t[-1]), initial, t_eval=t,
                    method='DOP853', rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.column_stack((sol.y[-1],sol.y[:-1].sum(axis=0))), time.process_time()-tick


def _epidemic_recurrence(t,P,S0,I0,beta):
    # Couple the trapezoid updates of susceptibles and the infectious convolution.
    n=len(t);h=t[1]-t[0];S=np.empty(n);I=np.empty(n);b=np.empty(n)
    S[0]=S0;I[0]=I0;b[0]=beta*S0*I0
    for j in range(1,n):
        A=S[j-1]-.5*h*b[j-1]
        C=I0*P[j]+.5*h*b[0]*P[j]
        # Evaluate the past-history sum with ordinary NumPy array algebra.
        C += h*np.dot(b[1:j], P[j-1:0:-1])
        # Solve b=beta*(A-h*b/2)*(C+h*b/2), using its stable positive root.
        z=1-.5*beta*h*(A-C)
        if A<0:raise ValueError('Time step too large for nonnegative susceptible predictor')
        b[j]=2*beta*A*C/(z+np.sqrt(z*z+beta*beta*h*h*A*C))
        S[j]=A-.5*h*b[j];I[j]=C+.5*h*b[j]
    return np.column_stack((S,I))

def epidemic_integral(survival,t,S0,I0,beta=.1):
    tic=time.process_time()
    solution=_epidemic_recurrence(t,np.asarray(survival(t)),S0,I0,beta)
    return solution,time.process_time()-tic
