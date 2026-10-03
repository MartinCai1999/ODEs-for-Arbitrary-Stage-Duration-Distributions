"""Evaluate the occupancy and maturation kernels for the population model."""
import numpy as np
from em_core import log_density
from dynamics import mixture_sf, POPULATION


def kernels(fit, t, parameters=POPULATION):
    # K is the surviving immature fraction; q is its maturation flux density.
    survival = np.exp(-parameters['mu_I']*t)
    K = mixture_sf(fit,t)*survival
    q = np.exp(log_density(t,np.asarray(fit['weights']),fit['rate']))*survival
    return K,q

