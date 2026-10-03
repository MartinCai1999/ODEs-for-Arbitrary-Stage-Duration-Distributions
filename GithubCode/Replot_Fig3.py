"""Recreate the four Figure 3 panels from the supplied samples and fitted parameters."""
from importlib import import_module
import numpy as np
from study import ROOT, MAIN, load_fit


if __name__ == '__main__':
    # Each panel keeps its original plotting function and visual settings.
    for name in MAIN:
        data = np.loadtxt(ROOT/'data'/f'{name}.csv', delimiter=',', skiprows=1)
        fit = load_fit(name)
        module = import_module('Fig3_EM_'+name)
        module.plot_data_and_fit(data, fit)
