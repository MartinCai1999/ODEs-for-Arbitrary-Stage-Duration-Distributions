"""Redraw Figure 7 from saved Hunan results without repeating EM.

For a full reproduction from the workbook, run Hunan_incubation.py first.
Both commands use the same plotting function and produce the same figure.
Parameters and reference-CDF masses are read from the saved estimation outputs.
"""
import json
import numpy as np
from Hunan_incubation import OUT, plot_figure7


def main():
    # The summary contains the selected mixture and independent Weibull fit.
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    reference = np.loadtxt(OUT/'nonparametric.csv', delimiter=',', skiprows=1)
    D, Dw = plot_figure7(summary['selected'], summary['weibull'],
                         reference[:,0], reference[:,1], summary['maximum_upper'])
    # Check that the plotted CDF differences match the saved assessments.
    assert np.isclose(D, summary['CDF_distance_to_nonparametric'], atol=1e-12, rtol=0)
    assert np.isclose(Dw, summary['weibull_CDF_distance'], atol=1e-12, rtol=0)
    print(f'Maximum absolute CDF differences: mixture={D:.10f}, Weibull={Dw:.10f}')


if __name__ == '__main__':
    main()
