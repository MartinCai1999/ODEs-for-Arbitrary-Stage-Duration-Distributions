"""Fit the normal target used in the Figure 6 sensitivity experiment."""
from study import sample, fit_data

if __name__ == '__main__':
    # Generate the sensitivity sample of 10,000 N(10, 1.5^2) observations.
    data = sample('sensitivity_normal')
    # Apply the same six starts, candidate L values, and BIC as Figure 3.
    fit_data('sensitivity_normal', data)
