# ODEs for arbitrary stage duration distributions

Research code accompanying the manuscript by Cihan Cai, Qiong Li and Yijun Lou，ODEs-for-Arbitrary-Stage-Duration-Distributions.

## Installation

Tested with Python 3.13.12, NumPy 2.4.3, SciPy 1.17.1, Matplotlib 3.10.8 and Numba 0.64.0. Install the tested versions from this directory:

```text
python -m pip install -r requirements.txt
```

For comparable CPU measurements, set numerical libraries to one thread before starting Python. In PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

Scripts locate data relative to their own directory. Run the commands below from this `code` directory. Outputs are saved in `results/` and `figures/`; rerunning a script replaces its corresponding outputs. Keep a copy of the supplied results when comparing a new run with the manuscript.

## Experiments and entry scripts

| Manuscript item | Script | Output |
|---|---|---|
| Experiment 1, Figure 3(a) | `Fig3_EM_lognormal.py` | Lognormal sample, six candidate fits, histogram and fitted PDF |
| Experiment 1, Figure 3(b) | `Fig3_EM_weibull.py` | Weibull sample, six candidate fits, histogram and fitted PDF |
| Experiment 1, Figure 3(c) | `Fig3_EM_normal_mixture.py` | Normal-mixture sample and fits |
| Experiment 1, Figure 3(d) | `Fig3_EM_normal_lognormal.py` | Normal–lognormal sample and fits |
| Experiment 2, Figure 4 | `Fig4_ODE_Vol.py` | Four population-trajectory comparisons |
| Experiment 3, Figure 5(a,b) | `3d_KM_St.py`, `3d_KM_It.py` | Susceptible and infected trajectories for three initial conditions |
| Experiment 4, Table 1 | `Fit_sensitivity_normal.py` | Six fits to the normal sensitivity sample |
| Experiment 4, Figure 6 | `Fig6a_QQPlot.py`, `Fig6a_PPPlot.py`, `Fig6b.py` | Distribution Q-Q/P-P plots and solution-value comparisons |
| Experiment 5, Figure 7 | `Hunan_incubation.py` | Interval-likelihood fits, independent Weibull fit and CDF comparisons |
| Figure 7 from saved fits | `Fig7_Hunan.py` | Both CDF panels |
| Experiment 6, Table 2 | `benchmark.py`, then `Summarize_benchmark.py` | Raw CPU batches, attained errors, selected settings |
| Distribution and trajectory errors | `Report_errors.py` | CDF distances, MSEs and refinement checks |

Shared routines are in `em_core.py` (EM and distribution functions), `study.py` (targets, seeds and file helpers), `dynamics.py` (ODE and integral equations), `plots.py` (quantiles and trajectories), and `Efficiency_methods.py` / `Volterra_methods.py` (efficiency solvers and kernels).

## Reproduce figures using the supplied fitted parameters

The package contains saved fitted parameters and numerical outputs. Plotting and forward-solution scripts read these files. To reconstruct Figures 3–7 without repeating parameter estimation:

```text
python Replot_Fig3.py
python Fig4_ODE_Vol.py
python 3d_KM_St.py
python 3d_KM_It.py
python Fig6a_QQPlot.py
python Fig6a_PPPlot.py
python Fig6b.py
python Fig7_Hunan.py
python Report_errors.py
```

Figures are saved as PDF, PNG and SVG. The exported trajectory CSV files retain chronological order for computing errors at matching times.

## Reproduce estimation from the input data

```text
python Fig3_EM_lognormal.py
python Fig3_EM_weibull.py
python Fig3_EM_normal_mixture.py
python Fig3_EM_normal_lognormal.py
python Fit_sensitivity_normal.py
python Hunan_incubation.py
```

Then run the plotting and reporting commands above. Synthetic samples contain 10,000 observations each, generated with seeds 9901, 9902, 9903, 9904 and 999 for lognormal, Weibull, normal mixture, normal–lognormal and sensitivity normal, respectively. Their distribution parameters are defined in `study.parts` and described in Figure 3 and Experiment 4. Supplied CSV files preserve the actual samples used. `target_sf` applies positive-support normalization to the survival function, while `target_cdf` follows the generating CDF on nonnegative times. These conventions reproduce the submitted results.

All exact-observation fits compare `L = 3, 5, 10, 20, 50, 100`, with shapes `1,...,L` and one common rate. Each candidate uses six deterministic starts: five uniform-weight starts and one shape-bin start. The latter combines 95% bin frequencies and 5% uniform weights. Stopping checks likelihood, rate and weights for five consecutive iterations. Each start allows 6,000 iterations; the best unfinished start receives up to 20,000 additional iterations. BIC uses `L` free parameters. The selected values for Figure 3 are 50, 5, 50 and 50; the sensitivity example selects 50.

The interval-data example uses the same initialization formulas applied to upper bounds, six starts per candidate, 30,000 initial iterations and the same continuation rule. It selects `L=3`; the Weibull comparator is independently estimated from the same observations. See ESM Section S3.

## CPU timing and supplementary numerical outputs

```text
python benchmark.py
python Summarize_benchmark.py
```

The four forward methods use the same fitted kernel on `[0,60]`. Errors are evaluated at 601 common times relative to a tightly solved branch ODE, checked against RK45 and direct quadrature. The first setting attaining each target, 0.001 or 0.00001, is retained. `met_tolerance` records whether the target was attained within the tested list.

CPU timing uses `time.process_time`. Each batch is calibrated to at least 0.2 CPU seconds. Five per-solve batch averages give the reported median. Timed work includes solver setup, kernel evaluation, integration and output interpolation. Reference construction, error assessment and file writing occur outside these batches.

| Supplementary result file under `results/benchmark/` | Contents |
|---|---|
| `selected_settings.csv` | Table 2 times, attained errors, target status and selected settings |
| `all_settings.csv` | Every tested setting |
| `*_raw.json` | Five CPU batch averages and batch sizes |
| `verification.json` | Kernel and quadrature refinement checks |
| `environment.json` | Hardware/software and benchmark settings |

Fitting is a separate preprocessing stage. The supplied `results/<distribution>/numba_cpu_runs.json` records the individual EM runs underlying the manuscript's fitting CPU totals. These totals sum calls to the EM routine, including the first call's compilation or cache loading, and exclude file writes. The regular fitting entry scripts time the full multistart calls, also including initialization and progress output. Their rerun costs therefore have a slightly broader boundary. Forward-solver timings are measured separately. New timings depend on the machine and execution environment.

The common fitted kernel isolates the forward-solution comparison. The shared kernel and any reusable setup can be retained for repeated calculations at fixed duration parameters.

## Data provenance and license

The duration records are the 268 observations in the `Main analysis` sheet of `data/hunan_incubation/data_Table_S3.xlsx`, released with [Hu et al. (2021)](https://doi.org/10.1038/s41467-021-21710-6). The original [repository](https://github.com/KristyWang/Cluster_Hunan), pinned commit, filenames and SHA-256 hashes are recorded in `data/hunan_incubation/provenance.json`. The package includes the source R analysis and original summary output for provenance; running this Python workflow requires only the listed Python packages. `read_intervals` extracts the numerical bounds directly from the workbook using the Python standard library.

Research software is covered by `LICENSE` (MIT). Third-party data and source materials retain their original attribution and terms, as described in `DATA_SOURCES.md`.
