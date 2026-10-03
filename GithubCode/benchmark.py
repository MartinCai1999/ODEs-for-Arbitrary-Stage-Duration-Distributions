"""Time the four forward solvers in Experiment 6 using one fitted kernel.

For each distribution, refine each method in a prescribed order and compare
its trajectories with an independently checked reference. Five CPU-time
batches determine the median cost per solve at each tested setting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, platform, json, csv
import numpy as np
import scipy
from study import ROOT, MAIN, load_fit, dump
from dynamics import population_integral, mixture_cdf, mixture_sf
from Efficiency_methods import solve, branch_ode, galerkin, cohort_flux

OUT = ROOT/'results'/'benchmark'
OUT.mkdir(exist_ok=True)
TIMES = np.linspace(0,60,601)
SETTINGS = {'ODE':(1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9),
            'Trapezoid':(.2,.1,.05,.025,.0125,.00625,.003125),
            'Galerkin':(8,16,32,64,128,256),
            # The finest upwind mesh contains 9600 age states.
            'MOL':(.1,.05,.025,.0125,.00625)}
THRESHOLDS = (1e-3,1e-5)
REPEATS = 5


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    # Optional arguments allow a short pilot on one distribution or one method.
    names = (sys.argv[1],) if len(sys.argv)>1 else MAIN
    methods = (sys.argv[2],) if len(sys.argv)>2 else tuple(SETTINGS)
    for name in names:
        # Supply the same fitted duration law to all four forward solvers.
        fit=load_fit(name)
        # Check a high-accuracy DOP853 reference against RK45 and quadrature.
        reference=branch_ode(fit,TIMES,1e-11,'DOP853')
        check=branch_ode(fit,TIMES,1e-11,'RK45')
        scale=np.max(np.abs(reference),axis=0)
        grid=np.linspace(0,60,19201)
        integ,_=population_integral(lambda t:mixture_cdf(fit,t),
                                    lambda t:mixture_sf(fit,t),grid)
        interp=np.column_stack([np.interp(TIMES,grid,integ[:,j]) for j in range(2)])
        dump(OUT/f'{name}_reference.json',dict(
            independent_RK_relative_error=float(np.max(np.abs(check-reference)/scale)),
            trapezoid_h=.003125,
            independent_trapezoid_relative_error=float(np.max(np.abs(interp-reference)/scale))))
        for method in methods:
            rows=[]
            for setting in SETTINGS[method]:
                values=solve(method,fit,TIMES,setting)
                err=float(np.max(np.abs(values-reference)/scale))
                # Calibrate batches to at least 0.2 process CPU seconds.
                # Average over the batch to resolve short individual solves.
                batch_size=1
                while True:
                    start=time.process_time()
                    for _ in range(batch_size):
                        values=solve(method,fit,TIMES,setting)
                    elapsed=time.process_time()-start
                    if elapsed>=.2:
                        break
                    batch_size*=2
                cpu=[]
                for repeat in range(REPEATS):
                    start=time.process_time()
                    for _ in range(batch_size):
                        values=solve(method,fit,TIMES,setting)
                    cpu.append((time.process_time()-start)/batch_size)
                row=dict(dataset=name,L=fit['L'],method=method,setting=setting,
                         relative_max_error=err,cpu_seconds=float(np.median(cpu)),
                         cpu_min=min(cpu),cpu_max=max(cpu),repeats=REPEATS,
                         solves_per_batch=batch_size,total_timed_solves=REPEATS*batch_size,
                         cpu_batch_means=cpu)
                rows.append(row)
                dump(OUT/f'{name}_{method}_raw.json',rows)
                print(row,flush=True)
                # Finish once the stricter target passes; otherwise refine
                # through the prescribed list and record the attained error.
                if err <= min(THRESHOLDS):
                    break
            write_csv(OUT/f'{name}_{method}.csv',rows)
    dump(OUT/'environment.json',dict(platform=platform.platform(),python=platform.python_version(),
         numpy=np.__version__,scipy=scipy.__version__,processor=platform.processor(),
         threads=1,clock='time.process_time CPU seconds per complete solve',
         repeats=REPEATS,batch_minimum_cpu_seconds=.2,backend='NumPy/SciPy',output_points=len(TIMES),time_interval=[0,60],
         thresholds=THRESHOLDS,settings=SETTINGS,
         fitting_note='Fitting CPU cost is read from selected.json and reported separately; fitting and forward-solution costs have separate timing boundaries.'))


if __name__=='__main__':
    main()
