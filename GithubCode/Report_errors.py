"""Report CDF discrepancies, trajectory MSEs.

Trajectory comparisons use values at the same time points. The CDF
comparison uses each known synthetic generating distribution."""
import csv
import json
import numpy as np
from scipy.stats import gamma
from study import ROOT, MAIN, CANDIDATES, parts, load_fit, target_cdf, target_sf, dump
from dynamics import population_integral, epidemic_integral, SENSITIVITY
from em_core import cdf, save_csv

OUT = ROOT/'results'/'quantitative_errors'


def distribution_error(name, fit):
    # Cover both target and fitted tails. Each CDF exceeds 1-1e-8 at the
    # endpoint, so their absolute difference beyond it is at most 1e-8.
    components = parts(name)
    upper = max(max(d.ppf(1-1e-8) for _,d in components),
                gamma.ppf(1-1e-8,fit['L'],scale=1/fit['rate']))
    # Use log-spaced points to resolve the origin and long lognormal tail.
    # Retain the same grid as a subset when doubling its density.
    errors=[]
    for n in (20001,40001):
        t=np.expm1(np.linspace(0,np.log1p(upper),n))
        truth=sum(weight*d.cdf(t) for weight,d in components)
        fitted=cdf(t,np.asarray(fit['weights']),fit['rate'])
        errors.append(float(np.max(abs(fitted-truth))))
    # Normal-containing targets were sampled on the real line in study.py.
    return dict(dataset=name,L=fit['L'],CDF_max_grid_error=errors[-1],
        grid_refinement_change=abs(errors[1]-errors[0]),grid_points=40001,
        upper=upper,tail_error_bound=1e-8,
        target_probability_nonpositive=float(sum(w*d.cdf(0) for w,d in components)))


def trajectory_error(filename, name, states):
    data=np.loadtxt(ROOT/'results'/filename,delimiter=',',skiprows=1)
    t, reference, fitted=data[:,0],data[:,1:3],data[:,3:5]
    # Scale each state by its reference maximum, this avoids division by nearly zero values along a trajectory.
    mse=np.mean((fitted-reference)**2,axis=0)
    maximum=np.max(abs(fitted-reference),axis=0)
    scale=np.max(abs(reference),axis=0)
    rows=[dict(case=name,state=state,MSE=float(mse[j]),max_absolute_error=float(maximum[j]),
        normalized_max_error=float(maximum[j]/scale[j]),
        reference_maximum=float(scale[j]),points=len(t),start=t[0],end=t[-1],
        dt=t[1]-t[0]) for j,state in enumerate(states)]
    return rows,t,reference


def grid_check(case, t, saved, refined):
    # The refined solve has twice as many intervals and the same endpoints.
    # Compare its coincident points with the numerical reference used in the
    # figure to measure the effect of refining its time grid.
    absolute=np.max(abs(refined[::2]-saved),axis=0)
    relative=absolute/np.max(abs(saved),axis=0)
    return dict(case=case,coarse_step=t[1]-t[0],fine_step=(t[1]-t[0])/2,
        max_absolute_change=float(absolute.max()),
        normalized_max_change=float(relative.max()))


def main():
    OUT.mkdir(exist_ok=True)
    distributions=[distribution_error(name,load_fit(name)) for name in MAIN]
    sensitivity=[distribution_error('sensitivity_normal',load_fit('sensitivity_normal',L))
                 for L in CANDIDATES]
    trajectories=[]; checks=[]
    for name in MAIN:
        rows,t,ref=trajectory_error(f'recomputed_Fig4_{name}.csv',name,('I','M'))
        trajectories.extend(rows)
        fine=np.linspace(t[0],t[-1],2*len(t)-1)
        values,_=population_integral(lambda z:target_cdf(name,z),
                                    lambda z:target_sf(name,z),fine)
        checks.append(grid_check(name,t,ref,values))
    sir=[]
    for s,i in [(5,15),(10,10),(15,5)]:
        case=f'S0={s},I0={i}'
        rows,t,ref=trajectory_error(f'KM_S0_{s}_I0_{i}.csv',case,('S','I'))
        sir.extend(rows)
        fine=np.linspace(t[0],t[-1],2*len(t)-1)
        values,_=epidemic_integral(lambda z:target_sf('weibull',z),fine,s,i)
        checks.append(grid_check(case,t,ref,values))
    sens_trajectories=[]
    for L in CANDIDATES:
        rows,t,ref=trajectory_error(f'Fig6b_L{L}_trajectories.csv',f'L={L}',('I','M'))
        sens_trajectories.extend(rows)
        if L==CANDIDATES[0]:
            fine=np.linspace(t[0],t[-1],2*len(t)-1)
            values,_=population_integral(lambda z:target_cdf('sensitivity_normal',z),
                                        lambda z:target_sf('sensitivity_normal',z),fine,SENSITIVITY)
            checks.append(grid_check('sensitivity',t,ref,values))
    result=dict(distribution=distributions,population=trajectories,SIR=sir,
        sensitivity_distribution=sensitivity,sensitivity_trajectory=sens_trajectories,
        reference_grid_checks=checks)
    dump(OUT/'summary.json',result)
    for name,rows in result.items(): save_csv(OUT/f'{name}.csv',rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
