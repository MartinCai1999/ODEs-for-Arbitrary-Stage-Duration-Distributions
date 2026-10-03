# Reproducible research code: all numerical inputs are read from fit files or generated from fixed seeds.
# See README.md for the reproduction order and numerical settings.


import numpy as np
from study import ROOT,plot_style,save_figure
from plots import sensitivity_fits
# Assign one color and hollow marker to each candidate L.
COLORS = [
    (183/255, 178/255, 208/255),  
    (236/255, 166/255, 128/255),  
    (122/255, 199/255, 226/255),  
    (201/255, 138/255, 164/255),  
    (84/255, 190/255, 170/255),  
    (247/255, 223/255, 135/255)   
]

MARKERS = ['d', 'h', '*', 's', '^', 'o']

from study import target_cdf,target_sf
from dynamics import population_ode,population_integral,SENSITIVITY

def main():
    """Plot sorted solution values using the original pastel hollow symbols.

    Keep the sensitivity parameters separate from the Fig4 parameters.
    The exported trajectories provide values at matching times for error calculations.
    The full trajectories are supplied in results/Fig6b_L*_trajectories.csv.
    """
    
    
    plt=plot_style();plt.rcParams['legend.fontsize']=10
    t=np.linspace(0,60,6001)
    reference,_=population_integral(lambda z:target_cdf('sensitivity_normal',z),
                                  lambda z:target_sf('sensitivity_normal',z),t,SENSITIVITY)
    fits=sensitivity_fits()
    solutions=[population_ode(f,t,SENSITIVITY)[0] for f in fits]
    for fit,ode in zip(fits,solutions):
        # Save chronological trajectories for comparisons at matching times.
        np.savetxt(ROOT/'results'/f'Fig6b_L{fit["L"]}_trajectories.csv',np.column_stack((t,reference,ode)),
                   delimiter=',',header='t,I_target,M_target,I_ODE,M_ODE',comments='')
    for col,state,upper in ((0,'I',5),(1,'M',.5)):
        fig=plt.figure()
        for fit,ode,marker,color in zip(fits,solutions,MARKERS,COLORS):
            
            plt.plot(np.sort(reference[:,col])[::25],np.sort(ode[:,col])[::25],marker,
                     markerfacecolor='None',color=color,linewidth=.5,
                     label=f'$L = {fit["L"]}, r = {fit["rate"]:.4f}$')
        low,high=np.min(reference[:,col]),np.max(reference[:,col])
        plt.plot([low,high],[low,high],'--',color='r',linewidth=1,label='45-degree line')
        plt.xlabel(f'Solutions of {state}(t) by Volterra integral')
        plt.ylabel(f'Solutions of {state}(t) by ODEs')
        plt.xlim(0,upper);plt.ylim(0,upper);plt.legend();plt.tight_layout()
        save_figure(fig,'Fig6b_'+state);plt.close(fig)

if __name__=='__main__':main()
