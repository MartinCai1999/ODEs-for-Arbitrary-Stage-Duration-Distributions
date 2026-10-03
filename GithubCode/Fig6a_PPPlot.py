# Reproducible research code: all numerical inputs are read from fit files or generated from fixed seeds.
# See README.md for the reproduction order and numerical settings.


import numpy as np
from study import ROOT,plot_style,save_figure,target_ppf
from plots import sensitivity_fits,mixture_ppf
from dynamics import mixture_cdf
# Assign one color and hollow marker to each candidate L.
# Colors follow the order L=3,5,10,20,50,100.
COLORS = [
    (183/255, 178/255, 208/255),  
    (236/255, 166/255, 128/255),  
    (122/255, 199/255, 226/255),  
    (201/255, 138/255, 164/255),  
    (84/255, 190/255, 170/255),  
    (247/255, 223/255, 135/255)   
]

MARKERS = ['d', 'h', '*', 's', '^', 'o']

def main():
    
    
    # Evaluate both distributions at the same probability levels.
    # Invert the fitted CDF to obtain each mixture quantile.
    plt=plot_style();plt.rcParams['legend.fontsize']=10
    fig=plt.figure();p=np.linspace(.001,.999,301)
    data=target_ppf('sensitivity_normal',p)
    export={'p':p,'target_quantile':data}
    # Read the common rate and weights of each saved candidate.
    for fit,marker,color in zip(sensitivity_fits(),MARKERS,COLORS):
        x=p
        y=mixture_cdf(fit,data)
        export[f'fitted_PP_L{fit["L"]}']=y
        plt.plot(x,y,marker,markerfacecolor='None',color=color,linewidth=.5,
                 label=f'$L = {fit["L"]}, r = {fit["rate"]:.4f}$')
    # Set the axis window and save the complete numerical coordinates.
    bounds=(0,1)
    plt.plot(bounds,bounds,'--',linewidth=1,color='r',label='45-degree line')
    plt.xlim(*bounds);plt.ylim(*bounds);plt.legend()
    
    plt.xlabel(r'$\mathcal{N}(10, 1.5^2)$ cdf')
    plt.ylabel('Mixture Erlang cdf')
    plt.tight_layout();save_figure(fig,'Fig6a_PPPlot');plt.close(fig)
    # Export the coordinates for independent numerical comparison.
    np.savetxt(ROOT/'results'/'Fig6a_PP_coordinates.csv',np.column_stack(list(export.values())),
               delimiter=',',header=','.join(export),comments='')

if __name__=='__main__':main()
