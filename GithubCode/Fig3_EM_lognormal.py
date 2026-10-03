"""Generate 10,000 observations, fit fixed shapes 1,...,L, and plot Figure 3."""
import numpy as np
from study import sample, fit_data, plot_style, save_figure
from em_core import log_density

DATASET = 'lognormal'

def plot_data_and_fit(data,fit):
    # plot Figure 3
    plt=plot_style();fig,ax=plt.subplots()
    bin_edges=np.histogram_bin_edges(data,bins='fd')
    ax.hist(data,bins=bin_edges,density=True,color='#99b9e9',alpha=.65,label='Data')
    x=np.linspace(0,np.quantile(data,.995),2000)
    pdf=np.exp(log_density(x,fit['weights'],fit['rate']))
    ax.plot(x,pdf,color='#e3716e',label='Fitted Mixture Erlang')
    ax.set(xlim=(0,x[-1]),xlabel='x',ylabel='Density');ax.legend(fontsize=10)
    fig.tight_layout();save_figure(fig,'Fig3_'+DATASET);plt.close(fig)

if __name__ == '__main__':
    # Step 1: generate the original target law with its fixed random seed.
    data = sample(DATASET)
    # Step 2: fit all six L values using the same data and six EM starts each.
    # The BIC penalty is L*log(n): L-1 independent weights plus one rate.
    fit = fit_data(DATASET, data)
    # Step 3: show data as histogram.
    plot_data_and_fit(data, fit)
