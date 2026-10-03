# Reproducible research code: all numerical inputs are read from fit files or generated from fixed seeds.
# See README.md for the reproduction order and numerical settings.


"""Research module. See README.md for inputs, outputs, and numerical conventions."""
import numpy as np
from matplotlib.patches import ConnectionPatch
from study import ROOT,MAIN,load_fit,target_cdf,target_sf,plot_style,save_figure
from dynamics import population_ode,population_integral
plt=plot_style()
def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.2,y_ratio=1):
    # plot setting
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    
    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom-0.1,ylim_bottom-0.1,ylim_top+0.1,ylim_top+0.1,ylim_bottom-0.1],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)

def draw(name,fit,suffix=''):
    """Compare target-kernel Volterra solutions with the selected fitted ODE.

    Model parameters remain in dynamics.POPULATION. The two methods use the
    same output times, but solve_ivp chooses its own adaptive internal steps.
    """
    
    t_values=np.linspace(0,60,6001);t1_values=t_values
    ode,_=population_ode(fit,t_values)
    reference,_=population_integral(lambda t:target_cdf(name,t),lambda t:target_sf(name,t),t_values)
    IODE,MODE=ode.T;y,y1=reference.T
    # Persist the actual arrays used in this figure, with explicit column names.
    np.savetxt(ROOT/'results'/f'recomputed_Fig4_{name}{suffix}.csv',
               np.column_stack((t_values,reference,ode)),delimiter=',',
               header='t,I_target,M_target,I_ODE,M_ODE',comments='')
    fig, ax = plt.subplots(1,1,figsize=(8, 6))

    plt.plot(t1_values, y1, color='#99b9e9', linewidth=2, label="M(t) Volterra")
    plt.plot(t_values, MODE, color='#e3716e', linewidth=2, label="M(t) ODEs")
    plt.plot(t1_values, y, color='#99b9e9', linestyle='--', linewidth=2, label="I(t) Volterra")
    plt.plot(t_values, IODE, color='#e3716e', linestyle='--', linewidth=2, label="I(t) ODEs")

    plt.xlabel("Time t")
    plt.ylabel("Solutions")
    plt.legend(loc='upper left')

    axins = ax.inset_axes((0.5, 0.3, 0.4, 0.3))

    axins.plot(t1_values, y1, color='#99b9e9', linewidth=2, label="M(t) Volterra")
    axins.plot(t_values, MODE, color='#e3716e', linewidth=2, label="M(t) ODEs")

    zone_and_linked(ax, axins, 4000, 4200, t_values , [y1 ,MODE], "bottom")

    save_figure(fig,'Fig4_'+name+suffix);plt.close(fig)

if __name__=='__main__':
    for name in MAIN:
        draw(name,load_fit(name))
