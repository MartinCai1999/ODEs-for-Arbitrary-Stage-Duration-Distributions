# Reproducible research code: all numerical inputs are read from fit files or generated from fixed seeds.
# See README.md for the reproduction order and numerical settings.


"""Research module. See README.md for inputs, outputs, and numerical conventions."""
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from study import plot_style,save_figure
from plots import km_data
plt=plot_style();plt.rcParams['font.size']=11
# Compute the trajectories using the current fitted parameters.
data,fit=km_data()
# These six positions are display offsets.
# Each adjacent pair represents the same real initial condition.
batch_sizes=[55,50,30,25,5,0]  
epochs=np.linspace(0,5,600).tolist()
show_epochs=[epochs[0],epochs[149],epochs[599]]
# Interpolate the freshly computed curves onto the original 600 display times.
accuracies=[]
column=1

# Order by the state displayed here; the S panel therefore reverses these cases.
order=np.argsort(data['I0'])[::-1]
for case in order:
    for method in ('target','ode'):
        accuracies.append(np.interp(epochs,data['t'],data[method][case,:,column]))
X, Y = np.meshgrid(epochs, batch_sizes)  
Z = np.array(accuracies)  

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('white')  
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')         

ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

norm = mcolors.Normalize(vmin=min(batch_sizes), vmax=max(batch_sizes))  
cmap = cm.viridis  

line_width = 1.0  

# filled planes, colormap, and paired annotations.
for i, batch_size in enumerate(batch_sizes): 
   
    ax.plot(epochs, [batch_size]*len(epochs), accuracies[i], label=f'Batch Size {batch_size}', color=cmap(norm(batch_size)), linewidth=line_width)

    ax.plot(epochs, [batch_size]*len(epochs), np.zeros(len(epochs)), color=cmap(norm(batch_size)), linewidth=line_width)
    
    verts = [(x, batch_size, z) for x, z in zip(epochs, accuracies[i])]  
    verts += [(x, batch_size, 0) for x in epochs[::-1]]  

    if (i % 2 == 0):
        poly = Poly3DCollection([verts], color='#99b9e9', alpha=0.3)  
    else:
        poly = Poly3DCollection([verts], color='#e3716e', alpha=0.3)  
    ax.add_collection3d(poly)  


p1, p2 = None, None 

for epoch in show_epochs:  
    for i in range(0, len(batch_sizes), 2):
        
        idx1 = i
        batch_size1 = batch_sizes[idx1]
        accuracy1 = accuracies[idx1][epochs.index(epoch)]

        p1 = ax.scatter(epoch, batch_size1, accuracy1, color='#99b9e9', s=20, edgecolor='black', zorder=10)

        ax.text(epoch, batch_size1, accuracy1 + 0.5, f'{accuracy1:.2f}', color='black', zorder=10)

        if i + 1 < len(batch_sizes):
            idx2 = i + 1
            batch_size2 = batch_sizes[idx2]
            accuracy2 = accuracies[idx2][epochs.index(epoch)]

            p2 = ax.scatter(epoch, batch_size2, accuracy2, color='#e3716e', s=20, edgecolor='black', zorder=10)
            ax.text(epoch, batch_size2, accuracy2 + 0.5, f'{accuracy2:.2f}', color='black', zorder=10)
            
            data_list_x = [epoch, epoch]                  
            data_list_y = [batch_size1, batch_size2]       
            data_list_z = [accuracy1, accuracy2]           
            
            ax.plot(data_list_x, data_list_y, data_list_z, color='black', linewidth=line_width, linestyle='--')

plt.yticks([]) 

if p1 and p2:
    plt.legend([p1, p2], ['I(t) KM', 'I(t) ODEs'], loc='upper right', bbox_to_anchor=(0.82, 0.78))

ax.set_xlabel('t')
ax.set_ylabel('Different Initial Value of I(t)', labelpad=-5)
ax.set_zlabel('I(t)')

ax.grid(False)

ax.view_init(elev=30, azim=225)

for text in ax.texts:
    text.set_zorder(999)


save_figure(fig,'3d_KM_It');plt.close(fig)  
