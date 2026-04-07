import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import  ConnectionPatch
from scipy.special import erf
import time
start_time = time.time()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 14            
plt.rcParams['axes.titlesize'] = 18      
plt.rcParams['axes.labelsize'] = 12       
plt.rcParams['xtick.labelsize'] = 12      
plt.rcParams['ytick.labelsize'] = 12      
plt.rcParams['legend.fontsize'] = 12     
plt.rcParams['figure.titlesize'] = 20     


# weights = [0.015768686431259497,0.1006363433376985,0.4504035644891835,0.020075009120406138,0.4131163966214513] # 0.4norm(2, 0.5) + 0.6norm(7, 1) 
# k = [39,39,39,39,11] # 0.4norm(2, 0.5) + 0.6norm(7, 1) 
# r = 5.552197920173898 # 0.4norm(2, 0.5) + 0.6norm(7, 1) 

weights = [0.6199700896795225,5.93919525069453e-05,0.0001310681487559368,0.01978202927568017,0.0037111420910327843,0.0044533705092392635,0.0002405055191739501,0.00025496947852518054,0.00030073703752834326,0.00033415226392037996,0.00037793724441620576,0.000639689878441916,0.10541151928000685,0.010391197854891608,0.07418260978380144,0.011875654691304763,0.0005048315963090324,0.14595441131077636,0.0007872687078531647,0.0006374236963129481] # lognorm(0, 1) 
k = [3,79,79,17,36,36,79,79,79,79,79,79,8,36,17,36,79,8,79,79] # lognorm(0, 1)
r = 4.166260982361012 # lognorm(0, 1) 

# weights = [1.0077537651416246e-05,0.05482011228007584,5.164593589998627e-05,0.05769487096005779,8.607655983330825e-05,0.08654230644008101,0.10096602418009774,0.11538974192011558,0.06496505776226587,0.5194740864239208] # weibull(1, 1.5)
# k = [8,3,7,2,7,2,2,2,9,5] # weibull(1, 1.5) 
# r = 4.478208705415666 # weibull(1, 1.5) 

# weights = [0.01583528707530573,0.011002473572690342,0.0061171370114858755,0.022004947145380684,0.0072874494595935434,0.02637838385863196,0.010202429243431491,0.6367104964411394,0.24988649727315324,0.014574898919187087] # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)
# k = [26,19,27,19,33,22,33,7,17,33] # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)
# r = 2.0787177103356456 # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4) 

l = np.max(k)
w = [0] * l
for i in range(len(k)):
    w[k[i] - 1] += weights[i]
ll = np.count_nonzero(w)
k = list(set(k))
len0 = sum(k)

mu_I = 0.05
mu_M = 0.15 
a = 5
I0 = 10
M0 = 5

def B(s):
    return a * s / (s + 1)

def odes(t, IODE):
    dI = np.zeros_like(IODE)
    ind = 0
    for i in range(ll):
        for j in range(k[i]):
            if (j == 0):
                dI[ind] = B(IODE[len0]) * w[k[i] - 1] - (mu_I + r) * IODE[ind]
            else:
                dI[ind] = r * IODE[ind - 1] - (mu_I + r) * IODE[ind]
                if (j == k[i] - 1):
                    dI[len0] += IODE[ind]
            ind += 1
    dI[len0] *= r
    dI[len0] -= mu_M * IODE[len0]
    return dI

I00 = np.zeros(len0 + 1) 
ind = 0
for i in range(ll):
    for j in range(k[i]):
        if (j == 0):
            I00[ind] = I0 * w[k[i] - 1]
        ind += 1
I00[len0] = M0  

start = 0
end = 60
num = 6000
t_values = np.linspace(start, end, num)
t_span = (start, end)
t_eval = np.linspace(t_span[0], t_span[1], num)

solution = solve_ivp(odes, t_span, I00, t_eval=t_eval)

t = solution.t
IM = solution.y
MODE = IM[len0]
IODE = 0
for i in range(len0):
    IODE += IM[i]

# with open('ODE_I.txt', 'w') as f:
#     f.write("[\n")
#     f.write(",".join(str(i) for i in IODE))
#     f.write("\n]\n")

# with open('ODE_M.txt', 'w') as f:
#     f.write("[\n")
#     f.write(",".join(str(i) for i in MODE))
#     f.write("\n]\n")

end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"ODE：{minutes} min {seconds} sec")

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, lognorm
from scipy.special import factorial
from sympy import diff
from sympy import symbols
import math
start_time = time.time()

mu = 3
sd = 1
mu1 = 0
sd1 = 1
mu2 = 2
sd2 = 0.4

lam = 1
k = 1.5

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.2,y_ratio=1):

    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio
    # print(y_data)

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    # ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
    #         [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")
    
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

def Normal(t, mu, sd):
    return np.exp(- (t - mu)**2 / 2 / sd**2) / sd / np.sqrt(2 * np.pi)

def P(t):

    # return 1 - (0.4 * norm.cdf(t, loc=2, scale=0.5) + 0.6 * norm.cdf(t, loc=7, scale=1)) # 0.4norm(2, 0.5) + 0.6norm(7, 1)
    return 1 - (1 + erf((np.log(t) - mu1) / (sd1 * np.sqrt(2)))) / 2 # lognorm(0, 1)
    # return np.exp(- (t / lam)**k) # weibull(1, 1.5)
    # return 1 - (0.5 * norm.cdf(t, loc=3, scale=0.8) + 0.5 * (1 + erf((np.log(t) - mu2) / (sd2 * np.sqrt(2)))) / 2) # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)

def dP(t):

    # return 0.4 * Normal(t, 2, 0.5) + 0.6 * Normal(t, 7, 1) # 0.4norm(2, 0.5) + 0.6norm(7, 1)
    return 1 / (t * sd * np.sqrt(2 * np.pi)) * np.exp(- ((np.log(t) - mu1) / sd1)**2 / 2) # lognorm(0, 1)
    # return k / lam * (t / lam)**(k - 1) * np.exp(- (t / lam)**k) # weibull(1, 1.5)
    # return 0.5 * Normal(t, 3, 0.8) + 0.5 * 1 / (t * sd2 * np.sqrt(2 * np.pi)) * np.exp(- ((np.log(t) - mu2) / sd2)**2 / 2) # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)

def B(s):
    return a * s / (s + 1)

Num = 6000
t1_values = np.linspace(0, 60, Num)  
# I_values = [I(t) for t in t_values]  
# M_values = [M(t) for t in t_values]

dt = t1_values[1] - t1_values[0]

epsilon = I0
y = epsilon * t1_values
epsilonf = I0
f = epsilonf * t1_values
epsilon1 = M0
y1 = epsilon1 * t1_values

# Picard 
max_iter = 1000
tolerance = 1e-6

for iteration in range(max_iter):
    f_new = np.zeros(Num)
    f_new[0] = epsilonf
    y1_new = np.zeros(Num)
    y1_new[0] = epsilon1
    for i in range(1, Num):
        t = t1_values[i]
        x_vals = t1_values[:i]
        f_vals = y1[:i]
        x1_vals = t1_values[:i]
        y1_vals = y1[:i]
        integrand = dP(t - x_vals) * B(f_vals) * np.exp(- mu_I * (t - x_vals)) 
        f_new[i] = np.trapz(integrand, x_vals) + I0 * dP(t) * np.exp(- mu_I * t)
        integrand1 = f[:i] * np.exp(- mu_M * (t - x1_vals)) 
        y1_new[i] = np.trapz(integrand1, x1_vals) + M0 * np.exp(- mu_M * t)
    if (np.linalg.norm(f_new - f, ord=np.inf) < tolerance) & (np.linalg.norm(y1_new - y1, ord=np.inf) < tolerance) :
        print(f"Converged after {iteration + 1} iterations.")
        break
    f = f_new
    y1 = y1_new

for iteration in range(max_iter):
    y_new = np.zeros(Num)
    y_new[0] = epsilon
    for i in range(1, Num):
        t = t1_values[i]
        x_vals = t1_values[:i]
        y_vals = y[:i]
        integrand = P(t - x_vals) * B(y1[:i]) * np.exp(- mu_I * (t - x_vals)) 
        y_new[i] = np.trapz(integrand, x_vals) + I0 * P(t) * np.exp(- mu_I * t)
    if np.linalg.norm(y_new - y, ord=np.inf) < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break
    y = y_new


# print("I_values = [")
# print(",".join(str(i) for i in y))
# print("]")
# print("M_values = [")
# print(",".join(str(i) for i in y1))
# print("]")

# with open('VOL_I.txt', 'w') as f:
#     f.write("[\n")
#     f.write(",".join(str(i) for i in y))
#     f.write("\n]\n")

# with open('VOL_M.txt', 'w') as f:
#     f.write("[\n")
#     f.write(",".join(str(i) for i in y1))
#     f.write("\n]\n")

end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"Volterra：{minutes} min {seconds} sec")

fig, ax = plt.subplots(1,1,figsize=(8, 6))

plt.plot(t1_values, y1, color='#99b9e9', linewidth=2, label="M(t) Volterra")
plt.plot(t_values, MODE, color='#e3716e', linewidth=2, label="M(t) ODEs")
plt.plot(t1_values, y, color='#99b9e9', linestyle='--', linewidth=2, label="I(t) Volterra")
plt.plot(t_values, IODE, color='#e3716e', linestyle='--', linewidth=2, label="I(t) ODEs")

plt.xlabel("Time t")
plt.ylabel("Solutions")
# plt.xlim(0, 31)
plt.legend(loc='upper left')
# plt.grid()

axins = ax.inset_axes((0.5, 0.3, 0.4, 0.3))

axins.plot(t1_values, y1, color='#99b9e9', linewidth=2, label="M(t) Volterra")
axins.plot(t_values, MODE, color='#e3716e', linewidth=2, label="M(t) ODEs")

zone_and_linked(ax, axins, 4000, 4200, t_values , [y1 ,MODE], "bottom")

plt.show()
