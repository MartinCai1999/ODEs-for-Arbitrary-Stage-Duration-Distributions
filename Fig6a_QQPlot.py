import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import numpy as np
from scipy.special import gamma, factorial
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from datetime import datetime
import time
import scipy.stats as stats

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 14            
plt.rcParams['axes.titlesize'] = 18       
plt.rcParams['axes.labelsize'] = 12       
plt.rcParams['xtick.labelsize'] = 12      
plt.rcParams['ytick.labelsize'] = 12     
plt.rcParams['legend.fontsize'] = 10      
plt.rcParams['figure.titlesize'] = 20     

weights = [
0.028001058698141422,0.056002117396282844,0.012789777939911205,0.881890749399159,0.021316296566517675
]
k = [
4,4,4,4,4
]
r = 0.40112928498406786

weights1 = [
0.001753466650689302,0.000511314319886104,0.09690675083286157,0.04891468444979749,0.10365492664615662,0.12438591197538092,0.14511689730461394,0.2746425133506452,0.18657886796307807,0.017534666506892303
]
k1 = [
7,7,7,7,7,7,7,7,7,7
]
r1 = 0.7019762487221113

weights2 = [
3.001387411235064e-05,0.07138998999320219,0.30664124098041307,0.00041258809610772413,0.20254617123171675,0.07892834977051173,0.09208307473226857,0.10523779969402264,0.011183522010115967,0.13154724961752565
]
k2 = [
22,22,22,22,22,22,22,22,22,22
]
r2 = 2.2062110674123385

weights3 = [
0.7644125081291954,1.298346208619654e-16,4.978850051022355e-10,0.19755791392482153,0.017286171550795062,0.02074340586095433,3.0533572593364723e-11,5.813915036917109e-12,9.37460726971495e-17,1.0416230299682432e-16
]
k3 = [
41,46,46,43,44,44,46,46,46,46
]
r3 = 4.162639386628882

weights4 = [
0.24788218356555763,1.5943376769698017e-12,3.820912568609961e-05,0.35442156246149614,0.1807528894786077,0.2169034673743282,1.4705341979909678e-06,2.1745668657473505e-07,8.742258521000461e-13,9.713620578889093e-13
]
k4 = [
59,74,74,73,74,74,74,74,74,74
]
r4 = 7.012476543769611

n_components = 5
n_components1 = 10
n_components2 = 10
n_components3 = 10
n_components4 = 10

def erlang_pdf(x, k, lambd):
    """Erlang PDF: f(x | k, lambda)"""
    t = (lambd**k * x**(k-1) * np.exp(-lambd * x)) / factorial(k-1)
    # t[t == 0] = np.finfo(float).eps
    return t

def erlang_cdf(x, k, lambd):
    """Erlang PDF: f(x | k, lambda)"""
    ans = 0
    for i in range(k):
        ans += (lambd * x)**i * np.exp(-lambd * x) / factorial(i)
    # t[t == 0] = np.finfo(float).eps
    return 1 - ans

np.random.seed(999)
n_samples = 1000

data = np.random.normal(10, 1.5, n_samples) 

x = np.linspace(0, np.max(data), n_samples)
pdf = np.sum([weights[j] * erlang_pdf(x, k[j], r) for j in range(n_components)], axis=0)
pdf1 = np.sum([weights1[j] * erlang_pdf(x, k1[j], r1) for j in range(n_components1)], axis=0)
pdf2 = np.sum([weights2[j] * erlang_pdf(x, k2[j], r2) for j in range(n_components2)], axis=0)
pdf3 = np.sum([weights3[j] * erlang_pdf(x, k3[j], r3) for j in range(n_components3)], axis=0)
pdf4 = np.sum([weights4[j] * erlang_pdf(x, k4[j], r4) for j in range(n_components4)], axis=0)

nn = 1

M = max(pdf) * 1.1  
proposal = lambda: np.random.uniform(0, np.max(data))  
samples = []
num = 0
while num < n_samples:
    x_proposal = proposal()
    u = np.random.uniform(0, 1)
    if u <= np.sum([weights[j] * erlang_pdf(x_proposal, k[j], r) for j in range(n_components)]) / M:
        samples.append(x_proposal)
        num += 1
data = np.sort(data) 
samples = np.sort(samples)
data = data[0:int(n_samples * nn)]
samples = samples[0:int(n_samples * nn)]

M1 = max(pdf1) * 1.1 
samples1 = []
num = 0
while num < n_samples:
    x_proposal = proposal()
    u = np.random.uniform(0, 1)
    if u <= np.sum([weights1[j] * erlang_pdf(x_proposal, k1[j], r1) for j in range(n_components1)]) / M1:
        samples1.append(x_proposal)
        num += 1
samples1 = np.sort(samples1)
samples1 = samples1[0:int(n_samples * nn)]

M2 = max(pdf2) * 1.1  
samples2 = []
num = 0
while num < n_samples:
    x_proposal = proposal()
    u = np.random.uniform(0, 1)
    if u <= np.sum([weights2[j] * erlang_pdf(x_proposal, k2[j], r2) for j in range(n_components2)]) / M2:
        samples2.append(x_proposal)
        num += 1
samples2 = np.sort(samples2)
samples2 = samples2[0:int(n_samples * nn)]

M3 = max(pdf3) * 1.1 
samples3 = []
num = 0
while num < n_samples:
    x_proposal = proposal()
    u = np.random.uniform(0, 1)
    if u <= np.sum([weights3[j] * erlang_pdf(x_proposal, k3[j], r3) for j in range(n_components3)]) / M3:
        samples3.append(x_proposal)
        num += 1
samples3 = np.sort(samples3)
samples3 = samples3[0:int(n_samples * nn)]

M4 = max(pdf4) * 1.1  
samples4 = []
num = 0
while num < n_samples:
    x_proposal = proposal()
    u = np.random.uniform(0, 1)
    if u <= np.sum([weights4[j] * erlang_pdf(x_proposal, k4[j], r4) for j in range(n_components4)]) / M4:
        samples4.append(x_proposal)
        num += 1
samples4 = np.sort(samples4)
samples4 = samples4[0:int(n_samples * nn)]


plt.plot(data, samples, 'd', markerfacecolor='None', color=(183/255, 178/255, 208/255), linewidth=0.5, label='$L = 4, r = 0.4011$') 
plt.plot(data, samples1, 'h', markerfacecolor='None', color=(236/255, 166/255, 128/255), linewidth=0.5, label='$L = 7, r = 0.702$') 
plt.plot(data, samples2, '*', markerfacecolor='None', color=(122/255, 199/255, 226/255), linewidth=0.5, label='$L = 21, r = 2.1059$') 
plt.plot(data, samples3, '^', markerfacecolor='None', color=(84/255, 190/255, 170/255), linewidth=0.5, label='$L = 46, r = 4.1177$') 
plt.plot(data, samples4, 'o', markerfacecolor='None', color=(247/255, 223/255, 135/255), linewidth=0.5, label='$L = 74, r = 7.0604$') 
plt.plot([6, 14], [6, 14], '--', linewidth=1, color='r', label='45-degree line') 
plt.legend()
plt.xlim(7, 13)
plt.ylim(7, 13)
plt.xlabel("$\mathcal{N}(10, 1.5^2)$ pdf")
plt.ylabel("Mixture Erlang pdf")

plt.show()