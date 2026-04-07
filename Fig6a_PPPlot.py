import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import norm
from scipy.special import gammaln 

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

weights = [0.028001058698141422,0.056002117396282844,0.012789777939911205,0.881890749399159,0.021316296566517675]
k = [4,4,4,4,4]
r = 0.40112928498406786
weights1 = [0.001753466650689302,0.000511314319886104,0.09690675083286157,0.04891468444979749,0.10365492664615662,0.12438591197538092,0.14511689730461394,0.2746425133506452,0.18657886796307807,0.017534666506892303]
k1 = [7,7,7,7,7,7,7,7,7,7]
r1 = 0.7019762487221113
weights2 = [3.001387411235064e-05,0.07138998999320219,0.30664124098041307,0.00041258809610772413,0.20254617123171675,0.07892834977051173,0.09208307473226857,0.10523779969402264,0.011183522010115967,0.13154724961752565]
k2 = [22,22,22,22,22,22,22,22,22,22]
r2 = 2.2062110674123385
weights3 = [0.7644125081291954,1.298346208619654e-16,4.978850051022355e-10,0.19755791392482153,0.017286171550795062,0.02074340586095433,3.0533572593364723e-11,5.813915036917109e-12,9.37460726971495e-17,1.0416230299682432e-16]
k3 = [41,46,46,43,44,44,46,46,46,46]
r3 = 4.162639386628882
weights4 = [0.24788218356555763,1.5943376769698017e-12,3.820912568609961e-05,0.35442156246149614,0.1807528894786077,0.2169034673743282,1.4705341979909678e-06,2.1745668657473505e-07,8.742258521000461e-13,9.713620578889093e-13]
k4 = [59,74,74,73,74,74,74,74,74,74]
r4 = 7.012476543769611

n_components = 5
n_components1 = 10
n_components2 = 10
n_components3 = 10
n_components4 = 10

def erlang_pdf(x, k, lambd):
    """Erlang PDF: f(x | k, lambda) - Numerically stable version"""
    if k < 1 or x <= 0:
        return 0
    log_pdf = (k * np.log(lambd) + (k - 1) * np.log(x) - (lambd * x) - gammaln(k))
    return np.exp(log_pdf)

np.random.seed(999)
n_samples = 1000

base_data = np.random.normal(10, 1.5, n_samples) 

x_range_max = np.max(base_data) * 1.5 
x_for_pdf_calc = np.linspace(np.finfo(float).eps, x_range_max, 2000)

pdf_vals = np.array([np.sum([w * erlang_pdf(x, ki, r) for w, ki in zip(weights, k)]) for x in x_for_pdf_calc])
M = np.max(pdf_vals) * 1.1 if np.max(pdf_vals) > 0 else 1
pdf_vals1 = np.array([np.sum([w * erlang_pdf(x, ki, r1) for w, ki in zip(weights1, k1)]) for x in x_for_pdf_calc])
M1 = np.max(pdf_vals1) * 1.1 if np.max(pdf_vals1) > 0 else 1
pdf_vals2 = np.array([np.sum([w * erlang_pdf(x, ki, r2) for w, ki in zip(weights2, k2)]) for x in x_for_pdf_calc])
M2 = np.max(pdf_vals2) * 1.1 if np.max(pdf_vals2) > 0 else 1
pdf_vals3 = np.array([np.sum([w * erlang_pdf(x, ki, r3) for w, ki in zip(weights3, k3)]) for x in x_for_pdf_calc])
M3 = np.max(pdf_vals3) * 1.1 if np.max(pdf_vals3) > 0 else 1
pdf_vals4 = np.array([np.sum([w * erlang_pdf(x, ki, r4) for w, ki in zip(weights4, k4)]) for x in x_for_pdf_calc])
M4 = np.max(pdf_vals4) * 1.1 if np.max(pdf_vals4) > 0 else 1

proposal = lambda: np.random.uniform(0, x_range_max)  

def rejection_sampling(n_samples, weights, k_arr, r_val, M_val):
    samples_list = []
    if M_val <= 0: return np.array([])
    while len(samples_list) < n_samples:
        x_proposal = proposal()
        u = np.random.uniform(0, 1)
        target_density = np.sum([w * erlang_pdf(x_proposal, ki, r_val) for w, ki in zip(weights, k_arr)])
        if u * M_val <= target_density:
            samples_list.append(x_proposal)
    return np.array(samples_list)

samples = rejection_sampling(n_samples, weights, k, r, M)
samples1 = rejection_sampling(n_samples, weights1, k1, r1, M1)
samples2 = rejection_sampling(n_samples, weights2, k2, r2, M2)
samples3 = rejection_sampling(n_samples, weights3, k3, r3, M3)
samples4 = rejection_sampling(n_samples, weights4, k4, r4, M4)

samples = np.sort(samples)
samples1 = np.sort(samples1)
samples2 = np.sort(samples2)
samples3 = np.sort(samples3)
samples4 = np.sort(samples4)

ecdf_y = np.arange(1, n_samples + 1) / n_samples

theo_prob = norm.cdf(samples, loc=10, scale=1.5)
theo_prob1 = norm.cdf(samples1, loc=10, scale=1.5)
theo_prob2 = norm.cdf(samples2, loc=10, scale=1.5)
theo_prob3 = norm.cdf(samples3, loc=10, scale=1.5)
theo_prob4 = norm.cdf(samples4, loc=10, scale=1.5)

plt.plot(theo_prob, ecdf_y, 'd', markerfacecolor='None', color=(183/255, 178/255, 208/255), linewidth=0.5, label='$L = 4, r = 0.4011$') 
plt.plot(theo_prob1, ecdf_y, 'h', markerfacecolor='None', color=(236/255, 166/255, 128/255), linewidth=0.5, label='$L = 7, r = 0.702$') 
plt.plot(theo_prob2, ecdf_y, '*', markerfacecolor='None', color=(122/255, 199/255, 226/255), linewidth=0.5, label='$L = 21, r = 2.1059$') 
plt.plot(theo_prob3, ecdf_y, '^', markerfacecolor='None', color=(84/255, 190/255, 170/255), linewidth=0.5, label='$L = 46, r = 4.1177$') 
plt.plot(theo_prob4, ecdf_y, 'o', markerfacecolor='None', color=(247/255, 223/255, 135/255), linewidth=0.5, label='$L = 74, r = 7.0604$') 
plt.plot([0, 1], [0, 1], '--', linewidth=1, color='r', label='45-degree line') 

plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("$\mathcal{N}(10, 1.5^2)$ cdf")
plt.ylabel("Mixture Erlang cdf")


plt.show()

