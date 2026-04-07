import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime
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

l = np.max(k)
w = [0] * l
for i in range(len(k)):
    w[k[i] - 1] += weights[i]
ll = np.count_nonzero(w)
k = list(set(k))
len0 = sum(k)

l1 = np.max(k1)
w1 = [0] * l1
for i in range(len(k1)):
    w1[k1[i] - 1] += weights1[i]
ll1 = np.count_nonzero(w1)
k1 = list(set(k1))
len1 = sum(k1)

l2 = np.max(k2)
w2 = [0] * l2
for i in range(len(k2)):
    w2[k2[i] - 1] += weights2[i]
ll2 = np.count_nonzero(w2)
k2 = list(set(k2))
len2 = sum(k2)

l3 = np.max(k3)
w3 = [0] * l3
for i in range(len(k3)):
    w3[k3[i] - 1] += weights3[i]
ll3 = np.count_nonzero(w3)
k3 = list(set(k3))
len3 = sum(k3)

l4 = np.max(k4)
w4 = [0] * l4
for i in range(len(k4)):
    w4[k4[i] - 1] += weights4[i]
ll4 = np.count_nonzero(w4)
k4 = list(set(k4))
len4 = sum(k4)

mu_I = 0.3
mu_M = 0.1
a = 5

I0 = 0.1
M0 = 0.05

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

I00 = np.zeros(len0 + 1) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]
# I00[0] = 10 # 0.1
ind = 0
for i in range(ll):
    for j in range(k[i]):
        if (j == 0):
            I00[ind] = I0 * w[k[i] - 1]
        ind += 1
I00[len0] = M0  # 0.05

def odes1(t, IODE):
    dI = np.zeros_like(IODE)
    ind = 0
    for i in range(ll1):
        for j in range(k1[i]):
            if (j == 0):
                dI[ind] = B(IODE[len1]) * w1[k1[i] - 1] - (mu_I + r1) * IODE[ind]
            else:
                dI[ind] = r1 * IODE[ind - 1] - (mu_I + r1) * IODE[ind]
                if (j == k1[i] - 1):
                    dI[len1] += IODE[ind]
            ind += 1
    dI[len1] *= r1
    dI[len1] -= mu_M * IODE[len1]
    return dI

I001 = np.zeros(len1 + 1) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]
# I00[0] = 10 # 0.1
ind = 0
for i in range(ll1):
    for j in range(k1[i]):
        if (j == 0):
            I001[ind] = I0 * w1[k1[i] - 1]
        ind += 1
I001[len1] = M0 # 0.05

def odes2(t, IODE):
    dI = np.zeros_like(IODE)
    ind = 0
    for i in range(ll2):
        for j in range(k2[i]):
            if (j == 0):
                dI[ind] = B(IODE[len2]) * w2[k2[i] - 1] - (mu_I + r2) * IODE[ind]
            else:
                dI[ind] = r2 * IODE[ind - 1] - (mu_I + r2) * IODE[ind]
                if (j == k2[i] - 1):
                    dI[len2] += IODE[ind]
            ind += 1
    dI[len2] *= r2
    dI[len2] -= mu_M * IODE[len2]
    return dI

I002 = np.zeros(len2 + 1) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]
# I00[0] = 10 # 0.1
ind = 0
for i in range(ll2):
    for j in range(k2[i]):
        if (j == 0):
            I002[ind] = I0 * w2[k2[i] - 1]
        ind += 1
I002[len2] = M0  # 0.05

def odes3(t, IODE):
    dI = np.zeros_like(IODE)
    ind = 0
    for i in range(ll3):
        for j in range(k3[i]):
            if (j == 0):
                dI[ind] = B(IODE[len3]) * w3[k3[i] - 1] - (mu_I + r3) * IODE[ind]
            else:
                dI[ind] = r3 * IODE[ind - 1] - (mu_I + r3) * IODE[ind]
                if (j == k3[i] - 1):
                    dI[len3] += IODE[ind]
            ind += 1
    dI[len3] *= r3
    dI[len3] -= mu_M * IODE[len3]
    return dI

I003 = np.zeros(len3 + 1) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]
# I00[0] = 10 # 0.1
ind = 0
for i in range(ll3):
    for j in range(k3[i]):
        if (j == 0):
            I003[ind] = I0 * w3[k3[i] - 1]
        ind += 1
I003[len3] = M0 # 0.05

def odes4(t, IODE):
    dI = np.zeros_like(IODE)
    ind = 0
    for i in range(ll4):
        for j in range(k4[i]):
            if (j == 0):
                dI[ind] = B(IODE[len4]) * w4[k4[i] - 1] - (mu_I + r4) * IODE[ind]
            else:
                dI[ind] = r4 * IODE[ind - 1] - (mu_I + r4) * IODE[ind]
                if (j == k4[i] - 1):
                    dI[len4] += IODE[ind]
            ind += 1
    dI[len4] *= r4
    dI[len4] -= mu_M * IODE[len4]
    return dI

I004 = np.zeros(len4 + 1) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.05]
# I00[0] = 10 # 0.1
ind = 0
for i in range(ll4):
    for j in range(k4[i]):
        if (j == 0):
            I004[ind] = I0 * w4[k4[i] - 1]
        ind += 1
I004[len4] = M0 # 0.05

start = 0
end = 60
num = 6000
t_values = np.linspace(start, end, num)
t_span = (start, end)
t_eval = np.linspace(t_span[0], t_span[1], num)

solution = solve_ivp(odes, t_span, I00, t_eval=t_eval)
solution1 = solve_ivp(odes1, t_span, I001, t_eval=t_eval)
solution2 = solve_ivp(odes2, t_span, I002, t_eval=t_eval)
solution3 = solve_ivp(odes3, t_span, I003, t_eval=t_eval)
solution4 = solve_ivp(odes4, t_span, I004, t_eval=t_eval)

t = solution.t
IM = solution.y
MODE = IM[len0]
IODE = 0
for i in range(len0):
    IODE += IM[i]

t1 = solution1.t
IM1 = solution1.y
MODE1 = IM1[len1]
IODE1 = 0
for i in range(len1):
    IODE1 += IM1[i]

t2 = solution2.t
IM2 = solution2.y
MODE2 = IM2[len2]
IODE2 = 0
for i in range(len2):
    IODE2 += IM2[i]

t3 = solution3.t
IM3 = solution3.y
MODE3 = IM3[len3]
IODE3 = 0
for i in range(len3):
    IODE3 += IM3[i]

t4 = solution4.t
IM4 = solution4.y
MODE4 = IM4[len4]
IODE4 = 0
for i in range(len4):
    IODE4 += IM4[i]

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

I0 = 0.1 
M0 = 0.05 

mu = 3
sd = 1

lam = 1
k = 1.5

def Normal(t, mu, sd):
    return np.exp(- (t - mu)**2 / 2 / sd**2) / sd / np.sqrt(2 * np.pi)

def P(t):

    # return 1 - (0.4 * norm.cdf(t, loc=2, scale=0.5) + 0.6 * norm.cdf(t, loc=7, scale=1)) # 0.4norm(2, 0.5) + 0.6norm(7, 1)
    return 1 - norm.cdf(t, loc=10, scale=1.5)
    # return 1 - (1 + math.erf((np.log(t) - mu) / (sd * np.sqrt(2)))) / 2 # lognorm(0, 1)
    # return np.exp(- (t / lam)**k) # weibull(1, 1.5)
    # return 1 - (0.5 * norm.cdf(t, loc=3, scale=0.8) + 0.5 * (1 + math.erf((np.log(t) - mu) / (sd * np.sqrt(2)))) / 2) # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)

def dP(t):

    # return 0.4 * Normal(t, 2, 0.5) + 0.6 * Normal(t, 7, 1) # 0.4norm(2, 0.5) + 0.6norm(7, 1)
    # return 1 / (t * sd * np.sqrt(2 * np.pi)) * np.exp(- ((np.log(t) - mu) / sd)**2 / 2) # lognorm(0, 1)
    # return k / lam * (t / lam)**(k - 1) * np.exp(- (t / lam)**k) # weibull(1, 1.5)
    # return 0.5 * Normal(t, 3, 0.8) + 0.5 * 1 / (t * sd * np.sqrt(2 * np.pi)) * np.exp(- ((np.log(t) - mu) / sd)**2 / 2) # 0.5norm(3, 0.8) + 0.5lognorm(2, 0.4)
    return Normal(t, 10, 1.5)

def B(s):
    return a * s / (s + 1)
    
def I(t):
    integral, _ = quad(lambda s: B(s) * P(t - s) * np.exp(- mu_I * (t - s)), 0, t) + I0 * P(t) * np.exp(- mu_I * t)  
    return integral

def F(t):
    integral, _ = quad(lambda s: B(s) * dP(t - s) * np.exp(- mu_I * (t - s)), 0, t) + I0 * dP(t) * np.exp(- mu_I * t)  

def M(t):
    integral, _ = quad(lambda s: F(s) * np.exp(- mu_M * (t - s)), 0, t) + M0 * np.exp(- mu_M * t) 
    return integral

Num = 6000
t1_values = np.linspace(0, 60, Num)  

dt = t1_values[1] - t1_values[0]

epsilon = I0
y = epsilon * t1_values
epsilonf = I0
f = epsilonf * t1_values
epsilon1 = M0
y1 = epsilon1 * t1_values

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

# t_values = np.linspace(0, 100, 1000) 
# I_values = [I(t) for t in t_values]  
# M_values = [M(t) for t in t_values]

# print("I_values = [")
# print(",".join(str(i) for i in I_values))
# print("]")
# print("M_values = [")
# print(",".join(str(i) for i in M_values))
# print("]")

print(y, y1)

end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"Volterra：{minutes} min {seconds} sec")

I_values = np.sort(y)
IODE = np.sort(IODE)
IODE1 = np.sort(IODE1)
IODE2 = np.sort(IODE2)
IODE3 = np.sort(IODE3)
IODE4 = np.sort(IODE4)
M_values = np.sort(y1)
MODE = np.sort(MODE)
MODE1 = np.sort(MODE1)
MODE2 = np.sort(MODE2)
MODE3 = np.sort(MODE3)
MODE4 = np.sort(MODE4)

plt.plot(I_values, IODE, 'd', markerfacecolor='None', color=(183/255, 178/255, 208/255), linewidth=0.5, label='$L = 4, r = 0.4011$') 
plt.plot(I_values, IODE1, 'h', markerfacecolor='None', color=(236/255, 166/255, 128/255), linewidth=0.5, label='$L = 7, r = 0.702$') 
plt.plot(I_values, IODE2, '*', markerfacecolor='None', color=(122/255, 199/255, 226/255), linewidth=0.5, label='$L = 21, r = 2.1059$') 
plt.plot(I_values, IODE3, '^', markerfacecolor='None', color=(84/255, 190/255, 170/255), linewidth=0.5, label='$L = 46, r = 4.1177$') 
plt.plot(I_values, IODE4, 'o', markerfacecolor='None', color=(247/255, 223/255, 135/255), linewidth=0.5, label='$L = 74, r = 7.0604$') 

plt.plot([np.min(I_values), np.max(I_values)], [np.min(I_values), np.max(I_values)], '--', color='r', linewidth=1, label='45-degree line') 
plt.legend()
plt.xlabel("Solutions of I(t) by Volterra integral")
plt.ylabel("Solutions of I(t) by ODEs")
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.show()

plt.plot(M_values, MODE, 'd', markerfacecolor='None', color=(183/255, 178/255, 208/255), linewidth=0.5, label='$L = 4, r = 0.4011$') 
plt.plot(M_values, MODE1, 'h', markerfacecolor='None', color=(236/255, 166/255, 128/255), linewidth=0.5, label='$L = 7, r = 0.702$') 
plt.plot(M_values, MODE2, '*', markerfacecolor='None', color=(122/255, 199/255, 226/255), linewidth=0.5, label='$L = 21, r = 2.1059$') 
plt.plot(M_values, MODE3, '^', markerfacecolor='None', color=(84/255, 190/255, 170/255), linewidth=0.5, label='$L = 46, r = 4.1177$') 
plt.plot(M_values, MODE4, 'o', markerfacecolor='None', color=(247/255, 223/255, 135/255), linewidth=0.5, label='$L = 74, r = 7.0604$') 

# plt.plot([0, 20], [0, 20], '--', color='royalblue', label='45-degree line') 
plt.plot([np.min(M_values), np.max(M_values)], [np.min(M_values), np.max(M_values)], '--', color='r', linewidth=1, label='45-degree line')
plt.xlabel("Solutions of M(t) by Volterra integral")
plt.ylabel("Solutions of M(t) by ODEs")
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.legend()

plt.show()
