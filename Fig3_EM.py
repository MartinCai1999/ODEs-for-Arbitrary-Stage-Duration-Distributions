import numpy as np
from scipy.special import gamma, factorial
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from datetime import datetime
import time
import scipy.stats as stats
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

def erlang_pdf(x, k, lambd):
    """Erlang PDF: f(x | k, lambda)"""
    t = (lambd**k * x**(k-1) * np.exp(-lambd * x)) / factorial(k-1)
    # t[t == 0] = np.finfo(float).eps
    return t

np.random.seed(99)
n_samples = 1000

mu1, sigma1 = 3, 0.8 
mu2, sigma2 = 2, 0.4
a = [0.5, 0.5]
data1 = np.random.normal(mu1, sigma1, int(a[0] * n_samples))
data2 = np.random.lognormal(mu2, sigma2, int(a[1] * n_samples))
n_samples = int(a[0] * n_samples) + int(a[1] * n_samples)
data = np.hstack((data1, data2))
np.random.shuffle(data)

# data = np.random.lognormal(0, 1, n_samples) 

# k = 1.5  
# lambda_ = 1  
# weibull = stats.weibull_min(c=k, scale=lambda_)
# data = weibull.rvs(size=n_samples) 


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

mu1, sigma1 = 2, 0.5  
mu2, sigma2 = 7, 1.0  

weight1, weight2 = 0.4, 0.6  
choices = np.random.choice([0, 1], size=n_samples, p=[weight1, weight2])
samples1 = np.random.normal(mu1, sigma1, size=n_samples)
samples2 = np.random.normal(mu2, sigma2, size=n_samples)
data = np.where(choices == 0, samples1, samples2)
np.random.shuffle(data)

p80 = np.percentile(data, 80)
ma = max(data)
lambdas = 8 / p80 
n_components = 5 
ks = np.random.randint(20, 30, size=n_components)
t = np.arange(0, n_components + 1)
se1 = np.asarray(pd.cut(data, t / lambdas).value_counts().tolist())
weights = np.ones(n_components) / n_components
tolerance = 1e-6
max_iter = 2000

def em_algorithm_erlang(data, n_components, ks, lambdas, weights, max_iter=1000, tol=1e-9):
    n_samples = len(data)
    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step:
        responsibilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            responsibilities[:, i] = weights[i] * erlang_pdf(data, ks[i], lambdas)
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= row_sums

        N_k = responsibilities.sum(axis=0)
        weights = N_k / n_samples
        lambdas = np.sum(weights * ks) / np.sum(data / n_samples)

        # M-step:
        for i in range(n_components):
            k_values = np.arange(1, 50)  
            likelihoods = [
                np.sum(responsibilities[:, i] * np.log(erlang_pdf(data, k, lambdas))) 
                for k in k_values
            ]
            ks[i] = k_values[np.argmax(likelihoods)]
        EPSILON = 1e-9
        log_likelihood = np.sum(
            np.sum([(responsibilities[:, i] * (np.log(weights[i] + EPSILON) - data * lambdas + ks[i] * np.log(lambdas + EPSILON))) for i in range(n_components)], axis=0)
        )
        log_likelihoods.append(log_likelihood)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return weights, ks, lambdas, log_likelihoods

weights, ks, lambdas, log_likelihoods = em_algorithm_erlang(data, n_components, ks, lambdas, weights)

end_time = time.time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"EM：{minutes} min {seconds} sec")

print("weights = [")
print(",".join(str(i) for i in weights))
print("]")
print("k = [")
print(",".join(str(i) for i in ks))
print("]")
print("r = ")
print(lambdas)
print("theta:")
print(1 / lambdas)
print("log likelihoods:")
print(log_likelihoods[-1])

final_log_likelihood = np.sum(
    np.log(np.sum([weights[i] * erlang_pdf(data, ks[i], lambdas) for i in range(n_components)], axis=0))
)
print("log likelihoods:")
print(log_likelihoods[-1]) 
print("Final (True) Log-Likelihood for BIC:", final_log_likelihood)
l = np.max(ks)
w = [0] * l
for i in range(len(ks)):
    w[ks[i] - 1] += weights[i]
ll = np.count_nonzero(w)
print(sum(set(ks)) + 1)
BIC = (2 * ll + 1) * np.log(n_samples) - 2 * final_log_likelihood
ODEBIC = (sum(set(ks)) + 1) * np.log(n_samples) - 2 * final_log_likelihood
print("Normal BIC:")
print(BIC)
print("ODE BIC:")
print(ODEBIC)

x = np.linspace(0, np.max(data), n_samples)
pdf = np.sum([weights[j] * erlang_pdf(x, ks[j], lambdas) for j in range(n_components)], axis=0)
n_bins = 50
heights, bin_edges = np.histogram(data, bins=n_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.scatter(bin_centers, heights, 
            color='#99b9e9',       
            alpha=1,               
            label="$0.4\,\mathcal{N}(2,\,0.5^2) + 0.6\,\mathcal{N}(7,\,1^2)$" 
           )
plt.plot(x, pdf, color='#e3716e', label="Fitted Mixture Erlang")
plt.legend()
# plt.title("EM Algorithm for Mixture Erlang Distribution")
plt.xlabel("x")
plt.ylabel("Density")

plt.show()


