import numpy as np
import math 
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

N_tot = 10000 #38100
N_PMT = 24 #24 for mDOM, 4 for WLS (ring), 1 for WLS (spot)
f_m = 500 #PMT dark noise rate
n_trig = 7 #7 for mDOM (optimal), 2-4 for WLS
t_trig = 20E-9 #20 ns for mDOM, 40ns for WLS (~2*WOM FWHM)

P_m = 1-np.exp(-f_m*t_trig)

def cumulative_binomial(m,n,p):
    s = 0
    for k in range(m+1):
        s += math.comb(n,k) * p**k * (1-p)**(n-k)
    return s

def cumulative_poissonian(m, l):
    s = 0
    for k in range(m+1):
        s = s + l**k * np.exp(-l) / math.factorial(k)

    return s

def significance2probability(significance):
    probability = special.ndtr(significance) - special.ndtr(-significance)

    return probability

def WLS_noise_supression(d, SN_type):
    significance_0 = 5 # 5 sigma
    if SN_type == 'heavy':
        d_0 = 39
    elif SN_type == 'light':
        d_0 = 30
    else:
        print('ERROR')
    significance = significance_0 * (d_0/d)**2
    probability = significance2probability(significance)

    return probability

def f_noise_tot(d, f_noise_mDOM, SN_type):
    P_WLS_collective = 1 - WLS_noise_supression(d, SN_type)
    f_noise_tot = f_noise_mDOM * P_WLS_collective

    return f_noise_tot

def obj1(d, N_nu, mu_0):
    P_SN = (1 - cumulative_poissonian(N_nu-1, mu_0*(10/d)**2))
    return (P_SN - 0.5)**2 #removed np.sqrt() because of error message

def get_d50(N_nu, mu_0):
    thresh = 1E-3
    loss = 1
    d0 = 50

    while loss >= thresh: 
        res = optimize.minimize(obj1, x0 = d0, args = (N_nu, mu_0), method='Nelder-Mead', tol = 1E-6)
        d = res.x
        d = float(d)
        
        loss = obj1(d, N_nu, mu_0)
        d0 = d0+50
        #print(loss < thresh)
        if d0 > 1000:
            break

    return d, loss

def unravel(input):
    n_coin, N_nu = input
    N = n_coin-1 # n_coin starts with 1 not 0
    mu_0 = Mu_0[SN_type][N]
    f_noise_mDOM = F_noise_mDOM[N]
    
    d, loss = get_d50(N_nu, mu_0)

    N_fSN = (1 - cumulative_poissonian(N_nu-2, f_noise_tot(d, f_noise_mDOM, SN_type)*t_SN)) * f_noise_tot(d, f_noise_mDOM, SN_type) * year
    return d, N_fSN

# number of neutrinos for 27M (heavy) SN for 1-fold - 8-fold multiplicity
mu_0_heavy = np.array([2.3E6, 1.84E5, 7E4, 3.35E4, 1.8E4, 9.87E3, 4.6E3, 1.95E3])#, 6.7E2, 1.77E2, 3.4E1, 5E0, 9E-1])

# number of neutrinos for 9.6M (light) SN for 1-fold - 8-fold multiplicity
mu_0_light = np.array([1.33E6, 1.55E5, 4.1E4, 1.93E4, 1.02E4, 5.36E3, 2.6E3, 1.1E3])#, 3.7E2, 9.3E1, 1.6E1, 2E0, 0])

Mu_0 = {'heavy' : mu_0_heavy, 'light': mu_0_light}

F_noise_mDOM = np.array([1.1E8, 3.1E4, 2.3E3, 8.5E1, 4.5E0, 3E-1, 1.6E-2, 5E-4])#, 3E-8])

SN_type = 'heavy' # 'heavy', 'light'
N_fSN_0 = 1 # 1, 0.01 # 1 (0.01) false SN per year
fit_weight = 1E-6
year = 3600*24*365
t_SN = 10 

def globj2(input):
    print(input)
    
    d, N_fSN = unravel(input)

    return np.sqrt((1/d)**2 + fit_weight * (N_fSN - N_fSN_0)**2)


#res = optimize.minimize(globj, x0 = [6,7], method='Nelder-Mead', tol = 1E-6, bounds = ((1,8),(1,1000)))

slices = (slice(7,8,1), slice(2,20,1))

res = optimize.brute(globj2, ranges = slices, disp=True, finish=None, full_output=True)

n_coin_opt, N_nu_opt = res[0]
n_coin_opt, N_nu_opt = int(n_coin_opt), int(N_nu_opt)

print(unravel((n_coin_opt, N_nu_opt)))

 