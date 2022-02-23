from cProfile import label
from cmath import sqrt
import numpy as np
import math 
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

##################################################################
# define functions
##################################################################

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

def obj_p2s(significance, probability):
    prob = significance2probability(significance)
    return (probability-prob)**2

def probability2significance(probability):
    res = optimize.minimize(obj_p2s, x0 = 1, args = (probability), method='Nelder-Mead', tol = 1E-6)
    significance = res.x
    significance = float(significance)
    return significance

def sod (d, s_0, d_0): #signal over distance
    s = s_0 * (d_0/d)**2
    return s

def significance(s, b, type = "poisson", u = False):
    '''
    s - signal
    b - background
    d - distance
    d_0 - initial distance (10 kpc)
    type - poisson, gauss
    u - false, np.array

    If type = "poisson" it returns the CERN recommeded formular for the significance Z for a given signal s and background b in the Poissonian regime. If u = False vanishing uncertainties are assumed. Otherwise a non-vanishing uncertainty scenario is assumed.
    If type = "gauss" it returns the significance assuming a Gaussian distribution.
    n = s + b
    '''
    n = s + b
    if type == "poisson":
        if u == False: # vanishing uncertainty u
            Z = np.sqrt(2*(n*np.log(n/b)-(n-b)))
            if n < b:
                Z *= -1

        else:
            Z = np.sqrt(2*(n*np.log(n*(b+u**2)/(b**2 + n*u**2))
                -b**2/u**2*np.log(1 + u**2*(n-b)/(b*(b+u**2)))))
            if n < b:
                Z *= -1
    elif type == "gauss":
        Z = np.abs(s)/np.sqrt(b)
    
    return Z

def Stouffer(z, u):
    z_tot = np.sum(z)/np.sqrt(np.sum(u))
    return z_tot

def SN_det_prob_mDOM(S0, d, N_nu):
    S = sod(d, S0, d_0 = 10)
    P_SN = 1 - cumulative_poissonian(N_nu-1, S)
    return P_SN

def false_SN_det_prob_mDOM(B, N_nu):
    P_fSN = 1 - cumulative_poissonian(N_nu-2, B)
    return P_fSN

def SN_det_prob_WLS(S0, d, B):
    S = sod(d, S0, d_0 = 10)
    Z = significance(S, B)
    P_SN = significance2probability(Z)
    return P_SN


##################################################################
# debug plots
##################################################################

debug_gauss_poisson = 0


##################################################################
# define variables
##################################################################

# numbers
number_of_modules_upg = 400
number_of_modules_gen2 = 10000
number_of_PMTs_per_module = 24 #mDOM

upg_gen2_ratio = number_of_modules_upg/number_of_modules_gen2

Veff_mDOM = 1776
Veff_WLS = 721

WLS2_mDOM_ratio = 2 * 721/1776

# time measures
dT_SN = 10 #10 s SN duration
dT_T1 = 20E-9 # 20 ns trigger 1 time window 
year = 3600 * 24 * 365

# the following values are from the mDOM coincidence paper (https://arxiv.org/abs/2106.14199)

##################################################################
############################## mDOM ##############################
##################################################################

##################################################################
########################## IceCube-Gen2 ##########################
##################################################################

# number of signal events for IceCube-Gen2 assuming a 27M SN progenitor (heavy)
sig_gen2_heavy = np.array([2.3E6, 1.84E5, 7E4, 3.35E4, 1.89E4, 1E4, 4.8E3, 2.03E3])#, 6.7E2, 1.77E2, 3.4E1, 5E0, 9E-1])

# number of signal events for IceCube-Gen2 assuming a 9.6M SN progenitor (light)
sig_gen2_light = np.array([1.33E6, 1.55E5, 4.1E4, 1.93E4, 1.05E4, 5.5E3, 2.679E3, 1.12E3])#, 3.7E2, 9.3E1, 1.6E1, 2E0, 0])

# total noise rate for IceCube-Gen2 assuming low noise mDOM glass
f_noise_gen2 = np.array([1.1E8, 3.1E4, 2.3E3, 8.5E1, 5.5E0, 2.94E-1, 1.6E-2, 4.9E-4])#, 3E-8])

# number of background events for IceCube-Gen2 assuming low noise mDOM glass
bkg_gen2 = f_noise_gen2 * dT_SN
sig_gen2 = {'heavy' : sig_gen2_heavy, 'light' : sig_gen2_light}
gen2 = {'sig' : sig_gen2, 'bkg' : bkg_gen2, 'noise' : f_noise_gen2}

##################################################################
######################### IceCube Upgrade ########################
##################################################################

# number of signal events for IceCube Upgrade assuming a 27M SN progenitor (heavy)
sig_upg_heavy = sig_gen2_heavy * upg_gen2_ratio

# number of signal events for IceCube Upgrade assuming a 9.6M SN progenitor (light)
sig_upg_light = sig_gen2_light * upg_gen2_ratio

# total noise rate for IceCube Upgrade assuming the quiet mDOM noise model
f_noise_upg_quiet = np.array([9.3E6, 1.8E4, 1.4E2, 5E0, 3.2E-1, 1.8E-2, 9.5E-4, 2.8E-5])

# total noise rate for IceCube Upgrade assuming the noisy mDOM noise model
f_noise_upg_noisy = np.array([5E6, 1.23E4, 9.5E1, 3.5E0, 2.1E-1, 1.2E-2, 6.3E-4, 1.9E-5])

# number of background events for IceCube Upgrade assuming the quiet mDOM noise model
bkg_upg_quiet = f_noise_upg_quiet * dT_SN

# number of background events for IceCube Upgrade assuming the noisy mDOM noise model
bkg_upg_noisy = f_noise_upg_noisy * dT_SN


bkg_upg = {'quiet' : bkg_upg_quiet, 'noisy' : bkg_upg_noisy}
sig_upg = {'heavy' : sig_upg_heavy, 'light' : sig_upg_light}
noise_upg = {'quite' : f_noise_upg_quiet, 'noisy' : f_noise_upg_noisy}
upg = {'sig' : sig_upg, 'bkg' : bkg_upg, 'noise' : noise_upg}

MDOM = {'Gen2' : gen2, 'Upgrade' : upg}

##################################################################
############################ WLS tube ############################
##################################################################

##################################################################
########################## IceCube-Gen2 ##########################
##################################################################

f_noise_WLS_dark = 50 # 50 Hz dark noise rate for standard PMT
f_noise_WLS_radio = 200 # 200 Hz noise from radioactive decay


f_noise_WLS_gen2 = (f_noise_WLS_dark + f_noise_WLS_radio) * 2 * number_of_modules_gen2 # combined noise rate
bkg_WLS_gen2 = f_noise_WLS_gen2 * dT_SN # total background for Gen2

sig_WLS_gen2_heavy = sig_gen2_heavy[0] * WLS2_mDOM_ratio

sig_WLS_gen2_light = sig_gen2_light[0] * WLS2_mDOM_ratio

sig_WLS_gen2 = {'heavy' : sig_WLS_gen2_heavy, 'light' : sig_WLS_gen2_light}

WLS_gen2 = {'sig' : sig_WLS_gen2, 'bkg' : bkg_WLS_gen2, 'noise' : f_noise_WLS_gen2}

f_noise_WLS_upg = (f_noise_WLS_dark + f_noise_WLS_radio) * 2 * number_of_modules_upg # combined noise rate
bkg_WLS_upg = f_noise_WLS_upg * dT_SN # total background for Gen2

sig_WLS_upg_heavy = sig_upg_heavy[0] * WLS2_mDOM_ratio # 1-fold coincidence

sig_WLS_upg_light = sig_upg_light[0] * WLS2_mDOM_ratio # 1-fold coincidence

sig_WLS_upg = {'heavy' : sig_WLS_upg_heavy, 'light' : sig_WLS_upg_light}

WLS_upg = {'sig' : sig_WLS_upg, 'bkg' : bkg_WLS_upg, 'noise' : f_noise_WLS_upg}

WLS = {'Gen2' : WLS_gen2, 'Upgrade' : WLS_upg}

##################################################################
# Detection Probability
##################################################################






##################################################################
# brute force minimization
##################################################################

det_type = 'Gen2' # 'Gen2', 'Upgrade'
sig_type = 'heavy' # 'heavy', 'light'
bkg_type = 'quiet' # 'quiet', 'noisy'
n_coin = 7 # m-fold coincidence
N_nu = 7   # number of hit modules
d_0 = 10 # simulation for SN at 10 kpc

def unravel(input):

    n_coin, N_nu = input
    # number of signal events for this configuration
    S_mDOM = MDOM[det_type]['sig'][sig_type][n_coin-1]
    S_WLS = WLS[det_type]['sig'][sig_type]


    # number of background events for this configuration
    if det_type == 'Upgrade':
        B_mDOM = MDOM[det_type]['bkg'][bkg_type][n_coin-1]
        B_WLS = WLS[det_type]['bkg'][sig_type]
        F_mDOM = MDOM[det_type]['noise'][bkg_type][n_coin-1]
    else:
        B_mDOM = MDOM[det_type]['bkg'][n_coin-1]
        B_WLS = WLS[det_type]['bkg']
        F_mDOM = MDOM[det_type]['noise'][n_coin-1]

    #P_SN_tot = SN_det_prob_mDOM(S_mDOM, d, N_nu) * SN_det_prob_WLS(S_WLS, d, B_WLS)
    d_50, loss = get_d_50(N_nu, S_mDOM, S_WLS, B_WLS)

    P_fSN_tot = false_SN_det_prob_mDOM(B_mDOM, N_nu) * (1 - SN_det_prob_WLS(S_WLS, d_50, B_WLS))
    N_fSN_tot = year * F_mDOM * P_fSN_tot

    return d_50, N_fSN_tot

def get_d_50(N_nu, S_mDOM, S_WLS, B_WLS):
    thresh = 1E-3
    loss = 1
    d_ini = 50

    while loss >= thresh: 
        res = optimize.minimize(globj1, x0 = d_ini, args = (N_nu, S_mDOM, S_WLS, B_WLS), method='Nelder-Mead', tol = 1E-6)
        d_50 = res.x
        d_50 = float(d_50)
        
        loss = globj1(d_50, N_nu, S_mDOM, S_WLS, B_WLS)
        d_ini = d_ini+50
        #print(loss < thresh)
        if d_ini > 1000:
            break

    return d_50, loss

def globj1(d, N_nu, S_mDOM, S_WLS, B_WLS):
    P_SN_tot = SN_det_prob_mDOM(S_mDOM, d, N_nu) * SN_det_prob_WLS(S_WLS, d, B_WLS)
    return np.sqrt((P_SN_tot - 0.5)**2)

def globj2(input):
    
    d_50, N_fSN = unravel(input)

    return np.sqrt((1/d_50)**2 + fit_weight * (N_fSN - N_fSN_0)**2)

fit_weight = 1E-6
N_fSN_0 = 0.01

slices = (slice(7,8,1), slice(2,20,1))

res = optimize.brute(globj2, ranges = slices, disp=True, finish=None, full_output=True)

###########################################

n_coin, N_nu = np.array(res[0], dtype=int)
# number of signal events for this configuration
S_mDOM = MDOM[det_type]['sig'][sig_type][n_coin-1]
S_WLS = WLS[det_type]['sig'][sig_type]


# number of background events for this configuration
if det_type == 'Upgrade':
    B_mDOM = MDOM[det_type]['bkg'][bkg_type][n_coin-1]
    B_WLS = WLS[det_type]['bkg'][sig_type]
    F_mDOM = MDOM[det_type]['noise'][bkg_type][n_coin-1]
else:
    B_mDOM = MDOM[det_type]['bkg'][n_coin-1]
    B_WLS = WLS[det_type]['bkg']
    F_mDOM = MDOM[det_type]['noise'][n_coin-1]

if 1:
    d = np.arange(10, 500, 1)
    Z_mDOM = []
    P_SN_mDOM = []
    P_fSN_mDOM = []
    Z_WLS = []
    P_SN_WLS = []
    P_fSN_WLS = []
    for dd in d:
        Sd_mDOM = sod(dd, S_mDOM, 10)
        Sd_WLS =  sod(dd, S_WLS, 10)
        P_SN_mDOM.append(SN_det_prob_mDOM(S_mDOM, dd, N_nu))
        P_SN_WLS.append(SN_det_prob_WLS(S_WLS, dd, B_WLS))
        P_fSN_mDOM.append(false_SN_det_prob_mDOM(B_mDOM, N_nu))
        P_fSN_WLS.append(1-SN_det_prob_WLS(S_WLS, dd, B_WLS))
        Z_mDOM.append(significance(Sd_mDOM, B_mDOM, type = "poisson"))
        Z_WLS.append(significance(Sd_WLS, B_WLS, type = "poisson"))
    Z_mDOM = np.array(Z_mDOM)
    Z_WLS = np.array(Z_WLS)
    P_SN_mDOM = np.array(P_SN_mDOM)
    P_SN_WLS = np.array(P_SN_WLS)
    P_fSN_mDOM = np.array(P_fSN_mDOM)
    P_fSN_WLS = np.array(P_fSN_WLS)

    Z_comb = (Z_mDOM + Z_WLS)/np.sqrt(2)
    P_SN_comb = P_SN_mDOM * P_SN_WLS
    P_fSN_comb = P_fSN_mDOM * P_fSN_WLS

    fig, axs = plt.subplots(1,2, figsize = (8,4))

    ax1 = axs[0]
    ax21 = axs[1]
    ax22 = axs[1].twinx()

    ax1.plot(d, Z_mDOM, label = "mDOM", color = 'grey', ls = ':')
    ax1.plot(d, Z_WLS, label = "WLS", color = 'grey', ls = '--')
    ax1.plot(d, Z_comb, label = "WOM-Trap", color = 'black', ls = '-')
    ax21.plot(d, P_SN_mDOM, label = "mDOM", color = 'grey', ls = ':')
    ax21.plot(d, P_SN_WLS, label = "WLS", color = 'grey', ls = '--')
    ax21.plot(d, P_SN_comb, label = "WOM-Trap", color = 'black', ls = '-')
    ax22.plot(d, P_fSN_mDOM, color = 'C1', ls = ':')
    ax22.plot(d, P_fSN_WLS, color = 'C1', ls = '--')
    ax22.plot(d, P_fSN_comb, color = 'red', ls = '-')
    ax1.set_xlabel("Distance d [kpc]")
    ax1.set_ylabel("Significance Z")
    ax1.set_yscale("log")
    ax21.set_xlabel("Distance d [kpc]")
    ax21.set_ylabel(r"SN detection probability $P_{SN}$")
    ax22.set_ylabel(r"false SN detection probability $P_{fSN}$")
    ax22.yaxis.label.set_color('red')    
    ax22.tick_params(axis='y', colors='red')
    ax1.legend()
    ax21.legend()
    plt.tight_layout()
    plt.savefig('sig_prob_opt.png')
    plt.show()