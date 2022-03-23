import numpy as np

# define general variables

dT_SN = 10 # 10 s SN time window
dT_L1 = 20E-9 # 20 ns trigger L1 window
year = 3600*24*365

number_of_modules_upg = 400
number_of_modules_gen2 = 10000
number_of_PMTs_per_mDOM = 24 #mDOM
number_of_PMTs_per_LOM18 = 18 #LOM18
number_of_PMTs_per_LOM16 = 16 #LOM16

# If LOM data is available make sure to change to WLS2_LOM_ratio in WLS section!

upg_gen2_ratio = number_of_modules_upg/number_of_modules_gen2
WLS2_mDOM_ratio = 2 * 721/1776 # 2 * V_eff_WLS/V_eff_mDOM


distance_0 = 10 # 10 kpc

##################################################################
############################## mDOM ##############################
##################################################################

########################## IceCube-Gen2 ##########################

# number of signal events for IceCube-Gen2 assuming a 27M SN progenitor (heavy) in mDOMs
sig_mDOM_gen2_heavy = np.array([2.3E6, 1.84E5, 7E4, 3.35E4, 1.89E4, 1E4, 4.8E3, 2.03E3])#, 6.7E2, 1.77E2, 3.4E1, 5E0, 9E-1])

# number of signal events for IceCube-Gen2 assuming a 9.6M SN progenitor (light) in mDOMs
sig_mDOM_gen2_light = np.array([1.33E6, 1.55E5, 4.1E4, 1.93E4, 1.05E4, 5.5E3, 2.679E3, 1.12E3])#, 3.7E2, 9.3E1, 1.6E1, 2E0, 0])

# total noise rate for IceCube-Gen2 assuming low noise mDOM glass in mDOMs
f_noise_mDOM_gen2 = np.array([1.1E8, 3.1E4, 2.3E3, 8.5E1, 5.5E0, 2.94E-1, 1.6E-2, 4.9E-4])#, 3E-8])

######################### IceCube Upgrade ########################

# number of signal events for IceCube Upgrade assuming a 27M SN progenitor (heavy) in mDOMs
sig_mDOM_upg_heavy = sig_mDOM_gen2_heavy * upg_gen2_ratio

# number of signal events for IceCube Upgrade assuming a 9.6M SN progenitor (light) in mDOMs
sig_mDOM_upg_light = sig_mDOM_gen2_light * upg_gen2_ratio

# total noise rate for IceCube Upgrade assuming the noisy mDOM noise model in mDOMs
f_noise_mDOM_upg_noisy = np.array([9.3E6, 1.8E4, 1.4E2, 5E0, 3.2E-1, 1.8E-2, 9.5E-4, 2.8E-5])

# total noise rate for IceCube Upgrade assuming the quiet mDOM noise model in mDOMs
f_noise_mDOM_upg_quiet = np.array([5E6, 1.23E4, 9.5E1, 3.5E0, 2.1E-1, 1.2E-2, 6.3E-4, 1.9E-5])

##################################################################
############################ WLS tube ############################
##################################################################

# 50 Hz dark noise rate for standard PMT
f_noise_WLS_dark = 50
# 200 Hz noise from radioactive decay
f_noise_WLS_radio = 200

########################## IceCube-Gen2 ##########################
# number of signal events for IceCube-Gen2 assuming a 27M SN progenitor (heavy) in WLS tubes with 1-fold conincidence
sig_WLS_gen2_heavy = sig_mDOM_gen2_heavy[0] * WLS2_mDOM_ratio

# number of signal events for IceCube-Gen2 assuming a 9.6M SN progenitor (light) in WLS tubes with 1-fold conincidence
sig_WLS_gen2_light = sig_mDOM_gen2_light[0] * WLS2_mDOM_ratio

# total noise rate for IceCube-Gen2 in WLS tubes with 1-fold conincidence
f_noise_WLS_gen2 = (f_noise_WLS_dark + f_noise_WLS_radio) * 2 * number_of_modules_gen2

######################### IceCube Upgrade ########################
# number of signal events for IceCube Upgrade assuming a 27M SN progenitor (heavy) in WLS tubes with 1-fold conincidence
sig_WLS_upg_heavy = sig_mDOM_upg_heavy[0] * WLS2_mDOM_ratio

# number of signal events for IceCube Upgrade assuming a 9.6M SN progenitor (light) in WLS tubes with 1-fold conincidence
sig_WLS_upg_light = sig_mDOM_upg_light[0] * WLS2_mDOM_ratio

# total noise rate for IceCube Upgrade in WLS tubes with 1-fold conincidence
f_noise_WLS_upg = (f_noise_WLS_dark + f_noise_WLS_radio) * 2 * number_of_modules_upg

def allocator(sensor_type, update_type, SN_type, noise_type):
    if sensor_type == 'mDOM':
        number_of_PMTs = number_of_PMTs_per_mDOM
        if update_type == 'Gen2':
            noise_rate = f_noise_mDOM_gen2
            if SN_type == 'heavy':
                signal_events = sig_mDOM_gen2_heavy
            elif SN_type == 'light':
                signal_events = sig_mDOM_gen2_light
        elif update_type == 'Update':
            if SN_type == 'heavy':
                signal_events = sig_mDOM_upg_heavy
            elif SN_type == 'light':
                signal_events = sig_mDOM_upg_heavy
            if noise_type == 'noisy':
                noise_rate = f_noise_mDOM_upg_noisy
            elif SN_type == 'quiet':
                noise_rate = f_noise_mDOM_upg_quiet
    elif sensor_type == 'WLS':
        number_of_PMTs = 1
        if update_type == 'Gen2':
            noise_rate = f_noise_WLS_gen2
            if SN_type == 'heavy':
                signal_events = sig_WLS_gen2_heavy
            elif SN_type == 'light':
                signal_events = sig_WLS_gen2_light
        elif update_type == 'Update':
            noise_rate = f_noise_WLS_upg
            if SN_type == 'heavy':
                signal_events = sig_WLS_upg_heavy
            elif SN_type == 'light':
                signal_events = sig_WLS_upg_light
    elif sensor_type == 'LOM18':
        number_of_PMTs = 18
        signal_events = None
        noise_rate = None
    elif sensor_type == 'LOM16':
        number_of_PMTs = 16
        signal_events = None
        noise_rate = None

    return number_of_PMTs, signal_events, noise_rate