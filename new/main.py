import globals
from sensors import *
from scipy.optimize import brute

def AND_optimise(input):
    trigger_number_of_PMTs, trigger_number_of_modules = input
    log_and = AND('mDOM', 'wls', trigger_number_of_PMTs, trigger_number_of_modules)

    return np.sqrt((1/log_and.detection_horizon())**2 + globals.fit_weight * (log_and.false_alarm_rate() - globals.DES_false_alarm_rate)**2)

def OR_optimise(input):
    trigger_number_of_PMTs, trigger_number_of_modules = input
    log_and = OR('mDOM', 'wls', trigger_number_of_PMTs, trigger_number_of_modules)

    return np.sqrt((1/log_and.detection_horizon())**2 + globals.fit_weight * (log_and.false_alarm_rate() - globals.DES_false_alarm_rate)**2)

if __name__ == "__main__": 
    globals.initialize()

    print(globals.fit_weight)

    slices = (slice(7,8,1), slice(2,20,1))

    res = brute(OR_optimise, ranges = slices, disp=True, finish=None, full_output=True)


'''
def AND(trigger_number_of_PMTs, trigger_number_of_modules,distance):
    mDOM = MDOM('mDOM', trigger_number_of_PMTs, trigger_number_of_modules)
    wls = WLS('WLS')
    AND_noise_rate = mDOM.noise_rate * wls.noise_rate
    AND_false_detection_probability = mDOM.false_detection_probability()*wls.false_detection_probability()
    AND_detection_probability = AND_detection_probability(distance)
    AND_false_alarm_rate = AND_false_detection_probability*AND_noise_rate*year
    AND_significance = AND_significance(distance)
    AND_detection_horizon = AND_detection_horizon()
    f = lambda x : 

    return AND_false_alarm_rate, AND_detection_horizon


#### LOGICAL OR ####
def OR(trigger_number_of_PMTs, trigger_number_of_modules):
    mDOM = MDOM('mDOM', trigger_number_of_PMTs, trigger_number_of_modules)
    wls = WLS('WLS')
    OR_noise_rate = mDOM.noise_rate + wls.noise_rate
    OR_significance = stouffers([mDOM.signifance, wls.significance])
    OR_detection_probability = significance_to_probability(OR_significance)
    OR_false_detection_probability = 1 - OR_detection_probability
    OR_false_alarm_rate = OR_false_detection_probability*OR_noise_rate*year
'''