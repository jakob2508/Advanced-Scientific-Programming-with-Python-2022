import numpy as np
from scipy import special
from scipy.optimize import minimize
from scipy.stats import poisson

def cumulative_poissonian(k, mu):
    '''
    Calculates the cumulative Poissonian.

    Returns the sum from 0 to k of a Poissonian probability distribution function centered around mu.

    Parameters:
    -----------
    k : interger
    maximum k for until which to sum up to

    mu: float
    mean value of poissonian distribution

    Returns:
    --------
    cumulative_poissonian : float
    cumulative poissonian

    See Also:
    ---------
    scipy.stats.poisson.cdf

    Examples:
    ---------
    >>> cumulative_poissonian(3,5)                      #doctest: +SKIP
    0.2650259152973616
    >>> cumulative_poissonian(0,1)                      #doctest: +SKIP
    1.0
    '''
    dist = poisson(mu)
    cumulative_poissonian = dist.cdf(k)
    return cumulative_poissonian

def stouffer(significances, widths):
    '''
    Combines independent significances.

    Returns the weighted sum of a list of individual significances from independent measurements using Stouffer's method.

    Parameters:
    -----------
    significances : list or array of floats
    list of significances of the independent measurements

    widths: list or array of floats
    list of widths of the independent measurements

    Returns:
    --------
    combined_significance : float
    combined significance

    Examples:
    ---------
    >>> stouffer([10,5],[1,1])                          #doctest: +SKIP
    10.606601717798211
    >>> stouffer([1,4],[1,0.5])                         #doctest: +SKIP
    4.08248290463863
    '''
    combined_significance = np.sum(significances)/np.sqrt(np.sum(widths**2))
    return combined_significance
    

def significance_to_probability(significance):
    '''
    Transforms significance into probability.

    Returns the probability (p-value) for a given significance assuming a Gaussian distribution. Calculates the cumulative probability density function in the interval from -significance to +significance.

    Parameters:
    -----------
    significances : float
    significance in multiples of the Gaussian standard deviation

    Returns:
    --------
    probability : float
    probability (p-value)

    Examples:
    ---------
    >>>  significance_to_probability(1)                 #doctest: +SKIP
    0.6826894921370859
    >>>  significance_to_probability(2)                 #doctest: +SKIP
    0.9544997361036416          
    >>>  significance_to_probability(3)                 #doctest: +SKIP
    0.9973002039367398
    '''
    probability = special.ndtr(significance) - special.ndtr(-significance)
    return probability
    
def obj_probability_to_significance(significance, probability):
    '''
    Objective function for minimization in probability_to_significance.

    Returns the squared difference between the input probability and the corresponding significance probability from significance_to_probabilty(significance).
    '''
    return (probability-significance_to_probability(significance))**2

def probability_to_significance(probability):
    '''
    Transforms probability into significance.

    Returns the probability (p-value) for a given significance assuming a Gaussian distribution. As there is no analytic formular for the reverse probability, we use scipy.optimize.minimize to minimize the objective function obj_probability_to_significance.

    Parameters:
    -----------
    probability : float
    probability (p-value)

    Returns:
    --------
    significances : float
    significance in multiples of the Gaussian standard deviation

    See Also:
    ---------
    scipy.optimize.minimize
    obj_probability_to_significance

    Examples:
    ---------
    >>>  probability_to_significance(0.68)              #doctest: +SKIP
    0.9944580078125
    >>>  probability_to_significance(0.95)              #doctest: +SKIP
    1.9599639892578145          
    >>>  probability_to_significance(0.995)             #doctest: +SKIP
    2.807033538818364
    '''
    res = minimize(obj_probability_to_significance, x0 = 1, args = (probability), method='Nelder-Mead', tol = 1E-6)
    significance = float(res.x)
    return significance

def detection_probability(trigger_number_of_modules, signal_at_distance):
    '''
    Calculates the detection probability of a segmented sensor.

    Returns the probability for trigger_number_of_modules events in the detector to trigger a detection if signal_at_distance signal events are expected from a CCSN at a specific distance.

    Parameters:
    -----------
    trigger_number_of_modules : integer
    number of modules to be hit

    signal_at_distance : float
    extrapolated signal events at a specific distance

    Returns:
    --------
    detection_probability : float
    detection probability

    See Also:
    ---------
    cumulative_poissonian
    false_detection_probability

    Examples:
    ---------
    >>>  detection_probability(10,10)                 #doctest: +SKIP
    0.5420702855281476
    '''
    detection_probability = (1 - cumulative_poissonian(trigger_number_of_modules - 1, signal_at_distance))
    return detection_probability

def false_detection_probability(trigger_number_of_modules, background_events):
    '''
    Calculates the false detection probability of a segmented sensor.

    Returns the probability for trigger_number_of_modules events in the detector to originate from backhround if one expects background_events background events.

    Parameters:
    -----------
    trigger_number_of_modules : integer
    number of modules to be hit

    background_events : float
    background events

    Returns:
    --------
    false_detection_probability : float
    false detection probability

    See Also:
    ---------
    cumulative_poissonian
    detection_probability

    Examples:
    ---------
    >>>  detection_probability(10,5)                 #doctest: +SKIP
    0.06809363472184837
    '''
    false_detection_probability = (1 - cumulative_poissonian(trigger_number_of_modules - 2, background_events))
    return false_detection_probability
