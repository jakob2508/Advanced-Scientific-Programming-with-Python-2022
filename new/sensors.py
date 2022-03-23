import numpy as np
import globals
from datasheet import *
from stat_functions import *
from scipy.optimize import minimize

class Sensor(object):
    def __init__(self, name, sensor_type, update_type = 'Gen2', SN_type = 'heavy', noise_type = None):
        self.sensor_type = sensor_type
        self.name = name
        self.update_type = update_type
        self.SN_type = SN_type
        self.noise_type = noise_type
    @property
    def background_events(self):
        return self.noise_rate * dT_SN

    def signal_at_distance(self, distance, distance_0 = distance_0):
        return self.signal_events * (distance_0/distance)**2

    def significance(self, distance, type = "poisson"):
        s = self.signal_at_distance(distance)
        b = self.background_events
        n = s + b
        if type == "poisson":
            Z = np.sqrt(2*(n*np.log(n/b)-(n-b)))
        elif type == "gauss":
            Z = np.abs(n-s)/np.sqrt(b)
        return Z

class MDOM(Sensor):
    def __init__(self, name, trigger_number_of_PMTs, trigger_number_of_modules, sensor_type = 'mDOM', update_type = 'Gen2', SN_type = 'heavy', noise_type = None):
        super().__init__(name, sensor_type, update_type, SN_type, noise_type)
        self.trigger_number_of_PMTs = trigger_number_of_PMTs
        self.trigger_number_of_modules = trigger_number_of_modules

        number_of_PMTs, signal_events, noise_rate = allocator(sensor_type, update_type, SN_type, noise_type)

        self.number_of_PMTs = number_of_PMTs
        self.signal_events = signal_events[trigger_number_of_PMTs-1]
        self.noise_rate = noise_rate[trigger_number_of_PMTs-1]
    
    def detection_probability(self, distance):
        return detection_probability(self.trigger_number_of_modules, self.signal_at_distance(distance))

    def false_detection_probability(self):
        return false_detection_probability(self.trigger_number_of_modules, self.background_events)
    
    def false_alarm_rate(self):
        return self.false_detection_probability()*year*self.noise_rate

    def detection_horizon(self):
        thresh = 1E-3
        f = lambda x: (detection_probability(self.trigger_number_of_modules,self.signal_at_distance(x))-0.5)**2
        loss = 1
        x0 = 1
        while loss > thresh:
            res = minimize(f, x0 = x0, method='Nelder-Mead', tol = 1E-6)
            loss = (detection_probability(self.trigger_number_of_modules,self.signal_at_distance(res.x.item()))-0.5)**2
            x0 += 50
            if x0 > 1E4:
                break
        return res.x.item()

class WLS(Sensor):
    def __init__(self, name, sensor_type = 'WLS', update_type = 'Gen2', SN_type = 'heavy', noise_type = None):
        super().__init__(name, sensor_type, update_type, SN_type, noise_type)

        number_of_PMTs, signal_events, noise_rate = allocator(sensor_type, update_type, SN_type, noise_type)

        self.number_of_PMTs = number_of_PMTs
        self.signal_events = signal_events
        self.noise_rate = noise_rate

    def detection_probability(self, distance):
        return significance_to_probability(self.significance(distance))

    def false_detection_probability(self, distance):
        return 1 - self.detection_probability(self, distance)

    def false_alarm_rate(self):
        return self.false_detection_probability()*year*self.noise_rate

    def detection_horizon(self):
        thresh = 1E-3
        f = lambda x: (significance_to_probability(self.significance(x))-0.5)**2
        loss = 1
        x0 = 1
        while loss > thresh:
            res = minimize(f, x0 = x0, method='Nelder-Mead', tol = 1E-6)
            loss = (significance_to_probability(self.significance(res.x.item()))-0.5)**2
            x0 += 50
            if x0 > 1E4:
                break
        return res.x.item()
   
class AND(object):
    def __init__(self, MDOM_name, WLS_name, trigger_number_of_PMTs, trigger_number_of_modules, MDOM_sensor_type = 'mDOM', WLS_sensor_type = 'WLS', update_type = 'Gen2', SN_type = 'heavy', noise_type = None):
        
        mdom = MDOM(MDOM_name, trigger_number_of_PMTs, trigger_number_of_modules, MDOM_sensor_type, update_type, SN_type, noise_type)
        wls = WLS(WLS_name, WLS_sensor_type, update_type, SN_type, noise_type)

        self.MDOM = mdom
        self.WLS = wls 

    @property
    def noise_rate(self):
        return self.MDOM.noise_rate + self.WLS.noise_rate

    def detection_probability(self, distance):
        return self.MDOM.detection_probability(distance) * self.WLS.detection_probability(distance)

    def false_detection_probability(self):
        wls_false_detection_probability = np.sqrt(self.noise_rate/(globals.DES_false_alarm_rate*year))
        return self.MDOM.false_detection_probability() * wls_false_detection_probability

    def false_alarm_rate(self):
        return self.false_detection_probability() * self.noise_rate * year

    def significance(self, distance):
        return probability_to_significance(self.detection_probability(distance))

    def detection_horizon(self):
        thresh = 1E-3
        f = lambda x: ((self.MDOM.detection_probability(x)*self.WLS.detection_probability(x))-0.5)**2
        loss = 1
        x0 = 1
        while loss > thresh:
            res = minimize(f, x0 = x0, method='Nelder-Mead', tol = 1E-6)
            loss = (self.detection_probability(res.x.item())-0.5)**2
            x0 += 50
            if x0 > 1E4:
                break
        return res.x.item()

class OR(object):
    def __init__(self, MDOM_name, WLS_name, trigger_number_of_PMTs, trigger_number_of_modules, MDOM_sensor_type = 'mDOM', WLS_sensor_type = 'WLS', update_type = 'Gen2', SN_type = 'heavy', noise_type = None):
        
        mdom = MDOM(MDOM_name, trigger_number_of_PMTs, trigger_number_of_modules, MDOM_sensor_type, update_type, SN_type, noise_type)
        wls = WLS(WLS_name, WLS_sensor_type, update_type, SN_type, noise_type)

        self.MDOM = mdom
        self.WLS = wls 

    def significance(self, distance):
        return stouffer([self.MDOM.significance(distance), self.WLS.significance(distance)], [1,1])

    def detection_probability(self, distance):
        return significance_to_probability(self.significance(distance))

    def false_alarm_rate(self):
        return self.MDOM.false_alarm_rate() * self.WLS.false_alarm_rate()

    def detection_horizon(self):
        thresh = 1E-3
        f = lambda x: (significance_to_probability(self.significance(x))-0.5)**2
        loss = 1
        x0 = 1
        while loss > thresh:
            res = minimize(f, x0 = x0, method='Nelder-Mead', tol = 1E-6)
            loss = (self.detection_probability(res.x.item())-0.5)**2
            x0 += 50
            if x0 > 1E4:
                break
        return res.x.item()