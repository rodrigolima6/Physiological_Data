from lib.biosignals import *
from lib.acquisition import *
from lib.sensors import *

def Process_ECG(data,fs,resolution):

    sensor = ECG(data,fs,resolution)
    ecg = np.array(sensor.data)
    fs = sensor.fs
    resolution = sensor.resolution
    time = bsnb.generate_time(ecg,fs)

    peaks = sensor.processECG()

    return peaks,time

def Process_HRV(data,fs,resolution):
    sensor = HRV(data,fs,resolution)

    time_features, poincare_features, frequency_features = sensor.getFeatures()

    return time_features, poincare_features, frequency_features