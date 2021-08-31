from Physiological_Data.lib.sensors import *
import numpy as np
from lib.sensors import *
import pandas as pd
from scipy.integrate import simps

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

    heart_rate,time_features, poincare_features, frequency_features = sensor.getFeatures()

    heart_rate_df = pd.DataFrame.from_dict(heart_rate,orient='columns')
    time_features_df = pd.DataFrame.from_dict(time_features,orient='columns')
    poincare_features_df = pd.DataFrame.from_dict(poincare_features,orient='columns')
    frequency_features_df = pd.DataFrame.from_dict(frequency_features,orient='columns')

    HRV_Dataframe= ((heart_rate_df.join(time_features_df)).join(poincare_features_df)).join(frequency_features_df)

    return HRV_Dataframe

def Process_fNIRS(data,fs,resolution):

    sensor = fNIRS(data,fs,resolution)

    sensor.processfNIRS()

    fnirs_features = sensor.getFeatures()

    fNIRS_Dataframe = pd.DataFrame.from_dict(fnirs_features,orient="columns")

    return fNIRS_Dataframe

def Process_RESP(data,fs,resolution):

    sensor = RESP(data,fs,resolution)

    signals,info = sensor.process_RESP()

    df = sensor.RESP_RRV(signals)

    resp_Dataframe = sensor.getFeatures(signals,df)

    return resp_Dataframe
