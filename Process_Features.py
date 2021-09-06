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

def Process_EDA(data,fs,resolution):

    sensor = EDA(data,fs,resolution)

    eda_phasic_dict, eda_tonic_dict, SCR_Amplitude_dict, SCR_RiseTime_dict,SCR_RecoveryTime_dict, frequency_features = sensor.getFeatures()

    EDA_dict = {"Phasic_AVG":[eda_phasic_dict["AVG"]],"Phasic_MAX":[eda_phasic_dict["Maximum"]],"Phasic_MIN":[eda_phasic_dict["Minimum"]],
                "Phasic_STD":[eda_phasic_dict["STD"]],
                "Tonic_AVG": [eda_tonic_dict["AVG"]], "Tonic_MAX": [eda_tonic_dict["Maximum"]],
                "Tonic_MIN": [eda_tonic_dict["Minimum"]],
                "Tonic_STD": [eda_tonic_dict["STD"]],
                "SCR_Amp_AVG": [SCR_Amplitude_dict["AVG"]], "SCR_Amp_MAX": [SCR_Amplitude_dict["Maximum"]],
                "SCR_Amp_MIN": [SCR_Amplitude_dict["Minimum"]],
                "SCR_Amp_STD": [SCR_Amplitude_dict["STD"]],
                "SCR_Rt_AVG": [SCR_RiseTime_dict["AVG"]], "SCR_Rt_MAX": [SCR_RiseTime_dict["Maximum"]],
                "SCR_Rt_MIN": [SCR_RiseTime_dict["Minimum"]],
                "SCR_Rt_STD": [SCR_RiseTime_dict["STD"]],
                "SCR_Rect_AVG": [SCR_RecoveryTime_dict["AVG"]], "SCR_Rect_MAX": [SCR_RecoveryTime_dict["Maximum"]],
                "SCR_Rect_MIN": [SCR_RecoveryTime_dict["Minimum"]],
                "SCR_Rect_STD": [SCR_RecoveryTime_dict["STD"]],
                }

    EDA_Dataframe = (pd.DataFrame.from_dict(EDA_dict)).join(pd.DataFrame.from_dict(frequency_features))

    return EDA_Dataframe
