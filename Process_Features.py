import pandas as pd

from lib.sensors import *
from Load import *
from Epochs import *
import pyxdf


def getEvents(
    users,
    openSignals_stream_name: str,
    EEG_stream_name: str,
    markers_stream_name: str,
    ratings_stream_name,
    sensors: list,
):

    data = {}
    for user in users.keys():
        data[user.split(".")[0]] = Load_Data(
            users[user],
            openSignals_stream_name,
            EEG_stream_name,
            markers_stream_name,
            ratings_stream_name,
            sensors,
        )
    # for user in data.keys():
    #     if "baseline" in data[user]["Markers"][0]:
    #         data[user]["Markers"][0][1] = "0"

    onset, offset, videos, valence, arousal = ({}, {}, {}, {}, {})
    for user in data.keys():
        valence[user] = data[user]["Valence"]
        arousal[user] = data[user]["Arousal"]
        onset[user], offset[user], videos[user] = getMarkers(
            data[user]["Markers"], data[user]["Markers Timestamps"]
        )

    onset_index = {}
    offset_index = {}
    onset_index_EEG = {}
    offset_index_EEG = {}

    for user in data.keys():
        onset_index[user], offset_index[user] = getMarkersIndex(
            onset[user], offset[user], data[user]["Signals"]["Time"]
        )
        onset_index_EEG[user], offset_index_EEG[user] = getMarkersIndex(
            onset[user], offset[user], np.array(data[user]["EEG"]["Time"])
        )

    events_diff = {}

    for keys in onset.keys():
        events_diff[keys] = CalculateEventsDiff(onset[keys], offset[keys])

    return (
        events_diff,
        videos,
        onset,
        offset,
        onset_index,
        offset_index,
        onset_index_EEG,
        offset_index_EEG,
        data,
        valence,
        arousal,
    )


def Run_files(fname):
    data, header = pyxdf.load_xdf(fname)

    return data


def Load_Data(
    data,
    openSignals_stream_name: str,
    EEG_stream_name: str,
    markers_stream_name: str,
    ratings_stream_name: str,
    sensors: list,
):
    Signals, EEG_Signals = pd.DataFrame(), pd.DataFrame()

    marker, timestamps = Load_PsychopyMarkers(data, markers_stream_name)
    valence, arousal = Load_Ratings(data, ratings_stream_name)
    opensignals_data, fs = Load_Opensignals(data, openSignals_stream_name)
    EEG_data, time_EEG, EEG_fs = Load_EEG(data, EEG_stream_name)

    if len(opensignals_data.keys()) > 0:
        Signals = pd.DataFrame(data=opensignals_data)
        sensors.insert(0, "Time")
        Signals.columns = sensors
    if len(EEG_data.keys()) > 0:
        EEG_Signals = pd.DataFrame.from_dict(EEG_data)
        EEG_Signals.insert(0, "Time", time_EEG)

    return {
        "Signals": Signals,
        "EEG": EEG_Signals,
        "Markers": marker,
        "Markers Timestamps": timestamps,
        "Valence": valence,
        "Arousal": arousal,
    }


def getDataframe(dataframe, fs, resolution):
    HRV_Dataframe = Process_HRV(dataframe["ECG"], fs, resolution)
    RESP_Dataframe = Process_RESP(dataframe["RESP"], fs, resolution)
    EDA_Dataframe = Process_EDA(dataframe["EDA"], fs, resolution)
    Dataframe = (HRV_Dataframe.join(EDA_Dataframe)).join(RESP_Dataframe)

    return Dataframe


def Process_ECG(data, fs, resolution):

    sensor = ECG(data, fs, resolution)
    ecg = np.array(sensor.data)
    fs = sensor.fs
    resolution = sensor.resolution
    time = bsnb.generate_time(ecg, fs)

    peaks = sensor.processECG()

    return peaks, time


def Process_HRV(data, fs, resolution):
    sensor = HRV(data, fs, resolution)

    (
        heart_rate,
        time_features,
        poincare_features,
        frequency_features,
    ) = sensor.getFeatures()
    # print(heart_rate)
    # print(time_features)
    # print(poincare_features)
    # print(frequency_features)

    heart_rate_df = pd.DataFrame.from_dict(heart_rate, orient="columns")
    time_features_df = pd.DataFrame.from_dict(time_features, orient="columns")
    poincare_features_df = pd.DataFrame.from_dict(poincare_features, orient="columns")
    frequency_features_df = pd.DataFrame.from_dict(frequency_features, orient="columns")

    HRV_Dataframe = (
        (heart_rate_df.join(time_features_df)).join(poincare_features_df)
    ).join(frequency_features_df)

    return HRV_Dataframe


# def Process_fNIRS(data,fs,resolution):
#
#     sensor = fNIRS(data,fs,resolution)
#
#     sensor.processfNIRS()
#
#     fnirs_features = sensor.getFeatures()
#
#     fNIRS_Dataframe = pd.DataFrame.from_dict(fnirs_features,orient="columns")
#
#     return fNIRS_Dataframe


def Process_RESP(data, fs, resolution):

    sensor = RESP(data, fs, resolution)

    signals, info = sensor.process_RESP()

    # df = sensor.RESP_RRV(signals)

    resp_Dataframe = sensor.getFeatures(signals)  # , df)

    columns_to_remove = [
        "RRV_VLF",
        "RRV_LF",
        "RRV_LFHF",
        "RRV_LFn",
        "RRV_HFn",
        "RRV_SD2",
        "RRV_SD2SD1",
    ]

    for column in columns_to_remove:
        if column in resp_Dataframe.columns:
            resp_Dataframe = resp_Dataframe.drop(column, axis=1)

    return resp_Dataframe


def Process_EDA(data, fs, resolution):

    sensor = EDA(data, fs, resolution)

    (
        eda_phasic_dict,
        eda_tonic_dict,
        SCR_Amplitude_dict,
        SCR_RiseTime_dict,
        SCR_RecoveryTime_dict,
        frequency_features,
    ) = sensor.getFeatures()

    EDA_dict = {
        "Phasic_AVG": eda_phasic_dict["AVG"],
        "Phasic_MAX": eda_phasic_dict["Maximum"],
        "Phasic_MIN": eda_phasic_dict["Minimum"],
        "Phasic_STD": eda_phasic_dict["STD"],
        "Tonic_AVG": eda_tonic_dict["AVG"],
        "Tonic_MAX": eda_tonic_dict["Maximum"],
        "Tonic_MIN": eda_tonic_dict["Minimum"],
        "Tonic_STD": eda_tonic_dict["STD"],
        "SCR_Amp_AVG": SCR_Amplitude_dict["AVG"],
        "SCR_Amp_MAX": SCR_Amplitude_dict["Maximum"],
        "SCR_Amp_MIN": SCR_Amplitude_dict["Minimum"],
        "SCR_Amp_STD": SCR_Amplitude_dict["STD"],
        "SCR_Rt_AVG": SCR_RiseTime_dict["AVG"],
        "SCR_Rt_MAX": SCR_RiseTime_dict["Maximum"],
        "SCR_Rt_MIN": SCR_RiseTime_dict["Minimum"],
        "SCR_Rt_STD": SCR_RiseTime_dict["STD"],
        "SCR_Rect_AVG": SCR_RecoveryTime_dict["AVG"],
        "SCR_Rect_MAX": SCR_RecoveryTime_dict["Maximum"],
        "SCR_Rect_MIN": SCR_RecoveryTime_dict["Minimum"],
        "SCR_Rect_STD": SCR_RecoveryTime_dict["STD"],
    }

    EDA_Dataframe = (pd.DataFrame.from_dict(EDA_dict)).join(
        pd.DataFrame.from_dict(frequency_features)
    )
    if "LF/HF" in EDA_Dataframe.columns:
        EDA_Dataframe = EDA_Dataframe.drop(["LF/HF"], axis=1)

    return EDA_Dataframe


def Process_EEG(data, fs, resolution):
    EEG_dict = {}
    EEG_filtered = {}
    band_powers = {}
    freqs = {}
    power = {}

    for keys in data.keys():
        EEG_dict[keys] = EEG(data[keys], fs, resolution)
        EEG_filtered[keys], freqs[keys], power[keys], band_powers[keys] = EEG_dict[
            keys
        ].getFeatures()

    bands_df = pd.DataFrame.from_dict(band_powers, orient="index")

    return bands_df


# def Process_TEMP(data, fs, resolution):
#     sensor = TEMP(data, fs, resolution)
#
#     temp = sensor.filterData()
#
#     Temp_Dataframe = sensor.getFeatures(temp)
#
#     return Temp_Dataframe
