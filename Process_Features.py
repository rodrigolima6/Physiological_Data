import pyxdf
from lib.sensors import *
from Load import *
from Epochs import *
from math import *


def getEvents(users):
    # Signals_P1, EEG_Signals_P1, marker_P1, timestamps_P1 = correctP1(users)
    # Signals_P2, EEG_Signals_P2, marker_P2, timestamps_P2 = correctP2(users)
    #
    # del users["P1_S2_GroupA_eeg_1.xdf"]
    # del users["P2_S1_GroupA_eeg_1.xdf"]

    data = {}
    for user in users.keys():
        data[user.split("_")[0] + "_" + user.split("_")[1]] = Load_Data(users[user])

    # data["P1_S2"] = (Signals_P1, EEG_Signals_P1, marker_P1, timestamps_P1)
    # data["P2_S1"] = (Signals_P2, EEG_Signals_P2, marker_P2, timestamps_P2)
    #
    # data["P4_S2"][1]["Time"] = np.arange(
    #     data["P4_S2"][0]["Time"][1],
    #     np.array(data["P4_S2"][0]["Time"])[-1],
    #     (np.array(data["P4_S2"][0]["Time"])[-1] - data["P4_S2"][0]["Time"][0])
    #     / len(data["P4_S2"][1]["Time"]),
    # )

    for keys in data.keys():
        if "baseline" in data[keys][2][0]:
            data[keys][2][0][1] = "0"

    onset = {}
    offset = {}
    videos = {}

    for keys in data.keys():
        onset[keys], offset[keys], videos[keys] = getMarkers(
            data[keys][2], data[keys][3]
        )

    onset_index = {}
    offset_index = {}
    onset_index_EEG = {}
    offset_index_EEG = {}

    for keys in data.keys():
        # print(keys)
        onset_index[keys], offset_index[keys] = getMarkersIndex(
            onset[keys], offset[keys], data[keys][0]["Time"]
        )
        onset_index_EEG[keys], offset_index_EEG[keys] = getMarkersIndex(
            onset[keys], offset[keys], np.array(data[keys][1]["Time"])
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
    )


def Run_files(fname):
    data, header = pyxdf.load_xdf(fname)

    return data


def Load_Data(data):
    marker, timestamps = Load_Psychopy(data)
    CH1, CH2, CH3, CH4, CH5, CH6, time_Opensignals, fs = Load_Opensignals(data)
    EEG_data, time_EEG, EEG_fs = Load_EEG(data)

    # time_EEG = np.arange(time_Opensignals[1],time_Opensignals[-1],(time_Opensignals[-1]-time_Opensignals[0])/len(time_EEG))

    # timestamps -= time_Opensignals[0]
    # time_Opensignals -= time_Opensignals[0]
    # time_EEG -= time_EEG[0]

    d = {
        "Time": time_Opensignals,
        "ECG": CH1,
        "EDA": CH2,
        "RESP": CH3,
        "TEMP": CH4,
        "fNIRS_RED": CH5,
        "fNIRS_IRED": CH6,
    }
    Signals = pd.DataFrame(data=d)

    EEG_Signals = pd.DataFrame.from_dict(EEG_data)
    EEG_Signals.insert(0, "Time", time_EEG)

    return Signals, EEG_Signals, marker, timestamps


def getDataframe(dataframe, fs, resolution):
    HRV_Dataframe = Process_HRV(dataframe["ECG"], fs, resolution)
    # Temp_Dataframe = Process_TEMP(dataframe["TEMP"], fs, resolution)
    # fNIRS_Dataframe = Process_fNIRS(
    #     np.vstack((dataframe["fNIRS_RED"], dataframe["fNIRS_IRED"])).T, fs, resolution
    # )
    RESP_Dataframe = Process_RESP(dataframe["RESP"], fs, resolution)
    EDA_Dataframe = Process_EDA(dataframe["EDA"], fs, resolution)

    # Dataframe = (
    #     ((HRV_Dataframe.join(EDA_Dataframe)).join(RESP_Dataframe)).join(fNIRS_Dataframe)
    # ).join(Temp_Dataframe)
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

    df = sensor.RESP_RRV(signals)

    resp_Dataframe = sensor.getFeatures(signals, df)
    resp_Dataframe = resp_Dataframe.drop(
        [
            "RRV_VLF",
            "RRV_LF",
            "RRV_LFHF",
            "RRV_LFn",
            "RRV_HFn",
            "RRV_SD2",
            "RRV_SD2SD1",
        ],
        axis=1,
    )
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
        "Phasic_AVG": [eda_phasic_dict["AVG"]],
        "Phasic_MAX": [eda_phasic_dict["Maximum"]],
        "Phasic_MIN": [eda_phasic_dict["Minimum"]],
        "Phasic_STD": [eda_phasic_dict["STD"]],
        "Tonic_AVG": [eda_tonic_dict["AVG"]],
        "Tonic_MAX": [eda_tonic_dict["Maximum"]],
        "Tonic_MIN": [eda_tonic_dict["Minimum"]],
        "Tonic_STD": [eda_tonic_dict["STD"]],
        "SCR_Amp_AVG": [SCR_Amplitude_dict["AVG"]],
        "SCR_Amp_MAX": [SCR_Amplitude_dict["Maximum"]],
        "SCR_Amp_MIN": [SCR_Amplitude_dict["Minimum"]],
        "SCR_Amp_STD": [SCR_Amplitude_dict["STD"]],
        "SCR_Rt_AVG": [SCR_RiseTime_dict["AVG"]],
        "SCR_Rt_MAX": [SCR_RiseTime_dict["Maximum"]],
        "SCR_Rt_MIN": [SCR_RiseTime_dict["Minimum"]],
        "SCR_Rt_STD": [SCR_RiseTime_dict["STD"]],
        "SCR_Rect_AVG": [SCR_RecoveryTime_dict["AVG"]],
        "SCR_Rect_MAX": [SCR_RecoveryTime_dict["Maximum"]],
        "SCR_Rect_MIN": [SCR_RecoveryTime_dict["Minimum"]],
        "SCR_Rect_STD": [SCR_RecoveryTime_dict["STD"]],
    }

    EDA_Dataframe = (pd.DataFrame.from_dict(EDA_dict)).join(
        pd.DataFrame.from_dict(frequency_features)
    )
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


def Process_TEMP(data, fs, resolution):
    sensor = TEMP(data, fs, resolution)

    temp = sensor.filterData()

    Temp_Dataframe = sensor.getFeatures(temp)

    return Temp_Dataframe


def correctP1(data):

    markers1, timestamps1 = Load_Psychopy(data["P1_S2_GroupA_eeg_1.xdf"])
    markers2, timestamps2 = Load_Psychopy(data["P1_S2_GroupA_eeg.xdf"])

    Signals, EEG_Signals, marker, timestamps = Load_Data(data["P1_S2_GroupA_eeg.xdf"])
    Signals_1, EEG_Signals_1, marker_1, timestamps_1 = Load_Data(
        data["P1_S2_GroupA_eeg_1.xdf"]
    )

    markers2.reverse()
    timestamps2.reverse()

    for element in markers2:
        markers1.insert(0, element)

    for timestamp in timestamps2:
        timestamps1.insert(0, timestamp)

    Signals_new = pd.concat([Signals, Signals_1], ignore_index=True)
    EEG_Signals = pd.concat([EEG_Signals, EEG_Signals_1], ignore_index=True)
    marker = markers1
    timestamps = timestamps1

    return Signals_new, EEG_Signals, marker, timestamps


def correctP2(data):

    Signals, EEG_Signals, markers4, timestamps4 = Load_Data(
        data["P2_S1_GroupA_eeg.xdf"]
    )

    (
        CH1_1,
        CH2_1,
        CH3_1,
        CH4_1,
        CH5_1,
        CH6_1,
        time_Opensignals_1,
        fs_1,
    ) = Load_Opensignals(data["P2_S1_GroupA_eeg_1.xdf"])
    EEG_data_1, time_EEG_1, EEG_fs_1 = Load_EEG(data["P2_S1_GroupA_eeg_1.xdf"])

    d = {
        "Time": time_Opensignals_1,
        "ECG": CH1_1,
        "EDA": CH2_1,
        "RESP": CH3_1,
        "TEMP": CH4_1,
        "fNIRS_RED": CH5_1,
        "fNIRS_IRED": CH6_1,
    }
    Signals_1 = pd.DataFrame(data=d)

    EEG_Signals_1 = pd.DataFrame.from_dict(EEG_data_1)
    EEG_Signals_1.insert(0, "Time", time_EEG_1)

    Signals_new = pd.concat([Signals_1, Signals], ignore_index=True)
    EEG_Signals = pd.concat([EEG_Signals_1, EEG_Signals], ignore_index=True)

    missing_markers = [
        ["EMDB/A/Scenery/5003.avi", "1"],
        ["EMDB/A/Scenery/5003.avi", "0"],
        ["EMDB/A/Scenery/5001.avi", "1"],
        ["EMDB/A/Scenery/5001.avi", "0"],
        ["EMDB/A/Scenery/5000.avi", "1"],
        ["EMDB/A/Scenery/5000.avi", "0"],
        ["EMDB/A/Scenery/5002.avi", "1"],
        ["EMDB/A/Scenery/5002.avi", "0"],
        ["EMDB/A/Erotic/2004.avi", "1"],
        ["EMDB/A/Erotic/2004.avi", "0"],
        ["EMDB/A/Erotic/2002.avi", "1"],
        ["EMDB/A/Erotic/2002.avi", "0"],
        ["EMDB/A/Erotic/2003.avi", "1"],
        ["EMDB/A/Erotic/2003.avi", "0"],
        ["EMDB/A/Erotic/2000.avi", "1"],
        ["EMDB/A/Erotic/2000.avi", "0"],
        ["EMDB/A/Erotic/2001.avi", "1"],
        ["EMDB/A/Erotic/2001.avi", "0"],
    ]

    missing_timestamps = [
        (timestamps4[0] - 78.49) + 40.45737604831811,
        timestamps4[0] - 78.49,
        (timestamps4[0] - 162) + 40.217181624873774,
        timestamps4[0] - 162,
        (timestamps4[0] - 247) + 40.61692033614963,
        timestamps4[0] - 247,
        (timestamps4[0] - 345) + 40.37409253764781,
        timestamps4[0] - 345,
        (timestamps4[0] - 427) + 40.13403391290922,
        timestamps4[0] - 427,
        (timestamps4[0] - 512) + 40.46176630666014,
        timestamps4[0] - 512,
        (timestamps4[0] - 602) + 40.653548489004606,
        timestamps4[0] - 602,
        (timestamps4[0] - 690) + 40.130320348078385,
        timestamps4[0] - 690,
        (timestamps4[0] - 789) + 40.13610309327487,
        timestamps4[0] - 789,
    ]

    for marker in missing_markers:
        markers4.insert(0, marker)
    for timestamps in missing_timestamps:
        timestamps4.insert(0, timestamps)

    return Signals_new, EEG_Signals, markers4, timestamps4
