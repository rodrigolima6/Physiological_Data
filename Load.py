import pyxdf


@staticmethod
def Load_PsychopyMarkers(data, stream_name: str):
    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            markers = list(stream["time_series"])
            timestamps = list(stream["time_stamps"])

    return markers, timestamps


@staticmethod
def Load_Ratings(data, stream_name: str) -> tuple:
    valence, arousal = list(), list()
    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            timeseries = list(stream["time_series"])
            timestamps = list(stream["time_stamps"])

    for item in timeseries:
        if float(item[1]) < 4:
            if item[0] == "Valence":
                valence.append("Low")
            else:
                arousal.append("Low")
        elif float(item[1]) >= 4 and float(item[1]) < 7:
            if item[0] == "Valence":
                valence.append("Medium")
            else:
                arousal.append("Medium")
        else:
            if float(item[0]) == "Valence":
                valence.append("High")
            else:
                arousal.append("High")

    return valence, arousal


def Load_Opensignals(data, stream_name: str):
    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            CH1 = stream["time_series"][:, 1]  # ECG
            CH2 = stream["time_series"][:, 2]  # EDA
            CH3 = stream["time_series"][:, 3]  # RESP
            CH4 = stream["time_series"][:, 4]  # TEMP
            CH5 = stream["time_series"][:, 5]  # fnirs RED
            CH6 = stream["time_series"][:, 6]  # fnirs IRED
            time_Opensignals = stream["time_stamps"]
            fs = int(stream["info"]["nominal_srate"][0])

    return CH1, CH2, CH3, CH4, CH5, CH6, time_Opensignals, fs


def Load_EEG(data, stream_name: str):

    EEG_data = {}

    for stream in data:
        if stream["info"]["name"][0] == stream_name:
            for i in range(0, 32):
                # print(i)
                EEG_data["EEG_" + str(i + 1)] = stream["time_series"][:, i]
            time_EEG = stream["time_stamps"]
            EEG_fs = int(stream["info"]["nominal_srate"][0])

    return EEG_data, time_EEG, EEG_fs


# data, header = pyxdf.load_xdf('G:\\O meu disco\\PhD\\1st Study\\data\\P8_S1_GroupA_eeg.xdf')
# EEG,time_EEG,EEG_fs = Load_EEG(data)
