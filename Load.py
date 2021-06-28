import pyxdf

def Load_Psychopy(data):
    for stream in data:
        if (stream['info']['name'][0] == "PsychoPyStream"):
            marker = stream['time_series']
            timestamps = stream['time_stamps']

    return marker,timestamps

def Load_Opensignals(data):
    for stream in data:
        if (stream['info']['name'][0] == "OpenSignals"):
            CH1 = stream['time_series'][:, 1]  # ECG
            CH2 = stream['time_series'][:, 2]  # EDA
            CH3 = stream['time_series'][:, 3]  # RESP
            CH4 = stream['time_series'][:, 4]  # fNIRS
            # CH5 = stream['time_series'][:,5] #TEMP
            time_Opensignals = stream['time_stamps']

    return CH1,CH2,CH3,CH4,time_Opensignals

def Load_EEG(data):

    EEG_data={}

    for stream in data:
        if (stream['info']['name'][0] == "openvibeSignal"):
            for i in range(0, 32):
                EEG_data["EEG_" + str(i + 1)] = stream['time_series'][:, i]
            time_EEG = stream['time_stamps']

    return EEG_data,time_EEG
