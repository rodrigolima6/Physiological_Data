import pyxdf
import matplotlib.pyplot as plt
import numpy as np

data, header = pyxdf.load_xdf('C:/Users/Rodrigo/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf')

plt.figure()
for stream in data:
    if(stream['info']['name'][0] == "PsychoPyStream"):
        y=stream['time_series']
        for i in range(0,len(stream['time_stamps']),2):
            print(stream['time_stamps'][i+1]-stream['time_stamps'][i])
        for timestamp,marker in zip(stream['time_stamps'],y):
            if (marker[1] == '1'):
                plt.axvline(x=timestamp,color='r')
            elif(marker[1] == '0'):
                plt.axvline(x=timestamp,color='g')
    elif(stream['info']['name'][0] == "OpenSignals"):
        for i in range(1,np.array(stream['time_series']).shape[1]):
            plt.plot(stream['time_stamps'],stream['time_series'][:,i],label="OpenSignals_CH"+str(i))
    elif(stream['info']['name'][0] == "EEG-EEG"):
        for i in range(1,np.array(stream['time_series']).shape[1]):
            plt.plot(stream['time_stamps'],stream['time_series'][:,i],label="EEG_CH"+str(i))
    else:
        raise RuntimeError('Unknown Stream format')

plt.legend()
plt.show()

