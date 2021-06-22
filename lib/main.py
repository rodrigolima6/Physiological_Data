try:
    from biosignals import *
    from psychopy import *
    from acquisition import *
except ModuleNotFoundError:
    from Physiological_Data.lib.biosignals import *
    from Physiological_Data.lib.psychopy import *
    from Physiological_Data.lib.acquisition import *

import matplotlib.pylab as plt

psycho = Psycho(r'..\..\acquisitions\Acquisitions\11_12_2020_1\results_3.csv')
device = Devices(r'..\..\acquisitions\Acquisitions\11_12_2020_1')

timestamps = psycho.getTimestamps()
labels = psycho.getActivity()
device.segmentSignals(device.data[:, 0], timestamps, labels)

# plt.figure()
# plt.plot(timestamps, np.zeros(len(timestamps)))
# plt.plot(device.data[:, 0], np.zeros(len(device.data[:, 0])))
# plt.show()
