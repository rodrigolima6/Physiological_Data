import numpy as np
from biosignalsnotebooks import load
import os
try:
    from Physiological_Data.lib.tools import *
except ModuleNotFoundError:
    from Physiological_Data.lib.tools import *
from json import loads

END_BASELINE = 'End of Baseline'
BEGIN_BASELINE = 'Begin of Baseline'
FNIRS = 'hSpO2'
EEG = 'EEG'
ACC = 'XYZ'
RESP = 'RESPIRATION'
ECG = 'ECG'
EDA = 'EDA'


class Devices:
    def __init__(self, path):
        self.path = path
        self.data, self.header = self.read_bio_data(self.path)
        self.macs = list(self.header.keys())
        self.fs = self.header[self.macs[0]]['sampling rate']

    def read_bio_data(self, path):
        signals = []
        header = ''
        for file in os.listdir(path):
            if 'modified.txt' in file and 'converted_modified.txt' not in file:
                print(os.path.join(path, file))
                file_path = os.path.join(path, file)
                header = self.readHeader(file_path)
                data = np.loadtxt(file_path)
                signals.append(data)
        if np.shape(signals)[0] > 1:
            return np.vstack(signals), header
        else:
            return signals[0], header

    @staticmethod
    def readHeader(path):
        with open(path, 'r') as f:
            i = 0
            for line in f.readlines():
                if i == 1:
                    return loads(line[2:])
                i += 1

    @staticmethod
    def join_signals(signals, header):
        new_signal = {}
        for signal in signals:
            for mac in signal.keys():
                sensors = header[mac]['sensor']
                for i, chn in enumerate(signal[mac].keys()):
                    if mac in new_signal.keys():
                        if sensors[i] not in new_signal[mac].keys():
                            new_signal[mac][sensors[i]+chn[-1]] = signal[mac][chn]
                        else:
                            new_signal[mac][sensors[i]+chn[-1]] = np.concatenate([new_signal[mac][sensors[i]], signal[mac][chn]])
                    else:
                        new_signal[mac] = {}
                        new_signal[mac][sensors[i]+chn[-1]] = signal[mac][chn]
        return new_signal

    def segmentSignalsIndex(self, signal, timestamps, labels, binary=False):
        """
        Segment arrays according to PsychoPy indexes
        """
        signalArray = np.array(signal)
        indexes = self.getSegmentationIndexes(signalArray[:, 0], timestamps)
        segmentedSignal = []
        segmentLabels = []
        for i in range(1, len(indexes[1:])):
            # This interval corresponds to reading instructions
            if not (labels[i-1] == END_BASELINE and labels[i] == BEGIN_BASELINE):
                segmentLabels.append(labels[i-1])
                segmentedSignal.append(signalArray[indexes[i-1]:indexes[i]])

        if binary:
            return segmentedSignal, self.convertBinaryLabels(segmentLabels)
        else:
            return segmentedSignal, segmentLabels

    def segmentSignalsWindowing(self, signal, timestamps, labels, timeWindow=1, overlap=0, binary=False, normalize=False, returnTime=False):
        signalArray = np.array(signal)
        if normalize:
            signalArray = self.removeNonPsycho(signal, timestamps)
        signalArray = self.normalizeSensors(signalArray, True)
        segments = windowing(signalArray, sampling_rate=self.fs, time_window=timeWindow, overlap=overlap)
        print(f"Segments Shape: {segments.shape}")
        segments, segmentLabels = self.makeLabels(segments, labels, timestamps)

        if binary:
            segmentLabels = self.convertBinaryLabels(segmentLabels)
        if returnTime:
            return segments[:, :, 1:], segmentLabels, segments[:, :, 0]
        return segments[:, :, 1:], segmentLabels

    def makeLabels_(self, segments, labels, timestamps, timeGap=1):
        '''
        Make labels based on the timestamps and delete the segments that have more than one label or which label is
        'End of Baseline'.
        :return:
        '''
        segmentLabels = []
        aux = 0
        total = len(segments)
        for i in range(len(segments)):
            print(f"{(i/total)*100:.1f}%", end='\r')
            oldLabel = None
            isMiddle = False
            index = i - aux
            for j in range(0, len(segments[index]), timeGap*self.fs):
                print("HERE")
                # Check the label for each timestamp
                if j >= len(segments[index]):
                    j = -1
                label = labels[closestIndex(timestamps, segments[index, j, 0])]
                if (label != oldLabel and oldLabel is not None) or (label == END_BASELINE):
                    segments = np.delete(segments, index, axis=0)
                    isMiddle = True
                    aux += 1
                    break

                if not isMiddle and (j == len(segments[index]) - 1 or j == -1):
                    print(f"{label}", end='\r')
                    segmentLabels.append(label)
                oldLabel = label

        return segments, segmentLabels

    def makeLabels(self, segments, labels, timestamps):
        '''
        Make labels based on the timestamps and delete the segments that have more than one label or which label is
        'End of Baseline'.
        :return:
        '''
        segmentLabels, deleteIndexes = [], []
        aux = 0
        total = len(segments)
        for i in range(len(segments)):
            print(f"{(i/total)*100:.1f}%", end='\r')
            # Check the label for each timestamp
            currentLabel = labels[closestIndex(timestamps, segments[i, 0, 0])] # first label
            lastLabel = labels[closestIndex(timestamps, segments[i, -1, 0])] # last label
            if (currentLabel != lastLabel) or (currentLabel == END_BASELINE):
                deleteIndexes.append(i)
            else:
                # print(f"{currentLabel}", end='\r')
                segmentLabels.append(currentLabel)

        return np.delete(segments, deleteIndexes, axis=0), segmentLabels

    def convertBinaryLabels(self, labels):
        new_labels = []
        for label in labels:
            if label != END_BASELINE:
                if 'baseline' in label.lower():
                    new_labels.append('baseline')
                else:
                    new_labels.append('task')
        return new_labels

    def getSensorsData(self, sensors=["EEG", "ECG", "RESPIRATION"]):
        indexes = [np.array([0])]
        sensorsOut = ['nSeq']
        for i, mac in enumerate(self.header.keys()):
            for sensor in sensors:
                if i == 0:
                    aux = 2  # Timestamps and Digital
                    nSensors = len(self.header[mac]['sensor'])  # get number of sensors to sum in the else statement
                else:
                    aux = 4 + nSensors  # Timestamps and Digital from both devices
                index = strIndex(self.header[mac]['sensor'], sensor) + aux
                if len(index) != 0:
                    indexes.append(index)
                    sensorsOut.append(sensor)
        print(np.concatenate(indexes))
        return {'data': self.data[:, np.concatenate(indexes)], 'sensors': sensorsOut}

    def getSegmentationIndexes(self, signal, timestamps):
        signalArray = self.removeNonPsycho(signal, timestamps)
        return np.array([closestIndex(signalArray, i) for i in timestamps])

    def removeNonPsycho(self, signal, timestamps):
        signalArray = np.array(signal)
        return signalArray[closestIndex(signalArray[:, 0], timestamps[0]):closestIndex(signalArray[:, 0], timestamps[-1])]

    def normalizeSensors(self, data, firstTime=True):
        if firstTime:
            data[:, 1:] = data[:, 1:] - np.mean(data[:, 1:], axis=0)
            data[:, 1:] = data[:, 1:] / np.std(data[:, 1:], axis=0)
            return data
        else:
            data = data - np.mean(data, axis=0)
            return data / np.std(data, axis=0)


class Sensor:
    def __init__(self, data, fs, resolution):
        self.data = data
        self.fs = fs
        self.resolution = resolution

    def convertPhys(self):
        pass

    @staticmethod
    def statistical_Features(signal):

        maximum = max(signal)
        minimum = min(signal)
        average = np.mean(signal)

        statistical_features = {"AVG": average, "Maximum": maximum, "Minimum": minimum}

        return statistical_features


if __name__ == '__main__':
    device = Devices(r'..\..\acquisitions\Acquisitions\03_11_2020')
    print(device.getSensorsData())
