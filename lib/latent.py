try:
    from acquisition import *
except ModuleNotFoundError:
    from Physiological_Data.lib.acquisition import *
from os import listdir
from os.path import join
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pylab as plt

AUDIO = 'audio.wav'
KEYBOARD = 'keyboard.csv'
MOUSE = 'mouse.csv'
SCREENSHOT = 'screenshot.csv'
SNAPSHOT = 'snapshot.csv'
files = [AUDIO, KEYBOARD, MOUSE, SCREENSHOT, SNAPSHOT]


class Latent:
    def __init__(self, pathFolder):
        self.path = self.getPath(pathFolder)
        self.data = {}

    def getPath(self, pathFolder):
        for folder in listdir(pathFolder):
            if 'ip' in folder:
                return join(pathFolder, folder)

    def readLatentFile(self, file):
        end = file[-4:]
        if '.csv' == end:
            data = read_csv(join(self.path, file))
            return data
        elif '.wav' == end:
            sampling_ratem, data = wavfile.read(join(self.path, AUDIO))
            return data, sampling_ratem
        return []

    def readData(self):
        for file in files:
            if AUDIO != file:
                self.data[file.split('.')[0]] = self.readLatentFile(file)
            else:
                self.data[file.split('.')[0]], self.audioFS = self.readLatentFile(file)

    @staticmethod
    def makeSpectrogram(signal, fs):
        f, t, Sxx = spectrogram(signal, fs)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        return f, t, Sxx

    def syncInterfaces(self):
        pass


if __name__ == '__main__':
    latent = Latent(r"D:\Google Drive\Faculdade\Doutoramento\Acquisitions\new_biosignalsnotebooks\acquisitions\Acquisitions\03_11_2020")
    latent.readData()
    # data, fs = latent.readLatentFile(AUDIO)
    # latent.makeSpectrogram(data[::1000, 0], fs/1000)

