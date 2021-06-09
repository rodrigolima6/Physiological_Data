import biosignalsnotebooks as bsnb
try:
    from acquisition import *
except ModuleNotFoundError:
    from lib.acquisition import *
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

class ECG(Sensor):

    def convertECG(self):

        """
        :param signal: Raw ECG signal
        :param resolution: channel resolution from device
        :return: ECG signal in milivolts and volts
        """

        VCC=3000
        gain = 1000
        signal_volts = ((np.array(self.data)/2**self.resolution)-1/2)*VCC/gain
        signal_mv=signal_volts*1000

        return signal_mv,signal_volts


    def filterECG(self):

        """
        :param signal: ECG signal
        :param fs: sampling frequency of the ECG signal
        :return: filtered_signal - Band Pass filtered ECG signal
        """

        """Band pass filter between 5 and 15 Hz"""
        filtered_signal = bsnb.detect._ecg_band_pass_filter(self.data,self.fs)

        return filtered_signal

    @staticmethod
    def differentiateECG(signal):

        """
        :param signal: Filtered ECG signal
        :return: Derivative of the filtered ECG signal
        """

        differentiated_signal = np.diff(signal)

        return differentiated_signal

    @staticmethod
    def squaredECG(signal):

        """
        :param signal: Derivative ECG signal
        :return: Squared signal of the ECG derivative
        """

        squared_signal = signal**2

        return squared_signal

    @staticmethod
    def integrateECG(signal,fs,window=0.080):

        """
        :param signal: Squared ECG signal
        :param fs: sampling frequency of the ECG signal
        :param window: window of integration (80 ms)
        :return: Integrated ECG signal
        """

        nbr_sampls_int_wind = int(window * fs)
        integrated_signal = np.zeros_like(signal)
        cumulative_sum = signal.cumsum()
        integrated_signal[nbr_sampls_int_wind:] = (cumulative_sum[nbr_sampls_int_wind:] -
                                                   cumulative_sum[:-nbr_sampls_int_wind]) / nbr_sampls_int_wind
        integrated_signal[:nbr_sampls_int_wind] = cumulative_sum[:nbr_sampls_int_wind] / np.arange(1,
                                                                                                nbr_sampls_int_wind + 1)

        return integrated_signal

    def detectRPeaks(self):

        """
        :param signal: Non-filtered ECG signal
        :param fs: sampling frequency of the ECG signal
        :return: time_peaks - time instant for each R-peak detected
                 amp_peaks - amplitude of each R-peak detected
        """

        time_peaks,amp_peaks = bsnb.detect_r_peaks(self.data,self.fs,time_units=True,plot_result=False)

        return time_peaks,amp_peaks

class HRV(Sensor):

    def RR_interval(self):

        """
        :param signal: ECG signal
        :param fs: sampling frequency of ECG signal
        :return: rr_interval - RR interval series (diff between R-peaks)
                 rr_interval_time - RR interval series time axis.
        """

        rr_interval,rr_interval_time = bsnb.tachogram(self.data,self.fs,signal=True,out_seconds=True)

        return rr_interval,rr_interval_time

    @staticmethod
    def remove_EctopyBeats(rr_interval,rr_interval_time):

        """
        :param rr_interval: RR interval series
        :param rr_interval_time: RR interval series time axis
        :return: rr_interval_NN - RR interval series with no ectopic beats
                 rr_interval_time_NN - RR interval series time axis with no ectopic beats
        """

        rr_interval_NN,rr_interval_time_NN = bsnb.remove_ectopy(rr_interval,rr_interval_time)

        rr_interval_NN = np.array(rr_interval_NN)
        rr_interval_time_NN = np.array(rr_interval_time_NN)

        return rr_interval_NN,rr_interval_time_NN

    @staticmethod
    def HeartRate(rr_interval_NN):
        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: array of Heart Rate in beats per minute (Bpm) along time.
        """
        heart_rate=(60.0/rr_interval_NN)

        return heart_rate

    @staticmethod
    def time_domain_Features(rr_interval_NN):

        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: dict with time-domain features of HRV
        """

        """Standard deviation of RR interval series with no ectopic beats"""
        SDNN=round(np.std(rr_interval_NN)*1000,4)

        """Root Mean Square of the Standard deviation"""
        RMSSD=round((np.sqrt(np.sum(rr_interval_NN)**2)/(len(rr_interval_NN)-1))*1000,4)

        """Number and percentage of RR interval longer than 50 ms"""
        NN50=sum(1 for i in rr_interval_NN if i > 0.05)
        pNN50=round((float(NN50)/len(rr_interval_NN)*100),4)

        """Number and percentage of RR interval longer than 20 ms"""
        NN20=sum(1 for i in rr_interval_NN if i > 0.02)
        pNN20=round((float(NN20)/len(rr_interval_NN)*100),4)

        time_domain_features={"SDNN":SDNN,"RMSSD":RMSSD,"NN50":NN50,"pNN50":pNN50,"NN20":NN20,"pNN20":pNN20}

        return time_domain_features

    @staticmethod
    def poincare_features(rr_interval_NN):

        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: dict with Poincare features - non-linear features
        """

        """Standard Deviation of RR interval series"""
        STD = round(float(np.std(rr_interval_NN)), 4)

        """Standard Deviation of the successive differences of RR interval series"""
        SDSD=round(float(np.std(np.diff(rr_interval_NN))),4)

        """Length of the longitudinal line in Poincaré plot"""
        SD2 = round(np.sqrt(2 * STD ** 2 - 0.5 * SDSD ** 2), 4)

        """Length of the transverse line in Poincaré plot"""
        SD1 = round(np.sqrt(0.5 * SDSD ** 2), 4)

        "SD2/SD1"
        SD_ratio = round(SD2 / SD1, 4)

        poincaré_features={"SD1":SD1,"SD2":SD2,"SD2/SD1":SD_ratio}

        return poincaré_features

    @staticmethod
    def evenlySpaced_RR(rr_interval_NN,time,new_freq):

        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :param time: time axis
        :param new_freq: Frequency to which the RR interval series will be downsampled
        :return: Interpolated and Downsampled RR interval series
        """

        """ Time axis downsampled to new frequency"""
        downsampled_time = np.linspace(0, int(time[-1]), int(time[-1] * new_freq))

        """Time array with RR interval series evenly time-spaced"""
        evenlySpaced_time = np.linspace(0, int(time[-1]), int(len(rr_interval_NN)))

        """
        (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and degree of spline
        Model of the RR interval series
        """
        tck = sc.interpolate.splrep(evenlySpaced_time, rr_interval_NN)

        """Interpolated RR interval series using the model obtained"""
        interpolatedRR = sc.interpolate.splev(downsampled_time, tck)

        return interpolatedRR

    @staticmethod
    def frequency_analysis(rr_interval_time_NN,rr_interval_NN,window='hanning',interpolation_rate = 4):


        init_time = int(rr_interval_time_NN[0])
        fin_time = int(rr_interval_time_NN[-1])
        tck = sc.interpolate.splrep(rr_interval_time_NN, rr_interval_NN)

        nn_time_even = np.linspace(init_time, fin_time, (fin_time - init_time) * interpolation_rate)
        nn_tachogram_even = sc.interpolate.splev(nn_time_even, tck)

        freq_axis, power_axis = sc.signal.welch(nn_tachogram_even, interpolation_rate,
                                                window=sc.signal.get_window(window,
                                                                            min(len(nn_tachogram_even),
                                                                                1000)),
                                                nperseg=min(len(nn_tachogram_even), 1000))

        freqs = [round(val, 3) for val in freq_axis if val < 0.5]
        power = [round(val, 4) for val, freq in zip(power_axis, freq_axis) if freq < 0.5]

        return freqs, power

class PPG(Sensor):

    def filterPPG(self,lowpassFreq=5,highpassFreq=0.1,lowpassOrder=2,highpassOrder=2):

        filteredPPG = bsnb.highpass(bsnb.lowpass(self.data,lowpassFreq,lowpassOrder,self.fs),highpassFreq,highpassOrder,self.fs)

        return filteredPPG

    @staticmethod
    def findPeaksPPG(signal):
        peaks = []
        peaks_location = []

        for i in range(len(signal) - 1):
            if (signal[i] > signal[i - 1] and signal[i] > signal[i + 1]):
                peaks.append(signal[i])
                peaks_location.append(i)

        peaksAmp = np.array(peaks)
        peaksIndex = np.array(peaks_location)

        return peaksAmp,peaksIndex

    @staticmethod
    def findValleysPPG(signal):

        valleys = []
        valleys_location = []

        for i in range(len(signal) - 1):
            if (signal[i] < signal[i - 1] and signal[i] < signal[i + 1]):
                valleys.append(signal[i])
                valleys_location.append(i)

        valleysAmp = np.array(valleys)
        valleysIndex = np.array(valleys_location)

        return valleysAmp,valleysIndex

    @staticmethod
    def remove1stPeak(peaksAmp,peaksIndex,valleysIndex):

        if (peaksIndex[0] < valleysIndex[0]):
            peaksAmp.remove(peaksAmp[0])
            peaksIndex.remove(peaksIndex[0])

        return peaksAmp, peaksIndex

    @staticmethod
    def peaks_valleyDiff(peaksAmp,valleysAmp):

        vpd=[]

        for i in range(len(peaksAmp)):
            vpd.append(peaksAmp[i] - valleysAmp[i])

        valley_peak_diff = np.array(vpd)

        return valley_peak_diff

    def detectPeaksPPG(self,lowpassFreq=5,highpassFreq=0.1):

        filteredPPG = PPG.filterPPG(self,lowpassFreq,highpassFreq)
        peaksAmp,peaksIndex = PPG.findPeaksPPG(filteredPPG)
        valleysAmp,valleysIndex = PPG.findValleysPPG(filteredPPG)
        peaksAmp, peaksIndex = PPG.remove1stPeak(peaksAmp, peaksIndex,valleysIndex)
        valley_peak_diff = PPG.peaks_valleyDiff(peaksAmp,valleysAmp)

        numberPeaks = len(valley_peak_diff)
        new_numberPeaks = -1
        flag=True

        while (flag):
            peak_location = []
            peak_amplitude=[]
            for i in range(2, len(peaksAmp) - 2):
                if (valley_peak_diff[i] > (0.7 * (valley_peak_diff[i - 2] + valley_peak_diff[i - 1] + valley_peak_diff[i] + valley_peak_diff[i + 1] + valley_peak_diff[i + 2]) / 5)):
                    peak_location.append(peaksIndex[i])
                    peak_amplitude.append(peaksAmp[i])

            new_peaks_number = len(peak_location)

            if (numberPeaks == new_peaks_number):
                flag = False
            else:
                numberPeaks = new_peaks_number

            peaksIndexes = np.array(peak_location)
            peaksAmplitude = np.array(peak_amplitude)

            return peaksAmplitude,peaksIndexes


if __name__ == '__main__':
    ecg = ECG(np.sin(2*np.pi/1000), 1000, 16)
