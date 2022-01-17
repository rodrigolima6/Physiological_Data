import biosignalsnotebooks as bsnb
import pandas as pd
import math
try:
    from Physiological_Data.lib.acquisition import *
except (ImportError, ModuleNotFoundError):
    from Physiological_Data.lib.acquisition import *
import scipy as sc
import json
import neurokit2 as nk
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from scipy.signal import welch
from scipy import integrate
from biosppy.signals.eda import *
import novainstrumentation as ni
from mes2hb.mes2hb import Mes2Hb
try:
    from Physiological_Data.lib.tools import *
    from Physiological_Data.lib.respRT import peak_detector_Resp
    from Physiological_Data.lib.signal_processing import integration_DO
except (ImportError, ModuleNotFoundError):
    from Physiological_Data.lib.tools import *
    from Physiological_Data.lib.respRT import peak_detector_Resp
    from Physiological_Data.lib.signal_processing import integration_DO

class TEMP(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data
        self.fs = fs
        self.resolution = resolution

class EDA(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data
        self.fs = fs
        self.resolution = resolution

    def filterEDA(self):

        lowfrequency=1
        order=2

        data_filtered = bsnb.lowpass(self.data,lowfrequency,order,self.fs,use_filtfilt=True)

        return data_filtered

    @staticmethod
    def componentsEDA(data_filtered):

        eda_components = nk.eda_phasic(data_filtered)
        eda_phasic = eda_components["EDA_Phasic"].values
        eda_tonic = eda_components["EDA_Tonic"].values

        return eda_phasic,eda_tonic


    def featuresSCR(self,eda_phasic):
        info,signals = nk.eda_peaks(eda_phasic,self.fs,method="neurokit")

        SCR_Amplitude = signals["SCR_Amplitude"]
        SCR_RiseTime = signals["SCR_RiseTime"]
        SCR_RecoveryTime = signals["SCR_RecoveryTime"]

        return SCR_Amplitude,SCR_RiseTime,SCR_RecoveryTime

    @staticmethod
    def frequencyAnalysis(data_filtered):

        downsampled1 = sc.signal.decimate(data_filtered, q=10,n=8)
        downsampled2 = sc.signal.decimate(downsampled1, q=10,n=8)
        downsampled3 = sc.signal.decimate(downsampled2, q=10, n=8)

        signal_filtered = bsnb.highpass(downsampled3, 0.01, order=8)

        freqs,power = sc.signal.welch(signal_filtered, nperseg=128, window='blackman')

        return freqs,power

    @staticmethod
    def frequencyFeatures(freq,power):
        """
                    Indexes of Frequencies of each component
                    """
        vlf_indexes = np.where((freq[:] >= 0.0033) & (freq[:] < 0.04))[0]
        lf_indexes = np.where((freq[:] >= 0.04) & (freq[:] < 0.15))[0]
        hf_indexes = np.where((freq[:] >= 0.15) & (freq[:] < 0.4))[0]
        total_power_indexes = np.where((freq[:] >= 0.0033) & (freq[:] <= 0.4))[0]
        """
        Power of each frequency component in the desired range of frequencies
        """

        vlf = round(sc.integrate.trapz(power[vlf_indexes], freq[vlf_indexes]) * 1000000, 4)
        lf = round(sc.integrate.trapz(power[lf_indexes], freq[lf_indexes]) * 1000000, 4)
        hf = round(sc.integrate.trapz(power[hf_indexes], freq[hf_indexes]) * 1000000, 4)
        total_power = round(sc.integrate.trapz(power[total_power_indexes], freq[total_power_indexes]) * 1000000, 4)

        """
        Frequency components in normalized units (n.u)
        Balance - LF(n.u)/HF(n.u)
        """

        lf_norm = round(lf / (total_power - vlf) * 100, 2)
        hf_norm = round(hf / (total_power - vlf) * 100, 2)
        ratio = round(lf_norm / hf_norm, 2)

        frequency_features = {"VLF Power": [vlf], "LF Power": [lf], "HF Power": [hf], "Total Power": [total_power],
                              "LF (nu)": [lf_norm], "HF (nu)": [hf_norm], "LF/HF": [ratio]}

        return frequency_features

    def getFeatures(self):
        eda_filtered = self.filterEDA()
        eda_phasic, eda_tonic = self.componentsEDA(eda_filtered)
        SCR_Amplitude, SCR_RiseTime, SCR_RecoveryTime = self.featuresSCR(eda_phasic)
        freqs, power = self.frequencyAnalysis(eda_filtered)
        frequency_features = self.frequencyFeatures(freqs,power)

        eda_phasic_dict = self.statistical_Features(eda_phasic)
        eda_tonic_dict = self.statistical_Features(eda_tonic)
        SCR_Amplitude_dict = self.statistical_Features(SCR_Amplitude)
        SCR_RiseTime_dict = self.statistical_Features(SCR_RiseTime)
        SCR_RecoveryTime_dict = self.statistical_Features(SCR_RecoveryTime)

        return eda_phasic_dict,eda_tonic_dict,SCR_Amplitude_dict,SCR_RiseTime_dict,SCR_RecoveryTime_dict,frequency_features


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
        signal_mv = signal_volts*1000

        return signal_mv


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
    def _differentiateECG(signal):

        """
        :param signal: Filtered ECG signal
        :return: Derivative of the filtered ECG signal
        """

        differentiated_signal = np.diff(signal)

        return differentiated_signal

    @staticmethod
    def _squaredECG(signal):

        """
        :param signal: Derivative ECG signal
        :return: Squared signal of the ECG derivative
        """

        squared_signal = signal**2

        return squared_signal

    @staticmethod
    def _integrateECG(signal, fs, window=0.080):

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

    def _detectRPeaks(self):

        """
        :param signal: Non-filtered ECG signal
        :param fs: sampling frequency of the ECG signal
        :return: time_peaks - time instant for each R-peak detected
                 amp_peaks - amplitude of each R-peak detected
        """
        # peaks, valleys = peak_detector(self.data, self.fs)
        peaks = ni.panthomkins.panthomkins(self.data,self.fs)

        # return peaks[1]
        return peaks
    
    def calculateHR(self, peaks):
        hr = []
        time = []
        for i in range(1, len(peaks)):
            value = (peaks[i] - peaks[i - 1]) * 60 / self.fs
            hr.append(value)
            time.append(peaks[i] - peaks[i - 1] / self.fs)
        return hr, time
    
    def processECG(self):
        peaks = self._detectRPeaks()

        return peaks
        # hr, time = self.calculateHR(peaks)
        # return hr, time


class HRV(Sensor):

    def RR_interval(self):

        """
        :param signal: ECG signal
        :param fs: sampling frequency of ECG signal
        :return: rr_interval - RR interval series (diff between R-peaks)
                 rr_interval_time - RR interval series time axis.
        """

        rr_interval, rr_interval_time = bsnb.tachogram(self.data, self.fs, signal=True, out_seconds=True)

        return rr_interval, rr_interval_time

    @staticmethod
    def remove_EctopyBeats(rr_interval, rr_interval_time):

        """
        :param rr_interval: RR interval series
        :param rr_interval_time: RR interval series time axis
        :return: rr_interval_NN - RR interval series with no ectopic beats
                 rr_interval_time_NN - RR interval series time axis with no ectopic beats
        """

        rr_interval_NN, rr_interval_time_NN = bsnb.remove_ectopy(rr_interval, rr_interval_time)

        rr_interval_NN = np.array(rr_interval_NN)
        rr_interval_time_NN = np.array(rr_interval_time_NN)

        return rr_interval_NN, rr_interval_time_NN

    def heart_rate(self,rr_interval_NN):
        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: array of Heart Rate in beats per minute (Bpm) along time.
        """
        heart_rate=(60.0/rr_interval_NN)

        return heart_rate

    def heartRate_features(self,heart_rate):

        statistical_features = self.statistical_Features(heart_rate)

        hr = {"Avg HR":[statistical_features["AVG"]],"Min HR":[statistical_features["Minimum"]],"Max HR":[statistical_features["Maximum"]],"SD":[statistical_features["STD"]]}

        return hr


    def timeDomainFeatures(self,rr_interval_NN):

        statistical_features = self.statistical_Features(rr_interval_NN)

        """
        :param rr_interval_NN: RR interval series with no ectopic beats
        :return: dict with time-domain features of HRV
        """

        rr_interval_diff = np.diff(rr_interval_NN)
        rr_interval_abs = np.abs(rr_interval_diff)

        """Standard deviation of RR interval series with no ectopic beats"""
        SDNN=round(np.std(rr_interval_NN)*1000,4)

        """Root Mean Square of the Standard deviation"""
        RMSSD=round(np.sqrt(np.sum((rr_interval_diff)**2)/(len(rr_interval_NN)-1))*1000,4)

        """Number and percentage of RR interval longer than 50 ms"""
        NN50=sum(1 for i in rr_interval_abs if i > 0.05)
        pNN50=round((float(NN50)/len(rr_interval_NN)*100),4)

        """Number and percentage of RR interval longer than 20 ms"""
        NN20=sum(1 for i in rr_interval_abs if i > 0.02)
        pNN20=round((float(NN20)/len(rr_interval_NN)*100),4)

        time_domain_features={"AVG RR":[statistical_features["AVG"]],"Minimum RR":[statistical_features["Minimum"]],"Maximum RR": [statistical_features["Maximum"]],"SDNN":[SDNN],"RMSSD":[RMSSD],"NN50":[NN50],"pNN50":[pNN50],"NN20":[NN20],"pNN20":[pNN20]}

        return time_domain_features

    @staticmethod
    def poincareFeatures(rr_interval_NN):

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

        poincaré_features={"SD1":[SD1*1000],"SD2":[SD2*1000],"SD2/SD1":[SD_ratio]}

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
    def frequencyAnalysis(rr_interval_time_NN, rr_interval_NN, window='hanning', interpolation_rate=4):


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

        freqs = np.array([round(val, 3) for val in freq_axis if val < 0.5])
        power = np.array([round(val, 4) for val, freq in zip(power_axis, freq_axis) if freq < 0.5])


        return freqs, power

    @staticmethod
    def frequencyFeatures(freq,power):
        """
            Indexes of Frequencies of each component
            """
        vlf_indexes = np.where((freq[:] >= 0.0033) & (freq[:] < 0.04))[0]
        lf_indexes = np.where((freq[:] >= 0.04) & (freq[:] < 0.15))[0]
        hf_indexes = np.where((freq[:] >= 0.15) & (freq[:] < 0.4))[0]
        total_power_indexes = np.where((freq[:] >= 0.0033) & (freq[:] <= 0.4))[0]
        """
        Power of each frequency component in the desired range of frequencies
        """

        vlf = round(sc.integrate.trapz(power[vlf_indexes], freq[vlf_indexes])*1000000, 4)
        lf = round(sc.integrate.trapz(power[lf_indexes], freq[lf_indexes])*1000000, 4)
        hf = round(sc.integrate.trapz(power[hf_indexes], freq[hf_indexes])*1000000, 4)
        total_power = round(sc.integrate.trapz(power[total_power_indexes], freq[total_power_indexes])*1000000, 4)

        """
        Frequency components in normalized units (n.u)
        Balance - LF(n.u)/HF(n.u)
        """

        lf_norm = round(lf / (total_power - vlf) * 100, 2)
        hf_norm = round(hf / (total_power - vlf) * 100, 2)
        ratio = round(lf_norm / hf_norm, 2)

        frequency_features={"HRV VLF Power":[vlf],"HRV LF Power":[lf],"HRV HF Power":[hf],"HRV Total Power":[total_power],"HRV LF (nu)":[lf_norm],"HRV HF (nu)":[hf_norm],"HRV LF/HF":[ratio]}

        return frequency_features


    def getFeatures(self):
        rr_interval, rr_interval_time = self.RR_interval()
        rr_interval_NN, rr_interval_time_NN = self.remove_EctopyBeats(rr_interval, rr_interval_time)
        heart_rate = self.heart_rate(rr_interval_NN)
        heart_rate_features = self.heartRate_features(heart_rate)
        freq, power = self.frequencyAnalysis(rr_interval_time_NN, rr_interval_NN)
        time_features = self.timeDomainFeatures(rr_interval_NN)
        poincare_features = self.poincareFeatures(rr_interval_NN)
        frequency_features = self.frequencyFeatures(freq,power)

        return heart_rate_features,time_features, poincare_features,frequency_features

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


class fNIRS(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)
        # self.red = self.convertPhys(data[:, 0], resolution)
        # self.infrared = self.convertPhys(data[:, 1], resolution)

        self.red = self.filterData(data[:,0], fs)
        self.infrared = self.filterData(data[:,1], fs)

        # print(self.red, self.infrared)
    
    @staticmethod
    def convertPhys(data, resolution):
        return (0.15 * data) / (2**resolution)

    def convertConcentration(self):
        converter = Mes2Hb()
        self.hbo, self.hb, self.hbt = converter.convert([self.red.copy(), self.infrared.copy()], wavelength=[660, 860])

    def detectPeaks(self):
        pass

    @staticmethod
    def filterData(data, fs):
        return bsnb.bandpass(data, 0.05, 0.4, fs=fs, use_filtfilt=True)

    @staticmethod
    def root_mean_square(signal):
        """Signal should be a segment"""
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def slope_regression(signal):
        """Signal should be a segment"""
        time = np.arange(0, len(signal)).reshape(-1, 1)
        model = LinearRegression()
        model = model.fit(time, signal.reshape(-1,1))
        return model.coef_[0]

    @staticmethod
    def slope_naive(signal):
        """Signal should be a segment"""
        return signal[-1] - signal[0]

    def processfNIRS(self):
        self.convertConcentration()

    def getFeatures(self):

        hb_dict = self.statistical_Features(self.hb)
        hbo_dict = self.statistical_Features(self.hbo)
        hbt_dict = self.statistical_Features(self.hbt)

        fNIRS_dict={"AVG Hb":[hb_dict["AVG"]],"Minimum Hb":[hb_dict["Minimum"]],"Maximum Hb":[hb_dict["Maximum"]],"STD Hb":[hb_dict["STD"]],
                    "AVG Hbo":[hbo_dict["AVG"]],"Minimum Hbo":[hbo_dict["Minimum"]],"Maximum Hbo":[hbo_dict["Maximum"]],"STD Hbo":[hbo_dict["STD"]],
                    "AVG Hbt":[hbt_dict["AVG"]],"Minimum Hbt":[hbt_dict["Minimum"]],"Maximum Hbt":[hbt_dict["Maximum"]],"STD Hbt":[hbt_dict["STD"]]}

        return fNIRS_dict


class EEG(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data # converted and filtered EEG data
        self.fs = fs
        self.resolution = resolution
        self.bands = {'alpha': [8, 14], 'betha': [14, 30], 'gamma': [30, 49], 'theta': [4, 8], 'delta': [.5, 4]}

    @staticmethod
    def ICA(data):
        ica = FastICA()

        EEG_ICA = ica.fit_transform(np.array(data).reshape(-1,1))

        return EEG_ICA

    @staticmethod
    def filterData(data,fs):
        EEG_shift = data[:,0]-np.mean(data)
        EEG_filtered = bsnb.bandpass(EEG_shift,1,40,order=8,fs=fs,use_filtfilt=True)

        return EEG_filtered

    @staticmethod
    def frequencyAnalysis(data,fs):
        freqs,power = welch(data,fs,nperseg=fs/2)

        alpha_indexes = np.where((freqs[:] >= 8) & (freqs[:] < 14))[0]
        betha_indexes = np.where((freqs[:] >= 14) & (freqs[:] < 30))[0]
        gamma_indexes = np.where((freqs[:] >= 30) & (freqs[:] < 49))[0]
        theta_indexes = np.where((freqs[:] >= 4) & (freqs[:] < 8))[0]
        delta_indexes = np.where((freqs[:] >= 0.5) & (freqs[:] < 4))[0]

        alpha = sc.integrate.trapz(power[alpha_indexes],x=freqs[alpha_indexes])
        betha = sc.integrate.trapz(power[betha_indexes], x=freqs[betha_indexes])
        gamma = sc.integrate.trapz(power[gamma_indexes], x=freqs[gamma_indexes])
        theta = sc.integrate.trapz(power[theta_indexes], x=freqs[theta_indexes])
        delta = sc.integrate.trapz(power[delta_indexes], x=freqs[delta_indexes])

        bands_power = {"alpha":alpha,"betha":betha,"gamma":gamma,"theta":theta,"delta":delta}

        return bands_power
    # def frequencyAnalysis(self,data):
    #     freqs,power = welch(data,self.fs,nperseg=self.fs/2)
    #
    #     alpha_indexes = np.where((freqs[:] >= self.bands["alpha"][0]) & (freqs[:] < self.bands["alpha"][1]))[0]
    #     betha_indexes = np.where((freqs[:] >= self.bands["betha"][0]) & (freqs[:] < self.bands["betha"][1]))[0]
    #     gamma_indexes = np.where((freqs[:] >= self.bands["gamma"][0]) & (freqs[:] < self.bands["gamma"][1]))[0]
    #     theta_indexes = np.where((freqs[:] >= self.bands["theta"][0]) & (freqs[:] < self.bands["theta"][1]))[0]
    #     delta_indexes = np.where((freqs[:] >= self.bands["delta"][0]) & (freqs[:] < self.bands["delta"][1]))[0]
    #
    #     alpha = sc.integrate.trapz(power[alpha_indexes],x=freqs[alpha_indexes])
    #     betha = sc.integrate.trapz(power[betha_indexes], x=freqs[betha_indexes])
    #     gamma = sc.integrate.trapz(power[gamma_indexes], x=freqs[gamma_indexes])
    #     theta = sc.integrate.trapz(power[theta_indexes], x=freqs[theta_indexes])
    #     delta = sc.integrate.trapz(power[delta_indexes], x=freqs[delta_indexes])
    #
    #     bands_power = {"alpha":alpha,"betha":betha,"gamma":gamma,"theta":theta,"delta":delta}
    #
    #     return freqs,power,bands_power


    @staticmethod
    def extractBand(data: np.array, band: list, fs: int):
        f1, f2 = band
        win = fs/2
        freq, power = welch(data, fs, nperseg=win)
        idx_band = np.logical_and(freq >= f1, freq <= f2)  # Get the band of frequencies
        power_freq = np.trapz(power[idx_band],x=freq[idx_band])  # Calculate the power of the band of frequencies

        return power_freq

    def extractAllBands(self,data,bands):
        # print(self.bands["alpha"][0])
        band_powers = {}

        for key, item in bands.items():
            band_powers[key] = self.extractBand(data, item, self.fs)

        return band_powers


    def getFeatures(self,data):
        # EEG_ICA = self.ICA()
        # EEG_filtered = self.filterData(EEG_ICA)

        freqs,power,band_powers = self.frequencyAnalysis(data)

        return freqs,power,band_powers

    def getDominantFreq(self, data, fs):
        win = 4 * fs
        freq, power = welch(data, fs, nperseg=win)
        dominant_freq = freq[np.argmax(power)]
        return dominant_freq
    
    def getCombinationFreq(self, power_freqs: dict):
        combinations = {}
        for key, item in power_freqs.items():
            for other_key, other_item in power_freqs.items():
                if key != other_key:
                    combinations[f"{key}/{other_key}"] = item/other_item
        return combinations
    
    # def getFeatures(self):
    #     power_freqs = self.extractAllBands(self.bands)
    #     dominant_freq = self.getDominantFreq(self.data, self.fs)
    #     combinations = self.getCombinationFreq(power_freqs)
    #     features = np.concatenate([[dominant_freq], list(power_freqs.values()), list(combinations.values())])
    #     return features

class TEMP(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data
        self.fs = fs
        self.resolution = resolution

    def filterData(self,lowpassfreq=0.1):

        data_filtered = bsnb.lowpass(self.data,lowpassfreq,order=2,fs=self.fs,use_filtfilt=True)

        return data_filtered

    def getFeatures(self,data):
        temp = self.statistical_Features(data)

        temp_dict = {"AVG Temp":[temp["AVG"]],"Max Temp":[temp["Maximum"]],"Min Temp":[temp["Minimum"]],"STD Temp":[temp["STD"]]}

        temp_Dataframe = pd.DataFrame.from_dict(temp_dict,orient="columns")

        return temp_Dataframe

class ACC(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)

        self.data = data * 9.8  # Convert to m/s
        self.fs = fs
        self.resolution = resolution
    
    def getIntegration(self, data, fs):
        # t = bsnb.generate_time(data, fs)
        return integration_DO(data, fs)

    def getVel(self, acc, fs):
        return self.vel
    
    def getDesl(self, vel, fs):
        return self.desl
    
    
    def getFeatures(self):
        self.acc = self.data
        self.vel = self.getIntegration(self.acc, self.fs)
        self.desl = self.getIntegration(self.vel, self.fs)

        features = []
        for i in [self.acc, self.vel, self.desl]:
            for func in [np.mean, np.min, np.max, np.std]:
                features.append(func(i))

        return np.array(features)


class RESP(Sensor):
    def __init__(self, data, fs, resolution):
        super().__init__(data, fs, resolution)
        
        self.data = np.array(data).astype(float)
        self.fs = fs
        self.resolution = resolution

    def process_RESP(self):
        signals,info = nk.rsp_process(self.data,self.fs,method="khodadad2018")

        return signals,info

    @staticmethod
    def RESP_RRV(signals):
        info,peak_signals=nk.rsp_peaks(signals["RSP_Clean"])

        rrv_dataframe=nk.rsp_rrv(signals["RSP_Rate"],peaks=peak_signals["RSP_Troughs"])

        return rrv_dataframe

    def getFeatures(self,signals,rrv_dataframe):
        rsp_rate_dict = self.statistical_Features(signals["RSP_Rate"])
        # rsp_amp_dict = self.statistical_Features(signals["RSP_Amplitude"])

        # rrv_dataframe.insert(0, "STD_RSP_Amplitude", rsp_amp_dict["STD"], True)
        # rrv_dataframe.insert(0, "Maximum_RSP_Amplitude", rsp_amp_dict["Maximum"], True)
        # rrv_dataframe.insert(0, "Minimum_RSP_Amplitude", rsp_amp_dict["Minimum"], True)
        # rrv_dataframe.insert(0,"Mean_RSP_Amplitude",rsp_amp_dict["AVG"],True)

        rrv_dataframe.insert(0, "STD_RSP_Rate", rsp_rate_dict["STD"], True)
        rrv_dataframe.insert(0, "Maximum_RSP_Rate", rsp_rate_dict["Maximum"], True)
        rrv_dataframe.insert(0, "Minimum_RSP_Rate", rsp_rate_dict["Minimum"], True)
        rrv_dataframe.insert(0, "Mean_RSP_Rate", rsp_rate_dict["AVG"], True)

        return rrv_dataframe

    def maxPeaks(self, peaks):
        return max(peaks)

    def meanAmpPeaks(self, peaks):
        return np.mean(peaks)

    def stdAmpPeaks(self, peaks):
        return np.std(peaks)

    def rmsAmpPeaks(self, peaks):
        return np.sqrt(np.mean(np.power(peaks, 2))) / len(peaks)

    def energyValue(self, data):
        return np.mean(np.power(data, 2)) / len(data)

    def meanValue(self, data):
        return np.mean(data)

    def minValue(self, data):
        return min(data)

    def maxValue(self, data):
        return max(data)

    def stdValue(self, data):
        return np.std(data)

    def rmsValue(self, data):
        return self.rmsAmpPeaks(data)

    def areaValue(self, data):
        return integrate.cumtrapz(data)

    def respFreq(self, peaks, fs):
        return 1/self.respInterval(peaks, fs)

    def respInterval(self, peaks, fs):
        return np.mean(np.diff(peaks))*fs

    def statisticsLastPeaks(self, peaks, fs):
        if len(peaks) > 10:
            peaks = peaks[-10:]
        diff_peaks = np.diff(peaks) * fs
        return np.mean(diff_peaks), np.std(diff_peaks), min(diff_peaks), max(diff_peaks), self.rmsValue(diff_peaks)

    def zeroCrossing(self, data, fs):
        return bsnb.zero_crossing_rate(data) * len(data)/fs
    #
    # def getFeatures(self):
    #     peaks, peaksAmp = peak_detector_Resp(self.data, self.fs)
    #     peaks = peaks[1]
    #     features = []
    #     funcs = {'peaks': [self.maxPeaks, self.meanAmpPeaks, self.stdAmpPeaks, self.rmsAmpPeaks, self.respFreq, self.respInterval, self.statisticsLastPeaks],
    #             'data': [self.energyValue, self.meanValue,  self.minValue, self.maxValue, self.stdValue, self.rmsValue, self.areaValue, self.zeroCrossing]}
    #     # Extract features from peaks
    #     for key in funcs:
    #         aux = self.data
    #         if key == 'peaks':
    #             aux = peaks
    #         for func in funcs[key]:
    #             try:
    #                 value = func(aux)
    #             except TypeError as e:
    #                 value = func(aux, self.fs)
    #
    #             if type(value) == np.float64 or type(value) == int:
    #                 value = [value]
    #             features.append(value)
    #
    #     return np.concatenate(features)


if __name__ == '__main__':
    # ecg = ECG(np.sin(2*np.pi/1000), 1000, 16)
    device = Devices(r'..\..\acquisitions\Acquisitions\03_11_2020')
    data = device.getSensorsData([FNIRS])
    fnirs = fNIRS(data['data'][:, 1:3], device.fs, device.resolution)
    fnirs.processfNIRS()
