__version__ = '1.0.0'

import sys
import numpy as np
import math
from .absorption_coefficients import AbsorptionCoefficients

from time import time
from numba import njit


@njit
def _compute_log(pos_red, pos_infrared, red, infrared, mean_baseline_red, mean_baseline_infrared):
    new_red = np.zeros(len(red))
    new_ir = np.zeros(len(infrared))
    len_red = len(pos_red)
    len_infrared = len(pos_infrared)
    pos = max(len_red, len_infrared)
    for i in range(pos):
        if i < len_red:
            value = pos_red[i]
            new_red[i] = math.log(mean_baseline_red/red[value][0])
        if i < len_infrared:
            value = pos_infrared[i]
            new_ir[i] = math.log(mean_baseline_infrared/infrared[value][0])
    return new_red, new_ir


class Mes2Hb:
    def __init__(self):
        self.coefficients = AbsorptionCoefficients()

    def convert(self, mes_data, baseline = [0, 100], wavelength = [690, 830]):
        """
            Cnverts optical density (OD) to oxy, de-oxy an total
            HB concentrations.
            The arrays returned will have baseline measurements
            zeroed out making the resulting in fewer rows than
            mes_data.

            params:
                mes_data(np.ndarray): a Nx2 dimensional array with
                1st column containing red wavelength values and
                2nd column containing infra-red wavelength values.

                baseline(list): first and last indices of rows to
                be accounted for baseline correction

                wavelength(list): precise wavelengths of red and infra-red
                channels obtained from the sensor.
            returns:
                hbo, hb, hbt(np.ndarray): 3 (N-baseline[1]-baseline[0], 1) arrays
                containing oxy, de-oxy and total haemoglobin concentrations.
        """
        
        t = time()
        
        red_mes_data = np.reshape(
            mes_data[0], (mes_data[0].shape[0], 1)
            )
        ir_mes_data = np.reshape(
            mes_data[1], (mes_data[1].shape[0], 1)
            )

        mes_data_shape = ir_mes_data.shape
        
        # print("Time to reshape: ", time() - t, "; Shape: ", mes_data_shape)

        wlen_red = wavelength[0]
        wlen_ir = wavelength[1]
        
        t = time()

        oxy_red = self.coefficients.get_coefficient(
            wlen_red, "oxy"
            )
        oxy_ir = self.coefficients.get_coefficient(
            wlen_ir, "oxy"
            )
        dxy_red = self.coefficients.get_coefficient(
            wlen_red, "dxy"
            )
        dxy_ir = self.coefficients.get_coefficient(
            wlen_ir, "dxy"
            )
        
        # print("Time to get coefficients: ", time() - t, "Coefficients: ", oxy_red, oxy_ir, dxy_red, dxy_ir)
        t = time()

        mean_baseline_red = np.mean(red_mes_data[baseline[0]:baseline[1]])
        mean_baseline_ir = np.mean(ir_mes_data[baseline[0]:baseline[1]])
        
        # print("Time to compute mean: ", time() - t)
        t = time()
        
        
        ####################################################### REDUCE TIME #############################################################
        pos_red = np.where(
            red_mes_data*mean_baseline_red > 0
            )

        pos_ired = np.where(
            ir_mes_data*mean_baseline_ir > 0
            )
        # print("Time to compute baseline: ", time()-t)
        t = time()

        a_red, a_ir = _compute_log(pos_red[0], pos_ired[0], red_mes_data, ir_mes_data, mean_baseline_red, mean_baseline_ir)

        # print("Time to compute log: ", time()-t)
        t = time()
        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        hb = np.zeros(mes_data_shape)
        hbo = np.zeros(mes_data_shape)
        hbt = np.zeros(mes_data_shape)

        ####### Oxy Hb #######
        if ((oxy_red*dxy_ir - oxy_ir*dxy_red)!=0):
            hbo = (a_red*dxy_ir - a_ir*dxy_red)/(oxy_red*dxy_ir - oxy_ir*dxy_red)
        # print("Time to compute Oxy Hb: ", time()-t)
        t = time()

        ####### DeOxy Hb #######
        if ((dxy_red*oxy_ir - dxy_ir*oxy_red)!=0):
        	hb = (a_red*oxy_ir - a_ir*oxy_red)/(dxy_red*oxy_ir - dxy_ir*oxy_red)
        # print("Time to compute Deoxy Hb: ", time()-t)

        hbt = hbo + hb
        return hbo[baseline[1]:], hb[baseline[1]:], hbt[baseline[1]:]
