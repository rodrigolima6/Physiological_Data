# -*- coding: utf-8 -*-

from numpy import mean, cumsum
import matplotlib.mlab as mlab
from novainstrumentation.peaks import bigPeaks
from novainstrumentation.smooth import smooth
from novainstrumentation.tools import plotfft


   
def fundamental_frequency(s,FS):
    # TODO: review fundamental frequency to guarantee that f0 exists 
    # suggestion peak level should be bigger 
    # TODO: explain code
    """Compute fundamental frequency along the specified axes.

    Parameters
    ----------
    s: ndarray
        input from which fundamental frequency is computed.
    FS: int
        sampling frequency    
    Returns
    -------
    f0: int
       its integer multiple best explain the content of the signal spectrum.
    """
    
    s = s - mean(s)
    f, fs = plotfft(s, FS, doplot=False)
    
    #fs = smooth(fs, 50.0)
  
    fs = fs[1:len(fs) / 2]
    f = f[1:len(f) / 2]
    
    cond = mlab.find(f > 0.5)[0]
    
    bp = bigPeaks(fs[cond:], 0)
    
    if bp==[]:
        f0=0
    else:
        
        bp = bp + cond
        
        f0 = f[min(bp)]
    
    return f0 


def max_frequency (sig,FS):
    """Compute max frequency along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which max frequency is computed.
    FS: int
        sampling frequency    
    Returns
    -------
    f_max: int
       0.95 of max_frequency using cumsum.
    """
    
    f, fs = plotfft(sig, FS, doplot=False)    
    t = cumsum(fs)
    
    ind_mag = mlab.find (t>t[-1]*0.95)[0]
    f_max=f[ind_mag]
    return f_max


def median_frequency(sig,FS):
    """Compute median frequency along the specified axes.

    Parameters
    ----------
    sig: ndarray
        input from which median frequency is computed.
    FS: int
        sampling frequency    
    Returns
    -------
    f_max: int
       0.50 of max_frequency using cumsum.
    """
    
    f, fs = plotfft(sig, FS, doplot=False)    
    t = cumsum(fs)
    
    ind_mag = mlab.find (t>t[-1]*0.50)[0]
    f_median=f[ind_mag]
    return f_median

