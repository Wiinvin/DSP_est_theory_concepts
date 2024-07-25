#!/bin/bash

import os, sys
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
import stft
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def main():

    ## load the input signal
    #
    in_sig = loadmat("signal2.mat")['y']


    winsize = 35
    hopsize = winsize/2
    nfft = 256
    wintype = "rect"
#    wintype = "blackman"
#    wintype = "hamming"
#    in_sig = np.arange(1,10)

    windowed_sig = stft.slide_win(in_sig, winsize, hopsize)

    ## Initialize the fourier coeff matrix
    #
    dft_mtx = np.zeros((len(windowed_sig), nfft), dtype=np.float32)
    
    ## collect fourier coefficients for each window
    #
    for i in range(len(windowed_sig)):
        dft_mtx[i] = stft.get_xns(windowed_sig[i], nfft, wintype)

    in_sig = np.asarray(in_sig) ## make sure that array is in numpy format
    in_sig = np.reshape(in_sig,(-1)) ## make sure that the array is one dimensional

    ## create the spectrogram
    #
    start_bounds, spec_est = stft.create_spectrogram(in_sig, winsize, nfft, 
                                                     wintype = wintype, noverlap=hopsize)

    ## expand the spectral estimates captured for the windows back to
    ##  number of samples of the signal
    #
    
    ## plot the spectrogram based on estimated values
    #
    stft.plot_spectrogram(spec_est, 1, 1, len(in_sig), start_bounds)


    ### THIS SECTION IS FOR TESTING ###
    ## 
    # def get_signal_Hz(Hz,sample_rate,length_ts_sec):
    #     ## 1 sec length time series with sampling rate 
    #     ts1sec = list(np.linspace(0,np.pi*2*Hz,sample_rate))
    #     ## 1 sec length time series with sampling rate 
    #     ts = ts1sec*length_ts_sec
    #     return(list(np.sin(ts)))

    # sample_rate   = 4000
    # length_ts_sec = 3
    # ## --------------------------------- ##
    # ## 3 seconds of "digit 1" sound
    # ## Pressing digit 2 buttom generates 
    # ## the sine waves at frequency 
    # ## 697Hz and 1209Hz.
    # ## ---------------------n------------ ##
    # ts1  = np.array(get_signal_Hz(697, sample_rate,length_ts_sec)) 
    # ts1 += np.array(get_signal_Hz(1209,sample_rate,length_ts_sec))
    # ts1  = list(ts1)
    
    # ## -------------------- ##
    # ## 2 seconds of silence
    # ## -------------------- ##
    # ts_silence = [0]*sample_rate*1
    
    # ## --------------------------------- ##
    # ## 3 seconds of "digit 2" sounds 
    # ## Pressing digit 2 buttom generates 
    # ## the sine waves at frequency 
    # ## 697Hz and 1336Hz.
    # ## --------------------------------- ##
    # ts2  = np.array(get_signal_Hz(697, sample_rate,length_ts_sec)) 
    # ts2 += np.array(get_signal_Hz(1336,sample_rate,length_ts_sec))
    # ts2  = list(ts2)

    # ## -------------------- ##
    # ## Add up to 7 seconds
    # ## ------------------- ##
    # ts = ts1 + ts_silence  + ts2

    # A,B = stft.create_spectrogram(ts, 100, 256, samprate = sample_rate ,out_samples = "signal", noverlap=84)

    # stft.plot_spectrogram(B, 1, sample_rate, len(ts), A, Nxticks=26)













if __name__=="__main__": 
    main()
