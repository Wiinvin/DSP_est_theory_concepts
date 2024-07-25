#!/usr/bin/env python
#

# import NEDC modules
#
import nedc_cmdl_parser as ncp
import nedc_file_tools as nft
import nedc_edf_reader as ner


# import system modules
#
import os
import sys
import math
import numpy as np
import struct
from numpy.lib import stride_tricks
import matplotlib
from scipy import signal
#matplotlib.use('agg')
import matplotlib.pyplot as plt


## feats analysis modules
#
import line_spectrum as lspec
import stft

DEF_EXT = "raw"

montage_01 = [
    'montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF',
    'montage = 1, F7-T3: EEG F7-REF -- EEG T3-REF',
    'montage = 2, T3-T5: EEG T3-REF -- EEG T5-REF',
    'montage = 3, T5-O1: EEG T5-REF -- EEG O1-REF',
    'montage = 4, FP2-F8: EEG FP2-REF -- EEG F8-REF',
    'montage = 5, F8-T4: EEG F8-REF -- EEG T4-REF',
    'montage = 6, T4-T6: EEG T4-REF -- EEG T6-REF',
    'montage = 7, T6-O2: EEG T6-REF -- EEG O2-REF',
    'montage = 8, A1-T3: EEG A1-REF -- EEG T3-REF',
    'montage = 9, T3-C3: EEG T3-REF -- EEG C3-REF',
    'montage = 10, C3-CZ: EEG C3-REF -- EEG CZ-REF',
    'montage = 11, CZ-C4: EEG CZ-REF -- EEG C4-REF',
    'montage = 12, C4-T4: EEG C4-REF -- EEG T4-REF',
    'montage = 13, T4-A2: EEG T4-REF -- EEG A2-REF',
    'montage = 14, FP1-F3: EEG FP1-REF -- EEG F3-REF',
    'montage = 15, F3-C3: EEG F3-REF -- EEG C3-REF',
    'montage = 16, C3-P3: EEG C3-REF -- EEG P3-REF',
    'montage = 17, P3-O1: EEG P3-REF -- EEG O1-REF',
    'montage = 18, FP2-F4: EEG FP2-REF -- EEG F4-REF',
    'montage = 19, F4-C4: EEG F4-REF -- EEG C4-REF',
    'montage = 20, C4-P4: EEG C4-REF -- EEG P4-REF',
    'montage = 21, P4-O2: EEG P4-REF -- EEG O2-REF'
]


DEF_AMP_RANGE = [-40, 60]

def main():
    
    ilist = "../lists/tmp_edf.list"
    
    flist = readflines(ilist)

    for edf_f in flist:
        print "working on file: ", edf_f
        
        edf_header = ner.return_header(edf_f)
        edf_dur = edf_header['n_records']
        edf_info = ner.load_edf(edf_f, 0, edf_dur, montage_01)

        samp_freq = edf_header['n_samples_per_record'][1]
        signal = edf_info[2]

        start_t = 50
        stop_t = start_t + 100
        ch = 15
        t_range = np.arange(start_t, stop_t, 1/float(samp_freq))
        sig_sample = prune_signal(signal, samp_freq, start_t, stop_t, ch)

	winsize = 35
	hopsize = winsize/2
	nfft = 256
	wintype = "rect"
#        wintype = "blackman"
#        wintype = "hamming"
#        in_sig = np.arange(1,10)

	windowed_sig = stft.slide_win(sig_sample, winsize, hopsize)

	## Initialize the fourier coeff matrix
	#
	dft_mtx = np.zeros((len(windowed_sig), nfft), dtype=np.float32)
    
	## collect fourier coefficients for each window
	#
	for i in range(len(windowed_sig)):
            dft_mtx[i] = stft.get_xns(windowed_sig[i], nfft, wintype)

	sig_sample = np.asarray(sig_sample) ## make sure that array is in numpy format
        sig_sample = np.reshape(sig_sample,(-1)) ## make sure that the array is one dimensional

        ## create the spectrogram
        #
        start_bounds, spec_est = stft.create_spectrogram(sig_sample, winsize, nfft, 
                                                     wintype = wintype, noverlap=hopsize)

        ## expand the spectral estimates captured for the windows back to
	##  number of samples of the signal
	#
    
	## plot the spectrogram based on estimated values
	#
	stft.plot_spectrogram(spec_est, 250, len(sig_sample), start_bounds)


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
    # import pdb;pdb.set_trace()
    # A,B = stft.create_spectrogram(ts, 100, 256, samprate = sample_rate ,out_samples = "signal", noverlap=84)

    # stft.plot_spectrogram(B, 1, sample_rate, len(ts), A, Nxticks=26)












# edf_sig_a: edf signal to be processed in numpy format collected from edf_reader
# samp_freq_q: sampling frequency of the signal
# start_t: start time in seconds
# stop_t: stop time in seconds
# ch_a: channel number (Zero indexed)
#
def prune_signal(edf_sig_a, samp_freq_a, start_t, stop_t, ch_a):

    start_samp = start_t * samp_freq_a
    stop_samp = stop_t * samp_freq_a

    return edf_sig_a[samp_freq_a][ch_a][start_samp: stop_samp]



def readflines(list_a):

    with open(list_a, 'r') as fl:
        return fl.read().splitlines()



if __name__=="__main__": 
    main()
