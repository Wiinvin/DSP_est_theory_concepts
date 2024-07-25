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

        ## Compute FFTs for reference
        #
        dat_len = len(sig_sample)
        fft_w = np.fft.fft(sig_sample, axis=0)
        p2 = abs(fft_w / dat_len);
        p1 = p2[: (dat_len/2) + 1]
        p1[1:] = 2*p1[1:]
        freq_range = np.linspace(0, math.pi, dat_len/2 + 1)

        ## feat extraction (line spectrum)
        #
        winsize = 1 * samp_freq# seconds
        hopsize = winsize / 2
        mdl_order = 18
        cov_lag = int(winsize / 2)
        #windowed_sig_sample = stft.slide_win(sig_sample, winsize, hopsize)
        #ft_mtx = np.zeros((len(windowed_sig_sample), mdl_order/2), dtype=np.float32)
        start_bounds, spec_est = stft.create_styw_spectrogram(sig_sample,
                                                              winsize,
                                                              mdl_order/2,
                                                              cov_lag,
                                                              noverlap = hopsize)
        stft.plot_spectrogram(spec_est, 250, len(sig_sample), start_bounds, Nxticks = 15)

        hoyw_w = lspec.high_order_yw(sig_sample, mdl_order, cov_lag, cov_lag)
        amp, phi = lspec.freqaphi(sig_sample, hoyw_w)


        ## plot the frequency estimates
        #
        ## infd the positive indices from the frequency estimations
        #
        ind = [i for i in range(len(hoyw_w)) if (hoyw_w[i] >= 0 and hoyw_w[i] <= math.pi)]
        fh = [ (hoyw_w[i]/(2*math.pi))   for i in ind] ## since 1 Hz = 2*pi radians
        amp_h = 2 * abs(amp[ind])

        ## do the same for the real frequencies, probably generated by FFT plots
        #
        f = freq_range[:] / (2*math.pi)


        ## reconstruct the signal
        #
        reconst_sig = lspec.reconst_sig(hoyw_w, amp, phi, dat_len)

        ## plot the signal
        #
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t_range, sig_sample,'k')
        axs[0].set_xlim(start_t, stop_t)
        axs[0].set_ylim(DEF_AMP_RANGE[0], DEF_AMP_RANGE[1])
        axs[0].grid(True)
        axs[0].set_xlabel("time (sec)")
        axs[0].set_ylabel("Amp (microvolts)")

        axs[1].stem(fh, amp_h, '-.')
        axs[1].plot(f, p1,'r')
        axs[1].set_xlabel("Normalized Freq.")
        axs[1].set_ylabel("Amp (Freq. est) ")


        axs[2].plot(t_range, reconst_sig)
        axs[2].set_xlim(start_t, stop_t)
#        axs[2].set_ylim(DEF_AMP_RANGE[0], DEF_AMP_RANGE[1])
        axs[2].grid(True)
        axs[2].set_xlabel("time (sec)")
        axs[2].set_ylabel("Amp (microvolts)")


        plt.show()

## end of method
#


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


if __name__ == "__main__": main()
