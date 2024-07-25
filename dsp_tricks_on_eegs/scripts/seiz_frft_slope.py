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
from mpl_toolkits.mplot3d import Axes3D

## feats analysis modules
#
import line_spectrum as lspec
import frft

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
        stop_t = start_t + 30
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
        winsize = (stop_t - start_t) * samp_freq# seconds
        mdl_order = 18
        lag = int(5 * mdl_order)
        music_w = lspec.root_music(sig_sample, mdl_order, lag)
        amp, phi = lspec.freqaphi(sig_sample, music_w)


        nsamp = len(sig_sample)
        inc = 0.01
        unit_circle_st = 0
        unit_circle_end = 2
        alpha_range = np.arange(unit_circle_st, unit_circle_end + inc, inc)
        out_frft_a_ = np.zeros((nsamp, len(alpha_range)), dtype = np.complex)
        sig_sample = np.reshape(sig_sample, (len(sig_sample), -1))

        for alpha_val_i in range(len(alpha_range)):
            out_frft_a_[:,alpha_val_i]  = np.reshape(frft.calc_frft(sig_sample, alpha_range[alpha_val_i]), (len(frft.calc_frft(sig_sample, alpha_range[alpha_val_i]))))
        ## end of for
        #

        ## calculate the maximum energy of for each alpha values and 
        ## find out the chirp rate of the signal
        #
        std_frft = np.zeros((len(alpha_range), 1))
        max_frft = np.zeros((len(alpha_range), 1))
        abs_frft = np.zeros((np.shape(out_frft_a_)))

        for i in range(len(alpha_range)):
            abs_frft[:,i] = abs(out_frft_a_[:,i])
            max_frft[i] = np.max(abs(out_frft_a_[:,i]))
            std_frft[i] = np.std(abs(out_frft_a_[:,i]))

        ## end of for
        #

        ## get the peaks of the signals and find out the peak alpha values
        #
        nfrft = len(out_frft_a_)
        min_frft_var = np.min(std_frft) ** 2
        offset = 5#int(nfrft * min_frft_var)  ## skip these many samples
        peak_args, peak_amps = frft.get_top_n_alphas(abs_frft, 2, offset)
        print "peak args are: ", peak_args 
    
        ## calculate the chirp rate
        #
        chirp_rates = []
        initial_freq = []
        ending_freq = []

        for peak in peak_args:

            ## estimate alpha
        #
            alpha_est = alpha_range[peak]
            print "alpha estimates: ", alpha_est

#        phi_est = 360 * alpha_est / 2
        
            ## calculate phi value (rotation angle)
            #
            phi_est = alpha_est * (np.pi/2)

            ## calculate chirp rate (-cot(phi)1/nfrft)
            #
            chirp_rate = -1 / (np.tan(phi_est)) * 1 / nfrft
        
            chirp_rates.append(chirp_rate)

            ## calculate initial frequency and ending frequencies
            #
            ## Project the signal on the frequency domain Or find the fourier bin
            ## which has maximum value for frft transform axis
            #
        
            ## get peak's arg (argmax) at estimated alpha values
            #

            max_enrgy_bin = np.argmax(abs_frft[:,peak])

            ## following quantities have been estimated from the paper:
            ## Application of the fractional Fourier transform to \
            ## moving target detection in airborne SAR, by Sun Hong-Bo et al.
            #
            modulation_rate = np.tan(alpha_est)
            doppler_centroid = max_enrgy_bin / float(nfrft)
            omega_begin = doppler_centroid * ( 1 / np.sin(alpha_est) )
            print "modulation rate is: ", modulation_rate

            ## get the end frequency from the formula
            #
            omega_end = (chirp_rate * nfrft) + omega_begin
            
            initial_freq.append(omega_begin)
            ending_freq.append(omega_end)

 

        print "chirp rates are: ", chirp_rates
        print "initial freqs are: ", initial_freq
        print "end freqs are : ", ending_freq

        ## plot the graph for different alpha values
        #
        fig_size = plt.rcParams["figure.figsize"]
#    print "current size ", fig_size
        fig_size[0] = 8
        fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size

        ## start plotting the signal
        #
        plt.figure(1)
        spec = out_frft_a_

        plt_spec = plt.imshow(abs(spec), origin="lower", aspect=1/50.)
        plt.xlabel("alpha with increments 0.001")
        plt.ylabel("FrFt bins")

        import pdb;pdb.set_trace()
        fig = plt.figure(2)
        ax = Axes3D(fig)
        x1 = np.arange(0, len(alpha_range), 1)
        x1 = x1 * inc
        y1 = np.arange(0, len(out_frft_a_), 1)
#        y1 = np.arange(0, len(out_frft_a_), 30*nfrft*(float(len(out_frft_a_))) )
#        y1 = y1 / float(nfrft)
        y1 = y1 / (float(len(out_frft_a_))/(2.0 * np.pi))
        X, Y = np.meshgrid(x1, y1)
        z1 = abs_frft

        ax.set_facecolor('c')
        ax.plot_surface(X,Y,z1, rstride=1, cstride=1, cmap='jet')
        ax.set_xlabel("Alpha values")
        ax.set_ylabel("FrFt est. Normalized")
        ax.set_zlabel("Amplitude")

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
