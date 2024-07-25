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
from sklearn import preprocessing
from scipy.stats import pearsonr

## feats analysis modules
#
import line_spectrum as lspec
import stft

DEF_EXT = "raw"
TOTAL_CH = 22

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

        winsize = 4
        start_t = 60
        stop_t = start_t + winsize
        t_range = np.arange(start_t, stop_t, 1/float(samp_freq))

        
        ovch_sig = np.zeros((TOTAL_CH, (stop_t * samp_freq) - (start_t * samp_freq)))

        ## get signal for all the channels
        for i in range(TOTAL_CH):
            ovch_sig[i] = prune_signal(signal, samp_freq, start_t, stop_t, i)

        ## calculate the coherence with respect to all other channels
        #
        calc_pearson = True
        normalize_chseg = False
        coher, pearsonr = calc_coher(ovch_sig, len(ovch_sig[0]), samp_freq, winsize,
                                     normalize = normalize_chseg,
                                     gen_pearson_coef = calc_pearson)

        fig, ax = plt.subplots()

        ## PLOT ALL CHANNELS COHERENCE ESTIMATES
        im = ax.imshow(np.transpose(coher), extent=[0,21,21,0], origin="upper")
        ax.set_xticks(np.arange(TOTAL_CH))
        ax.set_yticks(np.arange(TOTAL_CH))
        ax.set_xticklabels(["ch %d"%ele for ele in range(TOTAL_CH)])
        ax.set_yticklabels(["ch %d"%ele for ele in range(TOTAL_CH)])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                   rotation_mode="anchor")




        # ## PLOT THE PEARSON CORRELATION ESTIMATES BETWEEN THE CHANNELS
        # im = ax.imshow(np.transpose(pearsonr))
        # ax.set_xticks(np.arange(TOTAL_CH))
        # ax.set_yticks(np.arange(TOTAL_CH))
        # ax.set_xticklabels(["ch %d"%ele for ele in range(TOTAL_CH)])
        # ax.set_yticklabels(["ch %d"%ele for ele in range(TOTAL_CH)])

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #            rotation_mode="anchor")


        # # Loop over data dimensions and create text annotations.
        # for i in range(TOTAL_CH):
        #     for j in range(TOTAL_CH):
        #         text = ax.text(j, i, "%2.1f"%pearsonr[i, j], fontsize=7,
        #                        ha="center", va="center", color="w")
        # ax.set_title("Pearson R estimate")




        ## PLOT THE CHANNEL SPECIFIC STEM PLOTS
        # ch_x = 14
        # ch_y = 0
        # t_range = np.linspace(0, 0.5, len(coher[ch_x][ch_y*375: (ch_y*375 + 375)]))
        # im = ax.stem(t_range, coher[ch_x][ch_y*375: (ch_y*375 + 375)], '-.')
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #            rotation_mode="anchor")

        

        ## show the plot
        #
        fig.tight_layout()
        plt.show()




#         plt.figure(1)
# #        plt.imshow(np.transpose(coher), aspect = 1/float(len(coher[0])/float(len(coher))), extent=[0,21,0,21])
#         plt.imshow(np.transpose(coher), extent=[0,21,0,21], origin="upper")
#         plt.xticks(np.linspace(0,21,22))
#         plt.yticks(np.linspace(0,21,22))
        
#         plt.figure(2)
#         plt.imshow(np.transpose(pearsonr))
#         plt.xticks(np.linspace(0,21,22))
#         plt.yticks(np.linspace(0,21,22))

#         plt.show()

        # dt = 0.01
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(t_range, ovch_sig[0], t_range, ovch_sig[1])
        # axs[0].set_xlabel('time')
        # axs[0].set_ylabel('s1 and s2')
        # axs[0].grid(True)
        # cxy, f = axs[1].cohere(ovch_sig[0], ovch_sig[1], 250, 1. / dt)
        # axs[1].set_ylabel('coherence')
        # fig.tight_layout()
        # plt.show()

def calc_coher(ovch_sig, nfft, samp_freq, win=1, normalize=True, gen_pearson_coef=True):
    
    t_range = np.arange(0,win, 1/float(samp_freq))
    siglen = len(ovch_sig[0])
    total_ch = len(ovch_sig)
    welch_winsize = int(siglen / 3.0)
    
    out_sig = np.zeros(( total_ch, int(total_ch * siglen/2.0) ))
    pearson_r = np.zeros((total_ch, total_ch))

    for i in range(len(ovch_sig)):
        for j in range(len(ovch_sig)):

            f, cxy = signal.coherence(ovch_sig[i], ovch_sig[j], nfft = siglen, nperseg = welch_winsize)

            ## probably normalize the results first before loadin them
            ## into overall array
            #
            if normalize:
                cxy = np.reshape(cxy, (-1,1))
                min_max_scaler = preprocessing.MinMaxScaler()
                min_max_scaler.fit(cxy)
                cxy = min_max_scaler.fit_transform(cxy)
                cxy = np.reshape(cxy, (-1))
            ## end of if
            #

            ## collect the coherence results
            #
            out_sig[i][int(j*siglen/2.0):  int( (j*siglen/2.0) + (siglen/2.0) ) ] = cxy[1:]

            ## pearson_r calculation for each channel
            #
            pearson_r[i][j] = pearsonr(ovch_sig[i], ovch_sig[j])[0]

        ## end of for
        #
    ## end of for
    #

    ## return the coherence calculations
    #
    return out_sig, pearson_r
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



if __name__=="__main__": 
    main()
