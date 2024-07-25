#!/usr/bin/env python

import os, sys
import numpy as np
from sympy import *
from numpy import linspace, pi, sin
from collections import OrderedDict
from scipy.io import loadmat
import stft
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import frft
from itertools import izip
from mpl_toolkits.mplot3d import Axes3D

def main():

    ## load the input signal
    #
    in_sig = loadmat("signal2.mat")['y']


    ## calculate frft for alpha values
    #
    nsamp = len(in_sig)
    inc = 0.001
    unit_circle_st = 0
    unit_circle_end = 2
    alpha_range = np.arange(unit_circle_st, unit_circle_end + inc, inc)
    out_frft_a_ = np.zeros((nsamp, len(alpha_range)), dtype = np.complex)
    
    for alpha_val_i in range(len(alpha_range)):
        out_frft_a_[:,alpha_val_i]  = np.reshape(frft.calc_frft(in_sig, alpha_range[alpha_val_i]), (len(frft.calc_frft(in_sig, alpha_range[alpha_val_i]))))

    ## end of for
    #

    ## calculate the maximum energy of for each alpha values and 
    ## find out the chirp rate of the signal
    #
    mean_frft = np.zeros((len(alpha_range), 1))
    std_frft = np.zeros((len(alpha_range), 1))
    max_frft = np.zeros((len(alpha_range), 1))
    abs_frft = np.zeros((np.shape(out_frft_a_)))

    for i in range(len(alpha_range)):
        abs_frft[:,i] = abs(out_frft_a_[:,i])
        max_frft[i] = np.max(abs(out_frft_a_[:,i]))
        mean_frft[i] = np.mean(abs(out_frft_a_[:,i]))
        std_frft[i] = np.std(abs(out_frft_a_[:,i]))

    ## end of for
    #

    ## get the peaks of the signals and find out the peak alpha values
    #
    nfrft = len(out_frft_a_)
    min_frft_var = np.min(std_frft) ** 2
    offset = int(nfrft * min_frft_var)  ## skip these many samples
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

        ## initial frequency can be calculated using taking derivative of
        ## (the rate of change) of the phase. Check wikipedia for this
        #
        ## create a phase vector 
        #
        # phase_vec = [np.angle(x) for x in in_sig]
        # phase_vec = np.reshape(phase_vec, (len(phase_vec)))
        # t = np.arange(0, len(in_sig), 1)
        # for ((a, b), (c, d)) in zip(pairwise(phase_vec), pairwise(t)):
        #     print (d-c)/(b-a)
        # yprime = y.diff(phase_vec)
        # f = lambdify(x, yprime, 'numpy')
        # df = f(y)
        # inst_freq = phi_est / (2 * np.pi)
        # end_freq = (chirp_rate * nfrft) + inst_freq
        ## This was the wrong approach...
 

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

    plt_spec = plt.imshow(abs(spec), origin="lower", aspect=8)
    plt.xlabel("alpha with increments 0.001")
    plt.ylabel("FrFt bins")


    fig = plt.figure(2)
    ax = Axes3D(fig)
    x1 = np.arange(0, len(alpha_range), 1)
    x1 = x1 * inc
    y1 = np.arange(0, len(out_frft_a_), 1)
    y1 = y1 / (float(len(out_frft_a_))/(2.0 * np.pi))
    X, Y = np.meshgrid(x1, y1)
    z1 = abs_frft

    ax.set_facecolor('c')
    ax.plot_surface(X,Y,z1, rstride=1, cstride=1, cmap='jet')
    ax.set_xlabel("Alpha values")
    ax.set_ylabel("FrFt est. Normalized")
    ax.set_zlabel("Amplitude")

    plt.show()


def pairwise(iterable): 
    a = iter(iterable)
    return izip(a, a)




if __name__ == "__main__": main()    



