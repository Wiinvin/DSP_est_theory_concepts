
## import necessary modules
#
import os, sys
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
import line_spectrum as lspec
from scipy.fftpack import fft
import math 

import matplotlib.pyplot as plt
#plt.style.use('style/elegant.mplstyle')

def main():
    
    ## load the lynx data
    #
    lynxdata_a = loadmat("lynxdata.mat")

    ## separate the fields from the loaded file
    #
    lynx = lynxdata_a['lynx']
    year = lynxdata_a['year']
    lynxlog = lynxdata_a['loglynx']
    dat_len = len(lynxlog)
    xs = np.arange(0, 1, 1.0/dat_len)
    mdl_orders = range(4, 19, 2)
    plot_title = " MUSIC "
    lag_division_factor = 6
    ## For HOYW on Lynx data
    #
    rel_mse = OrderedDict()
    aic_val = OrderedDict()
    aicc_val = OrderedDict()
    bic_val = OrderedDict()

    ## make sure the method works
    temp = lspec.high_order_yw(lynxlog, 2, 12, 100)

    ## initizlize the counter for subplots
    #
    j = 1

    for n in mdl_orders:

        m = n + 1

        print "\nOrder ", n, " MUSIC System\n"

        music_w = lspec.root_music(lynxlog, n, m)
        print "For m values: ", m, " freq. estimates are: ", music_w

        amp, phi = lspec.freqaphi(lynxlog, music_w)
        print "Corresponding Amps are: ", amp
        print "Corresponding phases are: ", phi

        ## compute FFT for reference
        fft_w = np.fft.fft(lynxlog, axis=0)
        p2 = abs(fft_w / dat_len);
        p1 = p2[: (dat_len/2) + 1]
        p1[1:] = 2*p1[1:]
        freq_range = np.linspace(0, math.pi, dat_len/2 + 1)

        ## plot the frequency estimates
        #
        ## infd the positive indices from the frequency estimations
        #
        ind = [i for i in range(len(music_w)) if (music_w[i] >= 0 and music_w[i] <= math.pi)]
        fh = [ (music_w[i]/(2*math.pi))   for i in ind] ## since 1 Hz = 2*pi radians
        amp_h = 2 * abs(amp[ind])

        ## do the same for the real frequencies, probably generated by FFT plots
        #
        f = freq_range[:] / (2*math.pi)

        ## plot both plots for comparison
        #
        plt.figure(1)
        plt.suptitle(plot_title + " N: " + str(dat_len)  + " m = n + 1")
        subplot_number = int(str(42) + str(j))
        plt.subplot(subplot_number)
        plt.subplots_adjust(top = 0.95, hspace = 0.6)
        plt.xlim(-0.02, 0.5)
        markerline, stemlines, baseline = plt.stem(fh, amp_h, '-.')
        plt.setp(baseline, color='r', linewidth=2)
        plt.plot(f, p1)
        plt.title("model order n: " + str(n))
        plt.legend(( "FFT Est", "Line Spec. Est"), loc='upper right', prop={'size': 8})
        plt.xlabel("Normalized Freq.", fontsize = 8)
        plt.ylabel("Amplitude", fontsize = 8)

        ## reconstruct the signal
        #
        reconst_sig = lspec.reconst_sig(music_w, amp, phi, dat_len)
        
        rel_mse[n] = lspec.calc_mse(reconst_sig, lynxlog)
        print "Relative Mean Squared Error is: ", rel_mse 

        aic_val[n] = lspec.aic(lynxlog, reconst_sig, n)
        print "AIC value for ", n, " parameters is: ", aic_val

        aicc_val[n] = lspec.aic(lynxlog, reconst_sig, n, correction=True)
        print "AICc value for ", n, " parameters is: ", aicc_val

        bic_val[n] = lspec.bic(lynxlog, reconst_sig, n)
        print "BIC value for ", n, " parameters is: ", bic_val

        plt.figure(2)
        plt.suptitle(plot_title + " N: " + str(dat_len) +  " m = n + 1")
        subplot_number = int(str(42) + str(j))
        plt.subplot(subplot_number)
        plt.subplots_adjust(top = 0.95, hspace = 0.6)
        plt.plot(xs, reconst_sig)
        plt.plot(xs, lynxlog)
        plt.title("model order n: " + str(n))
        plt.legend(("Reconstructed Signal", "Real Signal" ), loc='upper right', prop={'size': 8})
        plt.xlabel("Time", fontsize = 8)
        plt.ylabel("Amp", fontsize = 8)

        ## increment the counter for subplots
        #
        j += 1
        
    ## end of for
    #
    
    ## plot the mean squared error for model order
    #
    x_values = []
    y_values = []
    for k,v in rel_mse.iteritems():
        x_values.append(k)
        y_values.append(v[0][0])

    plt.figure(3)
    plt.plot(x_values, y_values, 'r--', marker='s')
    plt.title("Mean Squared Error estimate")
    plt.legend("MSE for " + plot_title + " n order models")
    plt.xlabel("Model order")
    plt.ylabel("MSE")
    
    plt.show()

    ## print the relative mean squared error
    #
    print "Relative MSE\n", rel_mse
    print "AIC\n", aic_val
    print "AICc\n", aicc_val
    print "BIC\n", bic_val

## end of method
#













    

    








if __name__ == "__main__": main()
