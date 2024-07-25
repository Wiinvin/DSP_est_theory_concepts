import numpy as np
import sys, os
import math
import matplotlib.pyplot as plt


## calculate the fractional fourier transform of the complex signal for the 
## specified alpha value
#
def calc_frft(sig, alpha):

    siglen = len(sig)
    out_sig = np.zeros(np.shape(sig))

    ## get shifted version of the signal
    #
    tmp = np.arange(0, siglen, 1) +  np.floor(siglen/2)
    tmp_shifted = np.zeros((len(tmp)))
    for i in range(len(tmp)):
        tmp_shifted[i] = (tmp[i] % siglen) 
    ## end of for
    #
    
    sq_len = np.sqrt(siglen)

    ## get the value for unit circle, after 4th quarter everything repeats
    #
    alpha = alpha % 4

    ## at zero angle simply return the signal as is
    if (alpha == 0): 
        return sig;

    ## at 180 degrees signal flips itself. similar to taking fourier transform of the fourier transform
    if (alpha == 2): 
        return np.flip(sig,0);

    ## at 90 degrees, returned signal would be a fourier transform
    if (alpha == 1): 

        ## shift the signal based on shifted indices
        tmp_signal = [sig[int(ele)] for ele in tmp_shifted]
        tmpfft = np.fft.fft(tmp_signal, axis=0) / sq_len;
        
        ## shift fft again
        #
        shft_fft = [tmpfft[int(ele)] for ele in tmp_shifted]

        return shft_fft
    
    ## at 270 degrees, flipped fourier transform (probably) ---> Confirm this with plots and math
    #
    if (alpha == 3):
        ## shift the signal based on shifted indices
        tmp_signal = [sig[int(ele)] for ele in tmp_shifted]
        tmpifft = np.fft.ifft(tmp_signal, axis=0) * sq_len;

        ## shift ifft again
        #
        shft_ifft = [tmpifft[int(ele)] for ele in tmp_shifted]

        return shft_ifft
        
    ## reduce the interval to 0.5 and 1.5
    #
    if (alpha > 2.0):
        alpha = alpha - 2
        sig = np.flip(sig,0)

    if (alpha > 1.5):
        alpha = alpha - 1
        tmp_signal = [sig[int(ele)] for ele in tmp_shifted]        
        tmpfft = np.fft.fft(tmp_signal, axis=0) / sq_len;
        
        ## shift fft again
        #
        sig = [tmpfft[int(ele)] for ele in tmp_shifted]


    if (alpha < 0.5):
        alpha = alpha + 1
        tmp_signal = [sig[int(ele)] for ele in tmp_shifted]        
        tmpifft = np.fft.ifft(tmp_signal, axis=0) * sq_len;
        
        ## shift fft again
        #
        sig = [tmpifft[int(ele)] for ele in tmp_shifted]
        

    ## the general case for 0.5 < alpha < 1.5
    #
    alpha_truc = alpha * np.pi / 2.0
    
    ## 
    tana2 = np.tan(alpha_truc / 2.0)
    sina = np.sin(alpha_truc)

    ## chirp premultiplication
    #
    sinc_interp_sig = sinc_interp(sig)
    ## padded sig
    #
    sinc_interp_sig_padded = np.concatenate((np.zeros((siglen-1), dtype=np.complex), sinc_interp_sig))
    sinc_interp_sig_padded = np.concatenate((sinc_interp_sig_padded, np.zeros((siglen-1), dtype=np.complex)))

    ## padd zeros to both ends

    chrp = np.exp(-1j * np.pi/siglen * tana2/4 * np.arange(-2 * siglen+2, 2 * siglen-1)**2)

    f = chrp * sinc_interp_sig_padded

    ## chirp convolution
    #
    c = np.pi / siglen / sina/ 4
    
    conv_range_points = np.flip(np.arange(-(4*siglen - 4), (4 * siglen - 3)), 0) ** 2
    faf = fconv( np.exp( 1j * c * conv_range_points[:]), f)

    faf = faf[(4*siglen-3)-1: (8 *siglen-6)-1] * np.sqrt(c/np.pi)

    faf = chrp * faf

    ## calculate the fractional fourier transform
    #
    faf = np.exp(-1j * (1-alpha) * np.pi/4) * faf[siglen-1:len(faf)-siglen+1:2]

    ## return estimated coefficients gracefully
    #
    return faf

## end of method
#

## sinc interpolation
def sinc_interp(sig_a):

    siglen =len(sig_a)
    y = np.zeros((2 * siglen-1, 1), dtype = np.complex)

    ## subsample every 2nd element
    #
    y[0: 2*siglen-1: 2] = sig_a;

    ## interpolate
    #
    sinc_ind_points = np.arange(-(2*siglen-3), (2*siglen-2)) / 2.0

    xinterp = fconv( y[:2*siglen-1], np.sinc(sinc_ind_points[:]) )

    xinterp = xinterp[(2 * siglen -2) - 1: len(xinterp) - 2 * siglen + 3]

    ## return the sinc interpolated signal
    #
    return xinterp
## end of method
#
                 

## convolution
def fconv(x, y):

    ## convert input signal to one dimension
    #
    x = np.reshape(x, (len(x)))

    siglen = len(np.concatenate((x, y))) - 1
    p = 2 ** int(next_power_of_2(siglen))
    z = np.fft.ifft( np.fft.fft(x, n = p) * np.fft.fft(y, n = p) )
    z = z[:siglen]
    return z
## end of method
#


## get top / peaked n values from the graph
#
def get_top_n_alphas(abs_frft, npeaks, offset):

    tmp_abs_frft = np.copy(abs_frft)
    max_frft = np.max(tmp_abs_frft, axis=0)
    peakargs = []
    peakamps = []
    
    ## find the first peak
    #
    first_peak = np.argmax(max_frft)
    peakargs.append(first_peak)
    peakamps.append(max_frft[first_peak])
    npeaks -= 1

    ## essentially we are cleaning up the region around which the peak is 
    ## already detected. This way we can use argmax function over and over
    ## again on the same array, to get n different peaks
    #
    while npeaks != 0:

        ## update the regions which needs to be zeroed out so that other
        ## peaks could be found using the argmax function
        #
        if peakargs[-1] - offset > 0:
            clean_st = peakargs[-1] - offset
        else:
            clean_st = 0

        if peakargs[-1] + offset < len(abs_frft[0]):
            clean_end =  peakargs[-1] + offset
        else:
            clean_end = len(abs_frft[0])

        ## clean the entire region with zeros
        #
        max_frft[clean_st:clean_end] = 0.0

        ## update the peak indices and their amplitude values
        #
        peak = np.argmax(max_frft)
        peakargs.append(peak)
        peakamps.append(max_frft[peak])
        npeaks -= 1

    ## end of while
    #

    ## return peaks gracefully
    #
    return (peakargs, peakamps)
## end of method
#



## calculate next power of 2
#
def next_power_of_2(x):
    return 1 if x==0 else math.ceil(math.log(x,2))

## end of method
#
    






