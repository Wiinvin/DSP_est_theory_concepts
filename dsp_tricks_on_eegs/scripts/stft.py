#!/usr/bin/env python
#

## import necessary packages
#
import numpy as np
from numpy.lib import stride_tricks
import os, sys
import math
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import line_spectrum as lspec

def slide_win(sig_a, winsize_a, hopsize_a, padd = True, win_align = "center"):

    ## pre and post padd signals before applygin sliding window approach
    #
    fillval = 0
    sig_a = np.asarray(sig_a) ## make sure that array is in numpy format
    sig_a = np.reshape(sig_a,(-1)) ## make sure that the array is one dimensional
    prepost_zeros = winsize_a / 2;
    padd_sig = np.zeros((prepost_zeros), dtype=type(sig_a))

    ## update the signal with zeros at the ends
    #
    sig_ext = np.concatenate( (np.full(prepost_zeros, fillval), sig_a))
    sig_ext = np.concatenate( (sig_ext, np.full(prepost_zeros, fillval)))

    ## collect memory parameters of the datasize
    #
    dat_byteshift = sig_ext.strides[0]
    row_stride = sig_ext.itemsize * hopsize_a
    col_stride = sig_ext.itemsize

    ## This section is to calculate number of rows for windowed sampled array
    #
    trail_samples = (len(sig_ext) - winsize_a) % hopsize_a
    possible_dat_frames = (len(sig_ext)  - trail_samples)
    out_rows = ((possible_dat_frames - winsize_a) / hopsize_a) + 1

    ## sliding window samples
    #
    out_sig = stride_tricks.as_strided( sig_ext, strides = (row_stride, col_stride),
                                        shape = (out_rows, winsize_a) )
    
    
    ## return window samples gracefully
    #
    return out_sig
    
## end of method
#


def get_xn(Xs,n):
    '''
    calculate the Fourier coefficient X_n of 
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
    return(xn)


## Uncomment the sections in the method to see the applied window on data.
#
def get_xns(ts, nfft, wintype_a, samprate = 1):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2, 
    to account for the symetry of the Fourier coefficients above the Nyquest Limit. 
    '''

    mag = []
    fft_mag = []
    L = len(ts)

    ## apply the type of window with the signal
    #
#    plt.figure(1)
#    tmp_x = ts
#    xaxis = np.arange(0, len(ts), 1)
#    plt.plot(xaxis, tmp_x)

    if wintype_a == "hamming":
	print " applying hamming window..."
	ts = ts * np.hamming(len(ts))
    elif wintype_a == "blackman":
	ts = ts * np.blackman(len(ts))
	print " applying blackman window..."

    ## temp section
#    plt.figure(2)
#    plt.plot(xaxis, ts)
    ##


    ## if the signal lenth is greater than the Nyquist rate
    ## zero padd it
    #
    if L < nfft:
	zeros = nfft- L
	prepadd = zeros/ 2;
	postpadd = zeros - prepadd
	fillval = 0

	##  padding arrays
	#
	ts = np.concatenate( (np.full(prepadd, fillval), ts), axis=0)
	ts = np.concatenate( (ts, np.full(postpadd, fillval)), axis=0)

    ## end of if with finished zero padding
    #
    
    ## end of if
    #


    ## temp section
#    plt.figure(3)
#    tmp_axis = np.arange(0,len(ts),1)
#    plt.plot(tmp_axis, ts)
#    plt.show()

    #

#    for n in range(int(nfft/2)): # Nyquest Limit
#        mag.append(np.abs(get_xn(ts,n))*2)

    fft_mag.append( fft(ts,axis=0) )
    fft_mag = np.transpose(fft_mag)
    fft_mag = fftshift(fft_mag)
#    freq_tmp = fftfreq(L, 1/samprate)
#    freq_tmp = fftshift(fft_mag)
    fft_mag = np.reshape(fft_mag, (np.shape(fft_mag)[0]))
    return(np.abs(fft_mag))#, freq_tmp)


## create spectrogram
#
#  out_samples could be in the length of number of samples of the signal
#  or could be estimates for each window
#
def create_spectrogram(ts, winlen, nfft, samprate = 1, wintype = "rect", out_samples = "signal", noverlap = None):

    ## apply the sliding window approach and get windowed signals
    #
    windowed_sig = slide_win(ts, winlen, noverlap)
    padded_zeros = winlen / 2

    if noverlap is None:
        noverlap = nfft/2
    noverlap = int(noverlap)

    ## Get all the start points from where the window was created
    ## This helps in defining STFT boundaries
    #
    starts = np.arange(-(padded_zeros), len(ts) + padded_zeros,
                        noverlap, dtype=int)

    # remove the first and last indices which had padding in them
    #
    if padded_zeros > 0:
        starts = starts[1:-1]

    ## loop through all the windows and collect FFT estimates
    #
    xns = []
    for i,start in enumerate(starts):

        ## short term fast fourier transform
        #

        ts_window = get_xns(windowed_sig[i], nfft, samprate, wintype) 
        xns.append(ts_window)
    ## end of for
    #

    ## convert array to desired format
    #
    specX = np.array(xns).T

    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)

    if out_samples == "signal":
	spec = expand_specest4sig_len(spec, len(ts))
    elif out_samples == "win":
	print "window section in future releases.."
	pass ## nothing to do, this is already done using windows, this is for future changes

    ## return the start points/boundaries and spectral estimated gracefully
    #
    return(starts, spec)



## create stmt spectrogram
#
#  out_samples could be in the length of number of samples of the signal
#  or could be estimates for each window
#
def create_stmt_spectrogram(ts, winlen, target_fcomp, cov_lag, wintype = "rect", out_samples = "signal", noverlap = None):

    ## apply the sliding window approach and get windowed signals
    #
    if noverlap is None:
        noverlap = len(ts)/2
    noverlap = int(noverlap)

    windowed_sig = slide_win(ts, winlen, noverlap)
    padded_zeros = winlen / 2


    ## Get all the start points from where the window was created
    ## This helps in defining STFT boundaries
    #
    starts = np.arange(-(padded_zeros), len(ts) + padded_zeros,
                        noverlap, dtype=int)

    # remove the first and last indices which had padding in them
    #
    if padded_zeros > 0:
        starts = starts[1:-1]
	
    ## loop through all the windows and collect FFT estimates
    #
    xns = []
    mdl_order = target_fcomp * 2

    for i,start in enumerate(starts):
        
        ## initialize empty ft_matrix
        #
        ft_mtx = np.ones((len(windowed_sig[i])/2), dtype=np.float32)
        ft_mtx = ft_mtx #* 0.35
        ## short term fast fourier transform
        #
        music_w = lspec.root_music(windowed_sig[i], mdl_order, cov_lag)
        amp, phi = lspec.freqaphi(windowed_sig[i], music_w)
        ## infd the positive indices from the frequency estimations
        #
        ind = [i for i in range(len(music_w)) if (music_w[i] >= 0 and music_w[i] <= math.pi)]
        fh = [ (music_w[i]/(2*math.pi))   for i in ind] ## since 1 Hz = 2*pi radians
        amp_h = 2 * abs(amp[ind])
#        print fh
#        print amp_h
#        import pdb;pdb.set_trace()
        for i in range(len(fh)):
            bin_est = fh[i] * len(windowed_sig[i])
            ft_mtx[int(bin_est)] = amp_h[i]
            if not ((int(bin_est)-1 >= 0) or (int(bin_est)+1 >=0.5)):
                ft_mtx[int(bin_est)-1] = amp_h[i]/2.0
                ft_mtx[int(bin_est)+1] = amp_h[i]/2.0

        xns.append(ft_mtx)
    ## end of for
    #

    ## convert array to desired format
    #
    specX = np.array(xns).T

    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)

    if out_samples == "signal":
	spec = expand_specest4sig_len(spec, len(ts))
    elif out_samples == "win":
	print "window section in future releases.."
	pass ## nothing to do, this is already done using windows, this is for future changes

    ## return the start points/boundaries and spectral estimated gracefully
    #
    return(starts, spec)



## end of method
#



## create styw spectrogram
#
#  out_samples could be in the length of number of samples of the signal
#  or could be estimates for each window
#
def create_styw_spectrogram(ts, winlen, target_fcomp, cov_lag, wintype = "rect", out_samples = "signal", noverlap = None):

    ## apply the sliding window approach and get windowed signals
    #
    if noverlap is None:
        noverlap = len(ts)/2
    noverlap = int(noverlap)

    windowed_sig = slide_win(ts, winlen, noverlap)
    padded_zeros = winlen / 2


    ## Get all the start points from where the window was created
    ## This helps in defining STFT boundaries
    #
    starts = np.arange(-(padded_zeros), len(ts) + padded_zeros,
                        noverlap, dtype=int)

    # remove the first and last indices which had padding in them
    #
    if padded_zeros > 0:
        starts = starts[1:-1]
	
    ## loop through all the windows and collect FFT estimates
    #
    xns = []
    mdl_order = target_fcomp * 2

    for i,start in enumerate(starts):
#        import pdb;pdb.set_trace()        
        ## initialize empty ft_matrix
        #
        ft_mtx = np.ones((len(windowed_sig[i])/2), dtype=np.float32)
        ft_mtx = ft_mtx * 0.0035
        ## short term fast fourier transform
        #
        hoyw_w = lspec.high_order_yw(windowed_sig[i], mdl_order, cov_lag, cov_lag)
        amp, phi = lspec.freqaphi(windowed_sig[i], hoyw_w)
        ## infd the positive indices from the frequency estimations
        #
        ind = [i for i in range(len(hoyw_w)) if (hoyw_w[i] >= 0 and hoyw_w[i] <= math.pi)]
        fh = [ (hoyw_w[i]/(2*math.pi)) for i in ind] ## since 1 Hz = 2*pi radians
        amp_h = 2 * abs(amp[ind])
#        print fh
#        print amp_h 

        for i in range(len(fh)):
            bin_est = (fh[i] * winlen)
#            print bin_est
#            print amp_h[i]
            if int(bin_est) == 125:
                import pdb;pdb.set_trace()
            ft_mtx[int(bin_est)] = amp_h[i]
            if not ((int(bin_est)-1 >= 0) or (int(bin_est)+1 >=0.5 * len(windowed_sig[i]))):
                ft_mtx[int(bin_est)-1] = amp_h[i]/2.0
                ft_mtx[int(bin_est)+1] = amp_h[i]/2.0

        xns.append(ft_mtx)
    ## end of for
    #

    ## convert array to desired format
    #
    specX = np.array(xns).T

    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)

    if out_samples == "signal":
	spec = expand_specest4sig_len(spec, len(ts))
    elif out_samples == "win":
	print "window section in future releases.."
	pass ## nothing to do, this is already done using windows, this is for future changes

    ## return the start points/boundaries and spectral estimated gracefully
    #
    return(starts, spec)



## end of method
#




## expand the estimates to all the sample points of the input signal
## the spectral estimates are done the windows, so we need to provide
## those estmates back to the signal samples to get uniform spectral
## dimensions for any arbitrary window hop sizes.
#
def expand_specest4sig_len(win_specest_a, siglen):

    ## initialize the variables to be used
    #
    win_specest = np.transpose(win_specest_a)
    total_win = np.shape(win_specest)[0]
    spec_len = np.shape(win_specest)[1]
    outsig = np.zeros((siglen, spec_len))

    hopsize = (siglen - 1) / (total_win-1)
    win_index = 0

    ## loop through all the windows and update fft estimate. (But the last)
    for i in range(total_win-1):
        for j in range(hopsize): ## this would be redundant for samples within the window/hop
	    ## update the frequency estimate for the window
	    #
            outsig[win_index + j, :] = win_specest[i, :]
	## update the index when the frame is over (i.e. next hopsize location)
	#
	win_index += hopsize
	## end of for
	#
    ## end of for
    #
      

    ## add the last estimate to the remaining samples
    #
    outsig[win_index] = win_specest[-1]

    ## return the spectral estimates gracefully
    #
    return np.transpose(outsig)

## end of method
#

def plot_spectrogram(spec, sample_rate, L, starts, mappable = None, Nxticks = 10, Nyticks = 10, plot_type = "real", xlab_offset = 0, ylab_offset = 0):

    ## set up basic parameters for creating an image
    #
    fig_size = plt.rcParams["figure.figsize"]
    print "current size ", fig_size
    fig_size[0] = 8
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size


    ## Different calculations for real Vs. complex signals
    #
    if plot_type == "real":

	## just half of the spectrum estimation is required. The rest is usually redundant
	#
	spec = spec[ len(spec)/2: ] ## This is just giving the plot from 0 to pi

        print len(spec)
#        approx_aspect = ## put some exponential formula here...
#        print approx_aspect
	plt_spec = plt.imshow( spec , origin="lower", aspect=800)
	## create ylim
	ks = np.linspace(0, spec.shape[0], Nyticks)
	ksf = get_freq_scale_vec(ks, L, sample_rate,  normalized_f = True)

	plt.yticks(ks, ksf)
	plt.ylabel("$\pi$ radians/samples (Normalized Freq.)")

    ## plot both sides of the spectrum, specifically for complex signals.
    #
    else:
	plt_spec = plt.imshow( spec , origin = "lower", aspect=1)
	pass

    ## end of if/else
    #

    ## work on the X-axis of the plot
    #
    total_ts_sec = L/sample_rate
    ## create xlim
    print xlab_offset
    print spec.shape[1]
    ts_spec = np.linspace(0 - (xlab_offset * sample_rate),
                          spec.shape[1] - (xlab_offset * sample_rate),
                          Nxticks)

    ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0,total_ts_sec*starts[-1]/float(L),Nxticks)]
    plt.xticks(ts_spec,ts_spec_sec)
    plt.xlabel("Time (sec)")


    plt.title("Spectrogram L={} winsize={}".format(L, 2*(starts[2]-starts[1]) + 1))
    plt.colorbar(mappable,use_gridspec=True)
    plt.tight_layout()
    plt.show()
    return(plt_spec)

## end of method
#




## get a proper scale for the estimated frequency graph. Normalized frequencies are the
## easiest, crosscheck the frequencies in Herts though.
#
def get_freq_scale_vec(ks, Npoints, sample_rate, normalized_f = True):

    ## calculate normalized frequency
    #
    if normalized_f:
	freq_omega = ["%6.2f"%(2*math.pi*n / (ks[-1]*2.0) /(2.0 * np.pi)) for n in ks]
	return freq_omega

    else:
	freq_Hz = ks*sample_rate/Npoints
	#    freq_Hz  = [int(i) for i in freq_Hz ] 
	freq_Hz  = ["%6.2f"%i for i in freq_Hz ]
	return(freq_Hz )

## end of method
#
