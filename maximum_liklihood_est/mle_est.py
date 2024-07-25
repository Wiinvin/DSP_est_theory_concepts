import scipy.stats as stats
from scipy.optimize import minimize, minimize_scalar
import numpy as np
np.set_printoptions(suppress=True)
import time

import sympy
from sympy.abc import x,y
sympy.init_printing(use_latex='mathjax')

from scipy.io import loadmat
from scipy import signal


import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    
    insig = loadmat('sine_plus_awgn.mat')
    insig = insig['x'][0]
    sin_sig = insig[:100]
    noise_sig = insig[-100:]

    ## For testing the known case
    #x_range = np.linspace(0,99, 100)
    #sin_sig = 5.0 * np.cos(2 * np.pi * 0.1 * x_range + (30 * np.pi / 180)) + np.random.normal(0, 0.03, len(insig[-100:]))
    #noise_sig = np.random.normal(0, 0.03, len(insig[-100:]))
    #plt.plot(insig)
    #plt.show()

    ## calculate the noise variance
    #
    noise_var = est_noise_pow(noise_sig)

    sig_var = np.var(sin_sig)
    print ("Noise variance is: ")
    print (noise_var)
    #print ("signal variance is: ")
    #print (sig_var)
    
    ## estimate the peak frequency from the periodogram
    #
    fx, pxx_den = signal.periodogram(sin_sig)

    ## estimate the noise variance (assuming the signal is normalized
    #
    noise_power = np.mean(pxx_den)

    highest_freq_i = np.where(pxx_den == pxx_den.max())[0][0]
    f_hz =  highest_freq_i / len(sin_sig)
    f_rad = highest_freq_i / len(sin_sig) * (2 * np.pi)
    print ("Estimated frequency in Hz: ")
    print (f_hz)
    print ("Estimated frequency in Rad: ")    
    print (f_rad)

    ## estimate the phase of the sinusoid
    #
    initial_phi_params = [0]
    phi_results = minimize(est_sin_phi_loglik, x0 = initial_phi_params, bounds = [(0, np.pi)], args = (f_hz, sin_sig, noise_var), method = 'L-BFGS-B')
    phi_est = phi_results.x[0]
    phi_est_hz = phi_results.x[0] * 180 / np.pi
    print ("Estimated phase: ")
    print (phi_est_hz)

    ## estimate the amplitude of the sinusoid
    #
    initial_amp_params = [0]
    amp_results = minimize(est_sin_amp_loglik, x0 = initial_amp_params, bounds = [(0, 10)], args = (f_hz, phi_est, sin_sig, noise_var), method = 'L-BFGS-B')
    amp_est = amp_results.x[0]
    print ("Estimated Amplitude: ")
    print (amp_est)


def est_sin_phi_loglik(params, f0, y_obs, noise_var):

    phi = params[0]
    x_range = np.linspace(0, len(y_obs)-1, len(y_obs))
    y_pred = np.cos(2*np.pi*f0*x_range + phi)

    logpdf = stats.norm.logpdf(y_obs, y_pred, scale=np.sqrt(noise_var))
    loglik = -np.sum( logpdf )

    #print (loglik)
    return loglik


def est_sin_amp_loglik(params, f0, phi, y_obs, noise_var):

    amp = params[0]
    x_range = np.linspace(0, len(y_obs)-1, len(y_obs))
    y_pred = amp * np.cos(2*np.pi*f0*x_range + phi)

    logpdf = stats.norm.logpdf(y_obs, y_pred, scale=np.sqrt(noise_var))
    loglik = -np.sum( logpdf )

    #print (loglik)
    return loglik



def est_noise_pow(sig_a):

    ## initialize necessary variables
    #
    siglen = len(sig_a)

    ## mean value of the signal
    #
    samp_sum = 0
    for sample in sig_a:
        samp_sum += sample
    
    sig_mean = samp_sum / siglen
    #print ("Signal mean is : ", sig_mean)

    ## Estimate variance after calculating the mean
    ## which will have one degree of freedom
    #
    sig_var = 0
    for samp_i in range(len(sig_a)):

        sig_var += (abs(sig_a[samp_i] - sig_mean)) ** 2

    sig_var = 1 / (siglen - 1) * sig_var
    #print (sig_var)
    ## although we already know that it's a zero mean process so
    ## reiterating the same loop without the mean value 
    #
    sig_var = 0
    for samp_i in range(len(sig_a)):

        sig_var += (sig_a[samp_i]) ** 2

    sig_var = 1 / (siglen) * sig_var

    ## return the estimated power gracefully
    #
    return sig_var

        
if __name__ == "__main__": main()
