import scipy.stats as stats
from scipy.optimize import least_squares, minimize
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
    #plt.plot(insig)
    #plt.show()

    sin_sig = insig[:100]
    noise_sig = insig[-100:]
    
    f_comp = np.linspace(0.01, 0.49, len(sin_sig) / 2 -1)
    x_range = np.linspace(0, len(sin_sig)-1, len(sin_sig))


    ## for testin purposes
    #
    #sin_sig = 5.0 * np.cos(2 * np.pi * 0.1 * x_range + (30 * np.pi / 180)) + np.random.normal(0, 0.03, len(noise_sig))
    #noise_sig = np.random.normal(0, 0.03, len(noise_sig))

    ## calculate the noise variance
    #
    noise_var = est_noise_pow(noise_sig)
    print ("Noise variance is: ")
    print (noise_var)


    h_mat = np.zeros((2*len(f_comp), len(x_range)))

    for i in range(len(f_comp)):
        cos_vec = np.cos(2 * np.pi * f_comp[i] * x_range)
        sin_vec = np.sin(2 * np.pi * f_comp[i] * x_range)

        h_mat[i] = cos_vec
        h_mat[len(f_comp) + i] = sin_vec



    h_mat = h_mat.T
    h_mat_pinv = np.dot(np.linalg.inv(np.dot(h_mat.T, h_mat)), h_mat.T)
    freqs = np.dot(h_mat_pinv, sin_sig)
    #freqs = np.linalg.inv(h_mat.T * h_mat) * h_mat.T * sin_sig
    #print (freqs)

    amp = np.zeros((len(f_comp)))
    phi = np.zeros((len(f_comp)))


    ## calculate amplitudes and phases
    #
    for i in range(len(f_comp)):
        amp[i] = np.sqrt(freqs[i] ** 2  + freqs[len(f_comp) + i] ** 2)
        phi[i] = np.arctan2(freqs[len(f_comp) + i], freqs[i]) * 180 / np.pi

    #import pdb;pdb.set_trace()
    #print (amp)
    #print (phi)
    #print (np.argmax(amp))

    print ("Frequency of the signal in Hz: ")
    print (f_comp[np.argmax(amp)] )
    print ("Frequency of the signal in Rads: ")
    print (f_comp[np.argmax(amp)] * 2 * np.pi)
    print ("Phase of the signal: ")
    print (phi[np.argmax(amp)])
    print ("Amplitude of the signal: ")
    print (amp[np.argmax(amp)])



    ## collect amplitudes from all the remaining frequencies
    #
    noise_amp = []
    max_cos_idx = np.argmax(amp)
    max_sin_idx = int(np.argmax(amp) + len(freqs)/2)

    ## add all the amplitudes which are not from estimated frequencies
    #
    for i in range(int(len(freqs)/2)):
        if i == max_cos_idx:
            continue
        ## end of if
        #
        noise_amp.append(np.sqrt(freqs[i] ** 2  + freqs[len(f_comp) + i] ** 2))
    ## end of for
    #
    #print (noise_amp)

    param_limit = np.linspace(0, np.pi, 180)
    buck1 = []
    buck2 = []
    for i in range(len(param_limit)):

        #print ("Initialization: ", param_limit[i])
        ## estimate using simple cost function
        #
        initial_params = [param_limit[i]]
        phi_results = least_squares(cost_function, x0 = [initial_params[0]],
                                    args = (sin_sig, f_comp[np.argmax(amp)] * 2 * np.pi, phi[np.argmax(amp)], np.sqrt(noise_var)))
        phi_est = phi_results.x
        #print ("Estimates...")
        #print (phi_est * 180 / np.pi)

        results = minimize(cost_function_alt, x0 = [initial_params[0]],
                           args = (sin_sig, f_comp[np.argmax(amp)] * 2 * np.pi, phi[np.argmax(amp)], np.sqrt(noise_var)))
        #print (results.x * 180 / np.pi)

        buck1.append(abs(phi_est * 180 / np.pi))
        buck2.append(abs(results.x * 180 / np.pi))

    plt.plot(param_limit, buck1)
    #plt.plot(param_limit, buck2)
    plt.title("Phase Estimation via LS")
    plt.xlabel("Initialization point")
    plt.ylabel("Estimated value")
    plt.show()

        
def cost_function(params, y_obs, f0, amp, noise_var):

    #amp = params[0]
    #f0 = params[1]
    phi = params[0]
    #var = params[3]
    
    x_range = np.linspace(0, len(y_obs)-1, len(y_obs))
    y_pred = amp * np.cos(2*np.pi*f0*x_range + phi)


    cost = y_obs - y_pred

    #print (cost)
    return cost


## Alternate to be used with minimize
#
def cost_function_alt(params, y_obs, f0, amp, noise_var):

    #amp = params[0]
    #f0 = params[1]
    phi = params[0]
    #var = params[3]
    
    x_range = np.linspace(0, len(y_obs)-1, len(y_obs))
    y_pred = amp * np.cos(2*np.pi*f0*x_range + phi) #+ np.random.normal(0, var, len(x_range))

    cost = sum((y_obs - y_pred)**2)

    #print (cost)
    return cost

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


    ## Estimate variance after calculating the mean
    ## which will have one degree of freedom
    #
    sig_var = 0
    for samp_i in range(len(sig_a)):

        sig_var += (abs(sig_a[samp_i] - sig_mean)) ** 2

    sig_var = 1 / (siglen - 1) * sig_var
 
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

    
