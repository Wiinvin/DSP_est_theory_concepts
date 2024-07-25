import numpy as np
import math
import scipy
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

#%matplotlib inline
import matplotlib.pyplot as plt
#plt.style.use('style/elegant.mplstyle')

## TODO: (1) Add forward-backward approach (exchange/reversal matrix implementation)
#        (2) Modularize the methods so that they can work with generators
#        (3) The code only works with real numbers, make it general purpose
#        (4) Covariance methods could be added in a particular class
#        (5) Inherit Cov class in line spectrum class
#

## MUSIC method to collect estimated frequency components
#
#
#
#
#
## The m = n + 1 (cov_order = mdl_order + 1), this method becomes Pisarenko method
## which is very efficient for computational complexity
## But note that, accuracy of MUSIC increases significantly with increasing the 
## cov_order (lag).
## lag could be chosen as large as possible but not close to N (total data samples).
## 
def root_music(dat_vec, mdl_order, cov_order):

    ## calculate the covariance matrix
    #
    corr_mtx = cov(dat_vec, cov_order)

    ## perform singular value decomposition
    #
    uh, dh, vh = np.linalg.svd(corr_mtx)

    ## we are only interested in (corr_mtx).T * (corr_mtx)'s eigenvectors
    ## so let's collect last mdl_order columns from the U matrix. Ignore V.
    ## This is related to the Noise
    #
    g = uh[:, mdl_order: cov_order+1]

    ## precalculate G.G* of the denominator
    #
    c = np.dot(g, np.transpose(g))

    ## find the coefficients of the polynomial 
    #
    pols = np.zeros((cov_order*2-1), dtype=np.float32) 

    ## ignore the root at 0
    #
    for i in range(1, cov_order*2):
        pols[i-1] = sum(np.diag(c, i-cov_order))

    ## end of for
    #

    ## collect all the roots from the polynomials
    #
    ra = np.roots(pols)

    ## sort the roots and collect the n closest from the unit circle
    ## the roots are supposed to be inside the unit circle
    #
    
    ## get sorted arguments for roots
    #
    sort_ind = np.argsort(abs(ra))
    
    ## sort the roots
    #
    sorted_roots = ra[sort_ind[: cov_order-1]]

    ## pick the n roots closest to the unit circle
    #
    closest_ind = np.argsort(abs(abs(sorted_roots)-1))
    sorted_closest_roots = sorted_roots[closest_ind[:]]

    freq_est = []
    
    ## collect frequency estimates based on model order that you have selected
    #
    for i in range(mdl_order):

        freq_est.append(math.atan2(sorted_closest_roots[i].imag,
                                   sorted_closest_roots[i].real))

    ## return the frequency estimates gracefully
    #
    return freq_est

## end of method
#

## ESPRIT method for frequency estimation
#
#
#
#
#
#
#
#
#
def esprit(dat_vec, mdl_order, cov_order):
    
    ## calculate the covariance matrix
    #
    corr_mtx = cov(dat_vec, cov_order)

    ## perform singular value decomposition
    #
    uh, dh, vh = np.linalg.svd(corr_mtx)

    sh = uh[ : , :mdl_order]

    
    s1_pseudoinv = np.linalg.pinv(sh[ : cov_order-1, :])
    s2 = sh[ 1: cov_order, :]
    phi = np.dot(s1_pseudoinv, s2)

    ## get eigenvalues of phi
    #
    phi_eigvals, phi_eigvecs = np.linalg.eig(phi)
    
    freq_est = []
    
    ## collect frequency estimates based on model order that you have selected
    #
    for i in range(mdl_order):

        freq_est.append(math.atan2(phi_eigvals[i].imag,
                                   phi_eigvals[i].real))

    ## end of for
    #


    ## return frequency estimates gracefully
    #
    return freq_est
## end of method
#




## run higher order yule walker to estimate frequencies
#
# dat_vec:
#
# ar_mdl_order:
#
# yw_mat_cols:
#
# yw_mat_rows: 
#
#
#
## Always make sure that L+M (yw_mat_cols+yw_mat_rows) are in between
## N/3 and N/2. where N is number of samples in the data vector
#  Very high L+M will not help estimating the higher lag covariances.
#
def high_order_yw(dat_vec, ar_mdl_order, yw_mat_cols, yw_mat_rows):

    ## collect the estimated and one sided Covariance matrix
    #
    rs, cov_mtx = one_sided_cov(dat_vec, yw_mat_cols, yw_mat_rows)
    
    ## perform SVD
    #
    uh, sh, vh = np.linalg.svd(cov_mtx)

    ## make sh diagonal
    #
    sh  = np.diag(sh)

    ## find the matrix for the approximated omega (cov mtx) with rank of ar_mdl_order
    ## Get rid of the remaining M-n and L-n elements in U and V respectively.
    ## They point to the null space anyway (Noise eigenvectors)
    #
    u1 = uh[ :, :ar_mdl_order]
    s1 = sh[ :ar_mdl_order, :ar_mdl_order]
    v1 = np.transpose(vh[ :ar_mdl_order,  :])

    ## compute the estimates of the b polynomial in eq 4.4.8 and 4.4.16
    ## This is the least squares solution for the truncated  HOYW system
    #
    b = reduce(np.dot, [ -v1, np.linalg.inv(s1), np.transpose(u1),
                        rs[yw_mat_cols: yw_mat_cols + yw_mat_rows] ])
    
    ## prepend 1 in the roots as the first AR root
    #
    b = np.insert(b, 0, 1)

    ## find the roots of b
    #
    rb = np.roots(b)

    ## sort the roots to find out the n closest roots from the unit circle
    #
    sort_ind = np.argsort(abs(abs(rb)-1))
    
    ## sort the roots
    #
    sorted_roots = rb[sort_ind[: ar_mdl_order]]

    freq_est = []
    
    ## collect frequency estimates based on model order that you have selected
    #
    for i in range(ar_mdl_order):

        freq_est.append(math.atan2(sorted_roots[i].imag,
                                   sorted_roots[i].real))

    ## end of for
    #

    ## return the estimated frequencies
    #
    return freq_est

## end of method
#


## calculated the one sided covariance matrix for high order yule walker equations
#
def one_sided_cov(dat_vec, yw_mat_cols, yw_mat_rows):

    dat_len = len(dat_vec)
    
    ## initialize the one sided covariance matrix
    #
    r = np.zeros((yw_mat_cols + yw_mat_rows, 1), dtype=np.float32)

    ## create the matrix
    #
    for i in range(1, yw_mat_cols + yw_mat_rows + 1):
        r[i-1] = np.dot(np.transpose(dat_vec[ :dat_len-i]),
                        dat_vec[i :dat_len]) / dat_len

    ## end of for
    #

    ## form the covariance matrix in 4.4.8 (the approx. version)
    #
    omega = scipy.linalg.toeplitz(r[yw_mat_cols-1:yw_mat_cols + yw_mat_rows-1],
                                  np.flip(r[ :yw_mat_cols],0))

    ## return the matrix gracefully
    #
    return r, omega

## end of method
#
                               

## form a covariance method matrix
#
# dat_vec: should be generator function which dumps data after 
#          reading from a file/files.
#
# order: The expected size of covariance matrix.
#
# padding: could be "none", "prepadding", "postpadding", "boundaries"
#          This will padd zeros to boundaries or just directly start
#          from the first data sample.
#
# method: "iterative", "full"
#         Iterative method will update covariance matrix iteratively
#         which can save memory that we work with.. When full option
#         is passed, all the data samples will be dumped into RAM
#         at the same time.
#
# TODO : UPDATE THE ITERATIVE METHOD AND FLAGS LATER
#

def cov(dat_vec, order, padding="none", method="iterative"):
    
    ## prepadd /postpadd section begins
    #

    ## prepadd / postpadd section ends here
    #

    dat_len = len(dat_vec)

    ## initialize the correlation matrix
    #
    r = np.zeros((order, order), dtype=np.float32)
    
    ## loop through all the elements
    #
    for i in (range(len(dat_vec) - order + 1)):

        ## matrix multiplication of the section and normalization before
        ## adding it to the previous estimate
        #
        r += np.flip(dat_vec[i: i+order],0) * \
             np.flip(np.transpose(dat_vec[i: i+order]),0) / dat_len

    ## end of for
    #

    ## return cov/corr matrix gracefully
    #
    return r

## end of method
#


## estimate amplitude and phases of the vector based on estimated angular
## frequencies
#
def freqaphi(dat_vec, w_est):

    ## get the length of the input arguments
    #
    dat_len = len(dat_vec)
    mdl_order = len(w_est)

    ## split the elements of the Vandermonde matrix
    ## First part is the ewn => exp(omega order#)
    ## Second part is the power with what first part is multiplied with (int multiple of freq.)
    ## for each data sample
    #
    ewn = np.ones((dat_len, 1)) * np.exp(1j*np.transpose(w_est))
    power = np.reshape(np.transpose(np.arange(0, dat_len)), [-1,1]) * np.ones((1, mdl_order))
    
    ## create the Vandermonde Matrix --> eq. 4.3.5
    #
    b = (ewn ** power)

    ## Calculate the Beta value from the pseudoinverse of created Vandermonde matrix
    ## and data vector by taking the dot product --> eq. 4.3.8
    #
    beta = np.dot(np.linalg.pinv(b), dat_vec)

    ## calculate the nuisance parameters
    #
    amp = abs(beta)

    ## collect frequency estimates based on model order that you have selected
    #
    angle = []
    for i in range(mdl_order):

        angle.append(math.atan2(beta[i].imag,
                                beta[i].real))

    ## end of for
    #

    ## return nuisance parameters gracefully
    #
    return amp, angle

## end of method
#


###################################
# ## INFORMATION CRITERIA METHODS #
###################################

## Akaike Information Criterion (AIC)
#
#
#
def aic(dat_vec, pred_vec, total_params, correction = False):

    ## calculate negative log-liklihood based on model predictions
    #
    ## This would be the negative sum of the log of a normal PDF where
    ## the data vector are normally distributed around the mean value (pred_vec)
    ## and standard deviation of estimated value
    #
    neg_loglik = -(calc_loglik(dat_vec, pred_vec))

    ## calculate the AIC value based on the formula
    ## check if correction is applied or not (for # samples less than 60)
    #
    if not correction:
        aic = 2 * total_params - 2 * (calc_loglik(dat_vec, pred_vec))
    else:
        aic = (2 * total_params * len(dat_vec)/(len(dat_vec) - total_params - 1)) - \
              2 * (calc_loglik(dat_vec, pred_vec))        
    
    ## return aic gracefully
    #
    return aic

## end of method
#


def bic(dat_vec, pred_vec, total_params):

    ## calculate negative log-liklihood based on model predictions
    #
    ## This would be the negative sum of the log of a normal PDF where
    ## the data vector are normally distributed around the mean value (pred_vec)
    ## and standard deviation of estimated value
    #
    neg_loglik = -(calc_loglik(dat_vec, pred_vec))
    dat_len = len(dat_vec)

    ## calculat BIC value based on its formula
    #
    bic = (total_params * np.log(dat_len))- 2 * (calc_loglik(dat_vec, pred_vec))

    ## return the bic value gracefully
    #
    return bic
## end of method
#


## calculate the loglikelihood

## UNFORTUNATELY THIS ASSUMES THAT THE DISTRIBUTION IS NORMAL
## IF THE REAL DISTRIBUTION IS NOT NORMAL, THEN IT WILL HAVE ILL EFFECTS
# 
def calc_loglik(dat_vec, pred_vec):

    ## calculate stardard deviation for scaling purposes
    #
    sd = np.std(pred_vec)
    sd = np.std(dat_vec)

    ## return gracefully
    #
    return np.sum(stats.norm.logpdf(dat_vec, loc=pred_vec, scale=sd))

## end of method
#    


## reconstruct_sig: Signal reconstruction method for line spectrum results 
#
# 
#
#
#
#
#
# Simply use cosines and sines to reconstruct signal using exponentials
#
def reconst_sig(w, a, p, dat_len):

    ## initialize the signal
    #
    signal = np.zeros((dat_len, 1), dtype=np.complex_)
    xs = np.arange(0, dat_len)

    ## sanity check
    ## make sure the parameters are of same size
    #
    if (len(w) != len(a) != len(p)):
        print "%s:%s Error: w, a, p paramaters should have equal length" \
            %(__name__, "reconstruct_sig")
        exit(-1)

    ## iterate through all the frequency components and generate corresponding
    ## frequencies (and its associated amp, phases)
    #
    for i in range(len(w)):
        for j in range(dat_len):
            signal[j] += a[i] * np.exp(-1j*w[i] * xs[j] + p[i])
        ## end of for for samples
        #
    ## end of for for individual frequency components
    #

    ## return the reconstructed signal gracefully
    #
    return signal

## end of method
#



## end of method
#



## plotting method to plot frequency plots 
# w: input frequencies from the real signal, fft preferred
# a: amplitude values of the signal, calculated from the fft.
# wh: high resolution spectral est method related angular frequencies
# ah: corresponding estimated amplitudes (from nonlinear least squares method)
#
def freqplot(w, a, wh, ah):

    ## Use this snippet if confused for FFTs
    #
        #######################################################
        # fft_w = np.fft.fft(in_dat, axis=0)                  #
        # p2 = abs(fft_w / dat_len);                          #
        # p1 = p2[: (dat_len/2) + 1]                          #
        # p1[1:] = 2*p1[1:]                                   #
        # freq_range = np.linspace(0, math.pi, dat_len/2 + 1) #
        # freqplot(freq_range, p1, ...)                       #
        #######################################################

    ## find the positive indices from the frequency estimations
    #
    ind = [i for i in range(len(wh)) if (wh[i] >= 0 and wh[i] <= math.pi)]
    fh = [ (wh[i]/(2*math.pi))   for i in ind] ## since 1 Hz = 2*pi radians
    ah = 2 * abs(ah[ind])

    ## do the same for the real frequencies, probably generated by FFT plots
    #
    f = w[:] / (2*math.pi)
    
    ## plot both plots for comparison
    #
    plt.figure()
    plt.xlim(-0.02, 0.5)
    markerline, stemlines, baseline = plt.stem(fh, ah, '-.')
    plt.setp(baseline, color='r', linewidth=2)
    plt.plot(f, a)
    plt.show()

    ## return gracefully
    #

## end of method
#                    
    
## calculate mean sqaured error between two signals
#
# y_pred: pass in the predicted value of the signal (reconstructed signal)
#
# y: pass in the original signal
## This method calculates the mean squared error between two real signals.
#
def calc_mse(y_pred, y):

    ## Calculate the MSE by taking the ratio between predicted signal squred and 
    ## original signal squared.
    #
    mse = 1 - ( np.dot(np.transpose(y_pred), y_pred)) / (np.dot(np.transpose(y), y))

    ## return the absolute value of MSE
    mse = abs(mse)

    ## return the error value gracefully
    #
    return mse

## end of method
#
