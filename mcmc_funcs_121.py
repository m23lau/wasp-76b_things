# There are two other versions of mcmc_funcs, this one is for JWST data on WASP-121b
# The MCMC workflow used in these files is similar to the one from https://github.com/bmorris3/kelp/blob/main/notebooks/demo.ipynb

import kelp_models
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from statistics import mode


# Create model
def pc_model(params, t):
    """ Model phase curve using kelp_models.kelp_transit
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
    Returns:
         ndarray: Observed flux from planet-star system
    """

    rad_p, h_off, alb, c_11 = params
    new_t, flux = kelp_models.kelp_transit(t, t0 = 60244.520345, per = 1.27492504, inc = 88.345, rp = rad_p, ecc = 0,
                                    w = 10, a = 3.8002, q = [0.1, 0.1], fp = 1.0, T_s = 6776.00,
                                    rp_a = rad_p / 3.8002, limb_dark = 'quadratic', name = 'WASP-76b', channel = 'NIRISS SOSS Order 1',
                                    hotspot_offset = h_off, A_B = alb, c11 = c_11)      # Change filter to Order 2 if working with that data

    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """

    rad_p, h_off, alb, c_11, unc_scale = params     # unc_scale is a scaling factor for uncertainty in case it needs to be increased
    if (0.0 < rad_p < 1.0 and -np.pi < h_off < np.pi and
            -1.0 < alb < 1.0 and 0.0 < c_11 < 1.0 and 0 < unc_scale < 100):
        return 0

    else:
        return -np.inf


def log_likelihood(params, t, f, ferr):
    """ Log likelihood, maximum likelihood estimator of parameters
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
        f (ndarray): Flux data points
        ferr (ndarray): Error on flux
    Returns:
        float: Log-likelihood estimate
    """
    rad_p, h_off, alb, c_11, unc_scale = params
    scaled_err = unc_scale * ferr

    resid = f - pc_model((rad_p, h_off, alb, c_11), t)
    return -0.5 * np.sum((resid / scaled_err)**2 + np.log(2*np.pi * scaled_err**2))


def log_prob(params, t, f, ferr):
    """ Log probability, the sum of log_likelihood and log_prior
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
        f (ndarray): Flux data points
        ferr (ndarray): Error on flux
    Returns:
        float or -infinity
    """
    lp = log_prior(params)

    if np.isfinite(lp):
        return lp + log_likelihood(params, t, f, ferr)

    else:
        return -np.inf


def corner_plot(flatchain, labels):
    """ Scans for and takes out stray walkers, then plots a corner plot
    Args:
        flatchain (ndarray): 1D or 2D array that feeds into corner plot
        labels (list: str): List of labels for plot
    Returns:
         ndarray: Modified flatchain
    """

    rd = []             # Rounded values
    truths = []         # Modal values for each parameter

    for i in flatchain:
        for j in i:
            rd.append(round(j, 6))

    rd = np.array(rd).reshape(len(flatchain), len(flatchain[0])).transpose()      # rd is now a 2D array where each row is one param
                                                                                  # but all of their values are rounded off
    for i in rd:
        truths.append(mode(i))

    # Corner plot
    corner(flatchain, truths=truths, labels=labels, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, plot_datapoints=False)
    plt.show()

    return None
