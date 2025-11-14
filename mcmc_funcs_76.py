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

    h_off, b_alb, c_11 = params
    # change transit time according to data
    new_t, flux = kelp_models.kelp_transit(t, t0 = 57859.3174, per = 1.80988198, inc = 89.623, rp = 0.10852, ecc = 0,
                                    w = 51, a = 4.08, q = [0.01, 0.01], fp = 1.0, T_s = 6305.79,
                                    rp_a = 0.10852 / 4.08, limb_dark = 'quadratic', name = 'WASP-76b', channel = 'NIRSpec G395H',
                                    hotspot_offset = h_off, A_B = b_alb, c11 = c_11)

    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """

    h_off, b_alb, c_11 = params
    # change priors according to fit
    if (-np.pi < h_off < np.pi and 0.00 < b_alb < 1.00 and 0.00 < c_11 < 1.00):
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
    return -0.5 * np.sum((pc_model(params, t) - f)**2 / ferr**2)


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

    # Plot corner plot before removing strays
    corner(flatchain, truths=truths, labels=labels)
    plt.show()

    return None
