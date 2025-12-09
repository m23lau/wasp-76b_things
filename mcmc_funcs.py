# There are two other versions of mcmc_funcs, this one is for test purposes and uses Spitzer data on WASP-76b
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
    # ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, b_alb, c_11 = params
    # new_t, flux = kelp_models.kelp_transit(t, t0 = ttr, per = pd, inc = incl, rp = rad_p, ecc = 0,
    #                                 w = 51, a = semi_a, q = [q1, q2], fp = 1.0, T_s = 6305.79,
    #                                 rp_a = rad_p / semi_a, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
    #                                 hotspot_offset = h_off, A_B = b_alb, c11 = c_11)

    # # Remove q1, q2, incl, semi_a
    # ttr, pd, rad_p, h_off, b_alb, c_11 = params
    # new_t, flux = kelp_models.kelp_transit(t, t0 = ttr, per = pd, inc = 89.623, rp = rad_p, ecc = 0,
    #                                 w = 51, a = 4.08, q = [0.01, 0.01], fp = 1.0, T_s = 6305.79,
    #                                 rp_a = rad_p / 4.08, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
    #                                 hotspot_offset = h_off, A_B = b_alb, c11 = c_11)
    #
    # Only use last 3 kelp params
    h_off, b_alb, c_11 = params
    new_t, flux = kelp_models.kelp_transit(t, t0 = 57859.3174, per = 1.80988198, inc = 89.623, rp = 0.10852, ecc = 0,
                                    w = 51, a = 4.08, q = [0.01, 0.01], fp = 1.0, T_s = 6305.79,
                                    rp_a = 0.10852 / 4.08, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
                                    hotspot_offset = h_off, A_B = b_alb, c11 = c_11)

    # Uncomment these lines to test these three params ONLY (for sanity check)
    # ttr, pd, rad_p = params
    # new_t, flux = kelp_models.kelp_transit(t, t0=ttr, per=pd, inc=89.623, rp=rad_p, ecc=0,
    #                                        w=51, a=4.08, q=[0.01, 0.01], fp=1.0, T_s=6305.79,
    #                                        rp_a=rad_p / 4.08, limb_dark='quadratic', name='WASP-76b',
    #                                        channel=f"IRAC {2}",
    #                                        hotspot_offset=np.radians(-3), A_B=0.2, c11=0.33)
    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """
    # ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, b_alb, c_11 = params
    #
    # if (57859.31 < ttr < 57859.32 and 1.809 < pd < 1.811 and 89.0 < incl < 90.0
    #         and 0.105 < rad_p < 0.11 and 4.05 < semi_a < 4.10 and 0.00 < q1 < 0.02 and 0.00 < q2 < 0.02
    #         and np.radians(-10) < h_off < np.radians(5) and 0.10 < b_alb < 0.50 and 0.20 < c_11 < 0.50):
    #     prior_t0 = -0.5 * ((ttr - 57859.317) / 0.001)**2
    #     prior_per = -0.5 * ((pd - 1.80988198) / 0.00000064)**2
    #     prior_inc = -0.5 * ((incl - 89.623) / 0.034)**2
    #     prior_rad = -0.5 * ((rad_p - 0.10852) / 0.00096)**2
    #     prior_a = -0.5 * ((semi_a - 4.08) / 0.06)**2
    #     prior_offset = -0.5 * ((h_off - np.radians(-3)) / np.radians(10))**2
    #     return prior_t0 + prior_per + prior_inc + prior_rad + prior_a + prior_offset

    # # Remove q1, q2, incl, semi_a
    # ttr, pd, rad_p, h_off, b_alb, c_11 = params
    #
    # if 57859.31 < ttr < 57859.32 and 0.10 < b_alb < 0.50 and 0.20 < c_11 < 0.50:
    #     prior_per = -0.5 * ((pd - 1.80988198) / 0.00000064) ** 2
    #     prior_rad = -0.5 * ((rad_p - 0.10852) / 0.00096) ** 2
    #     prior_offset = -0.5 * ((h_off - np.radians(-3)) / np.radians(10)) ** 2
    #     return prior_per + prior_rad + prior_offset

    # Only use 3 kelp params
    h_off, b_alb, c_11, unc_scale = params

    if (np.radians(-10) < h_off < np.radians(15) and 0.05 < b_alb < 0.55 and 0.10 < c_11 < 0.55 and 0 < unc_scale < 100):
        return 0

    # Uncomment these lines to test these three params ONLY (for sanity check)
    # ttr, pd, rad_p = params
    #
    # if (57859.31 < ttr < 57859.32 and 1.809 < pd < 1.811
    #         and 0.105 < rad_p < 0.11):
    #     prior_t0 = -0.5 * ((ttr - 57859.317) / 0.001) ** 2
    #     prior_per = -0.5 * ((pd - 1.80988198) / 0.00000064)**2
    #     prior_rad = -0.5 * ((rad_p - 0.10852) / 0.00096)**2
    #     return prior_t0 + prior_per + prior_rad

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

    h_off, alb, c_11, unc_scale = params
    scaled_err = unc_scale * ferr

    resid = f - pc_model((h_off, alb, c_11), t)
    return -0.5 * np.sum((resid / scaled_err) ** 2 + np.log(2*np.pi * scaled_err**2))


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


def corner_plot(flatchain, labels, title):
    """ Scans for and takes out stray walkers, then plots a corner plot
    Args:
        flatchain (ndarray): 1D or 2D array that feeds into corner plot
        labels (list: str): List of labels for plot
        title (str or float): Word or number to be included in plot title
    Returns:
         ndarray: Modified flatchain
    """
    if type(title) == np.float64:
        title = round(title, 3)

    rd = []             # Rounded values
    truths = []         # Modal values for each parameter

    for i in flatchain:
        for j in i:
            rd.append(round(j, 6))

    rd = np.array(rd).reshape(len(flatchain), len(flatchain[0])).transpose()      # rd is now a 2D array where each row is one param
                                                                                  # but all of their values are rounded off
    for i in rd:
        truths.append(mode(i))

    # Plot corner plot
    corner(flatchain, truths=truths, labels=labels, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, plot_datapoints=False)
    plt.savefig(str(title)+'_corner.pdf')
    plt.close()

    return None

