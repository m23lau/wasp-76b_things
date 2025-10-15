import kelp_models
import numpy as np

# Create model
def pc_model(params, t):
    """ Model phase curve using kelp_models.kelp_transit
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
    Returns:
         ndarray: Observed flux from planet-star system
    """
    ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, p_s, b_alb, c_11 = params
    new_t, flux = kelp_models.kelp_transit(t, t0 = ttr, per = pd, inc = incl, rp = rad_p, ecc = 0,
                                    w = 51, a = semi_a, q = [q1, q2], fp = 1.0, t_secondary = 0, T_s = 6305.79,
                                    rp_a = rad_p / semi_a, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
                                    hotspot_offset = h_off, phase_shift = p_s, A_B = b_alb, c11 = c_11)
    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """
    ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, p_s, b_alb, c_11 = params

    if (57859. < ttr < 57860. and 0.00 < q1 < 0.02 and 0.00 < q2 < 0.02
            and 0.60 < p_s < 0.80 and 0.10 < b_alb < 0.50 and 0.20 < c_11 < 0.50):
        prior_per = -0.5 * ((pd - 1.80988198) / 0.00000064)**2
        prior_inc = -0.5 * ((incl - 89.623) / 0.034)**2
        prior_rad = -0.5 * ((rad_p - 0.10852) / 0.00096)**2
        prior_a = -0.5 * ((semi_a - 4.08) / 0.06)**2
        prior_offset = -0.5 * ((h_off - np.radians(-3)) / np.radians(10))**2
        return prior_per + prior_inc + prior_rad + prior_a + prior_offset

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