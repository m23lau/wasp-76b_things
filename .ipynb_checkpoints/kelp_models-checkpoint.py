from kelp import Filter, Planet, Model
import matplotlib.pyplot as plt
import numpy as np
from astro_models import transit_model, eclipse
import math

def kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
    """ Model a phase variation sinusoid (only one period) using kelp
    Args:
        time (ndarray): Number of times to calculate the model
        t0 (float): Transit time
        per (float): Orbital period, days
        inc (float): Orbital inclination, deg
        rp (float): Ratio of planet-to-star radius
        ecc (float): Eccentricity
        w (float): Argument of periastron, deg
        a (float): Semimajor axis normalized by stellar radius
        u (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
        t_sec (float): Time of secondary eclipse
        T_s (float): Temperature of host star, K
        rp_a (float): Radius of planet over semimajor axis
        limb_dark (str): Limb darkening law to use
        name (str): Name of planet
        channel (str): Name of filter
        hotspot_offset (float): Angle of hotspot offset, rad
        A_B (float): Bond albedo
        c11 (float): First spherical harmonic term
    Returns:
        ndarray: Phase variations
    """
    # Create kelp.Planet object and filter
    p = Planet(per, t0, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name)
    filt = Filter.from_name(channel)
    filt.bin_down(bins=10)

    alph = 0.6
    w_drag = 4.5

    # Create model
    model = Model(hotspot_offset = hotspot_offset, omega_drag = w_drag, alpha = alph, C_ml=[[0], [0, c11, 0]],
                  lmax = 1, A_B = A_B, planet = p, filt = filt)

    # Get phase variations
    angle = np.linspace(-np.pi, np.pi, len(time))
    pc = model.phase_curve(angle, omega = 0.0, g = 0.0)
    flux = pc[0].flux / 1e6

    return flux


def phase_offset(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, phase_shift, A_B, c11):
    """Apply a phase shift to a kelp sinusoid and display it for an arbitrary amount of time
    Args:
        time (ndarray): Number of times at which to calculate the model
        t0 (float): Transit time
        per (float): Orbital period, days
        inc (float): Orbital inclination, deg
        rp (float): Ratio of planet-to-star radius
        ecc (float): Eccentricity
        w (float): Argument of periastron, deg
        a (float): Semimajor axis normalized by stellar radius
        u (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
        t_sec (float): Time of secondary eclipse
        T_s (float): Temperature of host star, K
        rp_a (float): Radius of planet over semimajor axis
        limb_dark (str): Limb darkening law to use
        name (str): Name of planet
        channel (str): Name of filter
        hotspot_offset (float): Angle of hotspot offset, rad
        phase_shift (float): Arbitrary phase shift of sinusoid, between -pi and pi rad
        A_B (float): Bond albedo
        c11 (float): First spherical harmonic term
    Returns:
        tuple: ndarray, ndarray
        Times at which to calculate the model, phase variations from planet-star system over time
    """
    # Define time and flux for one period
    flux_1p = kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11)
    time_1p = np.linspace(0, per, len(flux_1p))

    # Shift values by phase_shift, then interpolate to move them back to original places
    t_shifted = (time_1p + phase_shift) % per
    f_interp = np.interp(t_shifted, time_1p, flux_1p)

    # Define the curve for multiple periods
    n_pd = (time.max() - time.min()) / per
    full_pd = int(n_pd)
    frac_pd = n_pd - full_pd

    f_tiled = np.tile(f_interp, int(n_pd))
    if math.isclose(0.0, frac_pd, abs_tol = 1e-3) == False:
        extra_len = round(frac_pd*len(f_interp))
        f_other = f_interp[:extra_len]
        f_new = np.concatenate([f_tiled, f_other])
        t_new = np.linspace(time.min(), time.max(), len(f_new))
        return t_new, f_new

    else:
        t_new = np.linspace(time.min(), time.max(), len(f_tiled))
        return t_new, f_tiled


def kelp_transit(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, phase_shift, A_B, c11):
    """ Model phase variation sinusoid using kelp
    Args:
        time (ndarray): Number of times to calculate the model
        t0 (float): Transit time
        per (float): Orbital period, days
        inc (float): Orbital inclination, deg
        rp (float): Ratio of planet-to-star radius
        ecc (float): Eccentricity
        w (float): Argument of periastron, deg
        a (float): Semimajor axis normalized by stellar radius
        u (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
        t_sec (float): Time of secondary eclipse
        T_s (float): Temperature of host star, K
        rp_a (float): Radius of planet over semimajor axis
        limb_dark (str): Limb darkening law to use
        name (str): Name of planet
        channel (str): Name of filter
        hotspot_offset (float): Angle of hotspot offset, rad
        phase_shift (float): Arbitrary phase shift of sinusoid, between -pi and pi rad
        A_B (float): Bond albedo
        c11 (float): First spherical harmonic term
    Returns:
        tuple: ndarray, ndarray
        Times at which to calculate the model, observed flux from planet-star system over time
    """

    new_time, flux = phase_offset(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name,
                                   channel, hotspot_offset, phase_shift, A_B, c11)

    u1 = 2 * np.sqrt(q[0]) * q[1]
    u2 = np.sqrt(q[0]) * (1 - 2 * q[1])

    transit, t_sec, anom = transit_model(new_time, t0, per, rp, a, inc, ecc, w, u1, u2)
    eclip = eclipse(new_time, t0, per, rp, a, inc, ecc, w, fp = fp, t_sec = t_sec)

    ff = flux * (eclip - 1) + transit
    return new_time, ff


def bin(x, y, nbins):
    """Bin values
    Args:
        x (ndarray): Values of the axis along which binning will occur
        y (ndarray): Array of values to bin
        nbins (int): The number of bins desired
    Returns:
        ndarray: Binned values
    """
    bins = np.linspace(x.min(), x.max(), nbins)
    digitized = np.digitize(x, bins)
    binned = np.array([np.nanmedian(y[digitized == i]) for i in range(1, nbins)])

    return binned

# binned = binValues(y, x, nbins)
