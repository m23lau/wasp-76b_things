from kelp import Filter, Planet, Model
import matplotlib.pyplot as plt
import numpy as np
from astro_models import transit_model, eclipse

# Parameters
t0   = 2458080.626165
per  = 1.80988198
rp   = 0.10852
a    = 4.08
inc  = 89.623
ecc = 0
w = 51
q = [0.01, 0.01]
fp   = 0.003687
t_secondary = 0.0
T_s = 6305.79
rp_a = rp / a
limb_dark = 'Quadratic'
name = 'WASP-76b'
channel = f"IRAC {2}"
hotspot_offset = np.radians(-3)
A_B = 0.0
c11 = 0.18

def kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
    """ Model phase variation sinusoid using kelp
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

    offset = -0.05236  # converted from -3 degrees E from "A Comprehensive Analysis of Spitzer 4.5 μm..."
    alpha = 0.6
    w_drag = 4.5
    c11 = 0.18         # alpha, omega_drag parameters are same as sample on the documentation

    # Create model
    model = Model(hotspot_offset = hotspot_offset, omega_drag = 4.5, alpha = 0.6, C_ml=[[0], [0, c11, 0]],
                  lmax = 1, A_B = A_B, planet = p, filt = filt)

    # Get phase variations for n periods
    angle = np.linspace(-np.pi, np.pi, len(time))
    pc = model.phase_curve(angle, omega = 0.0, g = 0.0)
    flux = pc[0].flux / 1e6

    n_periods = (time[-1] - time[0]) / per
    full_pd = int(n_periods)
    frac_pd = n_periods - full_pd

    for i in range(full_pd - 1):
        ff = pc[0].flux / 1e6
        flux = np.concatenate([ff, flux])

    if frac_pd != 0:
        extra_len = round(frac_pd * len(time))
        extra_angle = np.linspace(angle[0], angle[extra_len], extra_len)
        extra_flux = model.phase_curve(extra_angle, omega = 0.0, g = 0.0)[0].flux / 1e6
        flux = np.concatenate([flux, extra_flux])
        new_time = np.linspace(time[0], time[-1], len(flux))
        return new_time, flux

    else:
        new_time = np.linspace(time[0], time[-1], len(flux))
        return new_time, flux

time = np.linspace(0, 2*per, 1000)
flux = kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11)
plt.plot(flux[0], flux[1])
plt.show()



# def kelp_transit(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
#     """ Model phase variation sinusoid using kelp
#     time (ndarray): Number of times to calculate the model
#     t0 (float): Transit time
#     per (float): Orbital period, days
#     inc (float): Orbital inclination, deg
#     rp (float): Ratio of planet-to-star radius
#     ecc (float): Eccentricity
#     w (float): Argument of periastron, deg
#     a (float): Semimajor axis normalized by stellar radius
#     u (list): List of quadratic limb-darkening parameters
#     fp (float): Planetary flux out of eclipse
#     t_sec (float): Time of secondary eclipse
#     T_s (float): Temperature of host star, K
#     rp_a (float): Radius of planet over semimajor axis
#     limb_dark (str): Limb darkening law to use
#     name (str): Name of planet
#     channel (str): Name of filter
#     hotspot_offset (float): Angle of hotspot offset, rad
#     A_B (float): Bond albedo
#     c11 (float): First spherical harmonic term
#     Returns:
#         tuple: ndarra, ndarray
#         Times at which to calculate the model, observed flux from planet-star system over time
#
#     """
#
#     flux = kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, t_secondary, T_s, rp_a, limb_dark, name,
#                                    channel, n_periods, hotspot_offset, A_B, c11)
#
#     time_new = np.linspace(time[0], time[-1], len(flux))  # Re-define time to have the same length as flux due to np.concatenate
#
#     u1 = 2 * np.sqrt(q[0]) * q[1]
#     u2 = np.sqrt(q[0]) * (1 - 2 * q[1])
#
#     transit, t_sec, anom = transit_model(time_new, t0, per, rp, a, inc, ecc, w, u1, u2)
#     eclip = eclipse(time_new, t0, per, rp, a, inc, ecc, w, fp = 1.0, t_sec = t_sec)
#
#     ff = flux * (eclip - 1) + transit
#     return time_new, ff
#
