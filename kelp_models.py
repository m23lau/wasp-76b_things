from kelp_w_jwst import Filter, Planet, Model
import matplotlib.pyplot as plt
import numpy as np
from astro_models import transit_model, eclipse
import math

def kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
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
        q (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
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
    p = Planet(per, t0, inc, rp, ecc, w, a, q, fp, t0 - 0.5*per, T_s, rp_a, limb_dark, name)    # Time of secondary eclipse is
                                                                                                # half a period from transit
    filt = Filter.from_name(channel)
    filt.bin_down(bins=10)

    alph = 0.6
    w_drag = 4.5

    # Create model
    model = Model(hotspot_offset = hotspot_offset, omega_drag = w_drag, alpha = alph, C_ml=[[0], [0, c11, 0]],
                  lmax = 1, A_B = A_B, planet = p, filt = filt)

    # Get phase variations
    angle = np.linspace(-np.pi, np.pi, len(time))       # xi values (phase angles)
    pc = model.phase_curve(angle, omega = 0.0, g = 0.0)
    flux = pc[0].flux / 1e6

    return flux


def phase_offset(time, t0, per, inc, rp, ecc, w, a, q, fp, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
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
        q (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
        T_s (float): Temperature of host star, K
        rp_a (float): Radius of planet over semimajor axis
        limb_dark (str): Limb darkening law to use
        name (str): Name of planet
        channel (str): Name of filter
        hotspot_offset (float): Angle of hotspot offset, rad
        A_B (float): Bond albedo
        c11 (float): First spherical harmonic term
    Returns:
        tuple: ndarray, ndarray
        Times at which to calculate the model, phase variations from planet-star system over time
    """
    # Define flux for one period
    flux_1p = kelp_curve(time, t0, per, inc, rp, ecc, w, a, q, fp, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11)

    # Shift values by checking phase of planet at time[0], then interpolate & move the curve back to original spot
    t_sec = t0 - 0.5*per
    start_phase = np.abs(t_sec - time[0]) / per                         # Check how far planet is from an eclipse
    start_loc = int(len(time)/2 - start_phase*len(time))                # Gives start location of new phase curve on the kelp curve
    f_shifted = np.concatenate([flux_1p[start_loc:], flux_1p[:start_loc]])

    # Define the curve for multiple periods
    n_pd = (time.max() - time.min()) / per
    full_pd = int(n_pd)
    frac_pd = n_pd - full_pd

    f_tiled = np.tile(f_shifted, int(n_pd))
    if not math.isclose(0.0, frac_pd, abs_tol=1e-3):
        extra_len = round(frac_pd * len(f_shifted))
        f_other = f_shifted[:extra_len]
        f_new = np.concatenate([f_tiled, f_other])
        t_new = np.linspace(time.min(), time.max(), len(f_new))
        return t_new, f_new

    else:
        t_new = np.linspace(time.min(), time.max(), len(f_tiled))
        return t_new, f_tiled


def kelp_transit(time, t0, per, inc, rp, ecc, w, a, q, fp, T_s, rp_a, limb_dark, name, channel, hotspot_offset, A_B, c11):
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
        q (list): List of quadratic limb-darkening parameters
        fp (float): Planetary flux out of eclipse
        T_s (float): Temperature of host star, K
        rp_a (float): Radius of planet over semimajor axis
        limb_dark (str): Limb darkening law to use
        name (str): Name of planet
        channel (str): Name of filter
        hotspot_offset (float): Angle of hotspot offset, rad
        A_B (float): Bond albedo
        c11 (float): First spherical harmonic term
    Returns:
        tuple: ndarray, ndarray
        Times at which to calculate the model, observed flux from planet-star system over time
    """

    new_time, flux = phase_offset(time, t0, per, inc, rp, ecc, w, a, q, fp, T_s, rp_a, limb_dark, name,
                                   channel, hotspot_offset, A_B, c11)

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


def big_fig(t, f, ferr, model, best_fitp, median_sigmap, title):
    """ Plot phase curves and residuals
    Args:
        t (ndarray): Time values of data points
        f (ndarray): Flux values of data points
        ferr (ndarray): Uncertainty in f values of data points
        model (callable): Phase curve model to use
        best_fitp: Best fit parameters
        median_sigmap(ndarray): ndim x 3 array containing fit params for +/-1 sigma and median
        title (str or float): Word or number to be included in plot title
    Returns:
        ndarray: y values for best fit curve
    """
    if type(title) == np.float64:
        title = round(title, 3)

    f_fit = model(best_fitp, t)     # Best fit phase curve based on max log likelihood
    msigma = model(median_sigmap[0], t)    # -1 sigma
    med_fit = model(median_sigmap[1], t)   # 50th percentile fit
    psigma = model(median_sigmap[2], t)    # +1 sigma

    bin_x = np.linspace(t[0], t[-1], 50)
    bin_y = bin(t, f, 51)
    bin_yerr = bin(t, ferr, 51)

    fig = plt.figure()
    fig.suptitle(str(title)+ r'$\mu$' 'm Phase Curve + Residuals')
    grid = fig.add_gridspec(3, 2, hspace=0.0, wspace=0.0, width_ratios=[4, 1], height_ratios=[1, 1, 1])

    # Zoomed out phase curve
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.scatter(t, f, color = 'r' , s = 1, alpha = 0.3, label = 'Raw Data')
    ax1.errorbar(bin_x, bin_y, yerr=bin_yerr, fmt = '.k',  label = 'Binned Data', alpha=0.5)
    ax1.plot(t, f_fit, color='k', label = 'Best Fit', lw = 1)
    ax1.plot(t, med_fit, color='b', label = 'Median Fit', lw = 1)
    ax1.plot(t, msigma, color='g', label = r'$-\sigma$', lw = 1, alpha=0.5)
    ax1.plot(t, psigma, color='y', label = r'$+\sigma$', lw = 1, alpha=0.5)

    ax1.set_ylabel('Normalized Flux')
    ax1.legend(fontsize='x-small', loc=4)

    # Zoomed in phase curve
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    ax2.scatter(t, f, color='r', s=1, alpha=0.3, label='Raw Data')
    ax2.errorbar(bin_x, bin_y, yerr=bin_yerr, fmt='.k', label='Binned Data', alpha=0.5)
    ax2.plot(t, f_fit, color='k', label='Best Fit', lw = 1)
    ax2.plot(t, med_fit, color='b', label = 'Median Fit', lw = 1)
    ax2.plot(t, msigma, color='g', label = r'$-\sigma$', lw = 1, alpha=0.5)
    ax2.plot(t, psigma, color='y', label = r'$+\sigma$', lw = 1, alpha=0.5)

    ax2.axhline(y=1, color='k', alpha = 0.5, ls = '--')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_ylim(f[int(len(f)/2.1463)], max(f))

    # Residuals
    ax3 = fig.add_subplot(grid[2, 0], sharex=ax1)
    diff = f - f_fit
    ax3.scatter(t, diff, s=1, color='g', alpha=0.3)
    diff_med = f - med_fit
    ax3.scatter(t, diff_med, s=1, color='y', alpha = 0.3)

    ax3.axhline(y=0, color='k', alpha=0.5, ls='--')
    ax3.set_xlabel('Time [MJD]')
    ax3.set_ylabel('Difference in flux')

    # Residuals histogram
    ax4 = fig.add_subplot(grid[2, 1], sharey=ax3)
    ax4.hist(diff, bins=50, orientation='horizontal')
    ax4.axhline(y=0, color='k', alpha=0.5, ls='--')
    ax4.set_xlabel('Frequency')

    plt.subplots_adjust(hspace=0.0)
    plt.savefig(str(title) + '_pc.pdf')
    plt.close()
    return f_fit, med_fit, msigma, psigma