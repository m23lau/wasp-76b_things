from kelp import Filter, Planet, Model
import matplotlib.pyplot as plt
import numpy as np
import astro_models
import pickle

# Create the planet WASP-76b since it isn't in the json file
# Planetary flux out of eclipse, Time of secondary eclipse (fp, t_secondary) are left as 1.0 and 0 like all other planets in the json
# because they aren't listed anywhere that I can see
# u is converted from q --> u1 = 2*sqrt(q1) * q2, u2 = sqrt(q1) * (1 - 2*q2)
p = Planet(per = 1.80988198, t0 = 2458080.626165, inc = 89.623, rp = 0.10852 , ecc = 0, w = 51, a = 4.08, u = [0.002, 0.098],
           fp = 1.0, t_secondary = 0, T_s = 6305.79, rp_a = 0.0266, limb_dark = 'quadratic', name = 'WASP-76b')

offset = -0.05236  # converted from -3 degrees E from "A Comprehensive Analysis of Spitzer 4.5 μm..."
alpha = 0.6
w_drag = 4.5
c11 = 0.18                 # alpha, omega_drag parameters are same as sample on the documentation

filt = Filter.from_name(f"IRAC {2}")
filt.bin_down(bins = 10)
C_ml = [[0], [0, c11, 0]]

model = Model(hotspot_offset = 0.89, omega_drag = w_drag, alpha = alpha, C_ml = [[0], [0, 0.336, 0]], lmax = 1,
              A_B = 0.2, planet = p, filt = filt)           # changed hotspot offset, C_ml and A_B (from 0-->0.2)
                                                            #           Phase, Amplitude, Vertical Displacement

angle = np.linspace(-np.pi, np.pi, 500)
pc = model.phase_curve(angle, omega = 0.0, g = 0.0)     # can't find omega, g (scattering albedo, asymmetry factor)
flux1 = flux2 = pc[0].flux                              # show 2 full orbits' flux
flux = np.concatenate([flux1, flux2])/1000000           # divide by 1000000 since flux is in ppm

times = np.linspace(0, 2*p.per, 1000)         # 2 full orbits
transit, t_sec, anom = astro_models.transit_model(times, p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, 0.02, 0.098)
eclip = astro_models.eclipse(times, p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.fp, t_sec = t_sec)
flux = (flux * (eclip - 1)) + transit

# Create batman's transit model curve
norm_flux = astro_models.ideal_lightcurve(times, p.t0, p.per, p.rp, p.a, p.inc, ecosw = 0, esinw = 0, q1 = 0.01,
                                          q2 = 0.01, fp = 0.003687, A = 0.4, B = 0.0001)

# Import and scatter Spitzer data
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
real_time = data[1] - data[1][0] + 1  # I shifted the data points by 1 day to line up the transits
real_flux = data[2]

# Bin data by defining a fxn binValues (copied from notebook that was sent)
def binValues(values, binAxisValues, nbin, assumeWhiteNoise=False):
    """Bin values and compute their binned noise.
    Args:
        values (ndarray): An array of values to bin.
        binAxisValues (ndarray): Values of the axis along which binning will occur.
        nbin (int): The number of bins desired.
        assumeWhiteNoise (bool, optional): Divide binned noise by sqrt(nbinned) (True) or not (False, default).
    Returns:
        tuple: binned (ndarray; the binned values),
            binnedErr (ndarray; the binned errors)
    """
    bins = np.linspace(np.nanmin(binAxisValues), np.nanmax(binAxisValues), nbin)
    digitized = np.digitize(binAxisValues, bins)
    binned = np.array([np.nanmedian(values[digitized == i]) for i in range(1, nbin)])
    binnedErr = np.nanmean(np.array([np.nanstd(values[digitized == i]) for i in range(1, nbin)]))
    if assumeWhiteNoise:
        binnedErr /= np.sqrt(len(values)/nbin)
    return binned, binnedErr
calibrated_binned, calibrated_binned_err = binValues(data[2] / data[4], data[1], 50, assumeWhiteNoise=True)
#
# plt.plot(times, norm_flux, color = 'k', label = 'batman')
# plt.plot(times, flux, color = 'g', label = 'kelp')
# # plt.plot((data[1] - data[1][0] + 1), data[3], color = 'y', label = 'Corrected Lightcurve from Spitzer')
# plt.xlabel('Time (days)')
# plt.ylabel('$F_p/F_s$')
# plt.scatter(real_time, real_flux, s = 2, color = 'r', label = 'Spitzer, Raw Data')
# plt.scatter(real_time[::29], calibrated_binned, s = 3, color = 'b', label = 'Spitzer, Calibrated + Binned Points')
# plt.axhline(y = 1, color = 'grey', ls = '--', alpha = 0.3)
# plt.legend()
# plt.show()

# Residuals Plot
# plt.scatter(times, (norm_flux - flux))
# plt.ylim(-0.0005, 0.0005)
# plt.title('Residuals')
# plt.xlabel('Time (days)')
# plt.ylabel('batman - kelp')
# plt.show()
