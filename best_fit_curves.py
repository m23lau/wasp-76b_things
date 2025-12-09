# Please excuse the messiness of this file
# Also this is an outdated file used in preliminary testing/fitting, an updated, slightly neater version can be found in kelp_models.py (big_fig function)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from kelp_models import kelp_transit, bin
from statistics import stdev

# Create figure and subplots
fig = plt.figure()
fig.suptitle('White Light Curve + Residuals')
grid = fig.add_gridspec(3, 2, hspace = 0.0, wspace = 0.0, width_ratios = [4, 1], height_ratios = [1, 1, 1])

ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[1, 0], sharex = ax1)
ax3 = fig.add_subplot(grid[2, 0], sharex = ax1)
ax4 = fig.add_subplot(grid[2, 1], sharey = ax3)

# Scatter data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
real_time = data[1]
real_flux = data[2] / data[4]
ax1.scatter(real_time, real_flux, color = 'r' , s = 2.5, alpha = 0.3, label = 'Raw Data')
ax2.scatter(real_time, real_flux, color = 'r' , s = 2.5, alpha = 0.3, label = 'Raw Data')


# Open csv file with parameters and plot each curve
params = pd.read_csv('MCMC Results - all runs.csv')
ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, a_b, c_11 = (params['Transit time'], params['Period'],
                                                               params['Inclination'], params['Radius of Planet'], params['Semimajor Axis'],
                                                               params['q1'], params['q2'], params['Hotspot Offset'],
                                                               params['Bond Albedo'], params['C11'])

# Trials are separated into bad, mid and good depending on the average of their residuals (hindsight: not the greatest metric)
# "Good trials" show a solid, coloured curve, "mid trials" show a translucent yellow curve
# and "bad trials" are omitted and print a spongebob sad noise
good_trials = [1, 2, 6, 10]
mid_trials = [3, 9, 11, 12, 17, 21, 22]


# Loop through all the trials and for each trial, make a phase curve and check how good of a fit it is
for i in range(len(ttr)):
    test_curve = kelp_transit(real_time, t0 = ttr[i], per = pd[i], inc = incl[i], rp = rad_p[i], ecc = 0, w = 51,
                            a = semi_a[i], q = [q1[i], q2[i]], fp = 1.0, T_s=6305.79,
                            rp_a = rad_p[i] / semi_a[i], limb_dark='quadratic', name='WASP-76b', channel=f"IRAC {2}",
                            hotspot_offset = h_off[i], A_B = a_b[i],  c11 = c_11[i])

    if i == len(ttr) - 1:        # The very last line is the average of all the fits
        ax1.plot(test_curve[0], test_curve[1], label = 'Average', color = 'k')
        ax2.plot(test_curve[0], test_curve[1], label = 'Average', color = 'k')
    elif i+1 in mid_trials:
        ax1.plot(test_curve[0], test_curve[1], alpha = 0.3, color = 'y')
        ax2.plot(test_curve[0], test_curve[1], alpha = 0.3, color = 'y')
    elif i+1 in good_trials:
        ax1.plot(test_curve[0], test_curve[1], label = 'Trial #' + str(1 + i) + ', Good Fit')
        ax2.plot(test_curve[0], test_curve[1], label = 'Trial #' + str(1 + i) + ', Good Fit')
    else:
        print('womp womp')      # Bad trials get a womp womp


# Bin data points + plot them on top of everything
bin_t = np.linspace(real_time[0], real_time[-1], 50)
bin_rf = bin(real_time, real_flux, 51)
rf_err = np.array([8.227736082037473e-4 for i in range(len(bin_rf))])         # Largest average residual that was used
ax1.errorbar(bin_t, bin_rf, yerr = rf_err, fmt = '.k',  label = 'Binned Data')
ax2.errorbar(bin_t, bin_rf, yerr = rf_err, fmt = '.k',  label = 'Binned Data')

ax1.set_ylabel('Normalized Flux')
ax1.set_ylim(0.981, 1.006)
ax2.set_ylabel('Normalized Flux')     # Zoomed in, same plot as ax[0]
ax2.set_ylim(0.9981, 1.0045)


# Plot residuals and histogram
# Once again, loop through trials and check how good it was
diffs = []
for i in range(len(ttr)):
    test_curve = kelp_transit(real_time, t0 = ttr[i], per = pd[i], inc = incl[i], rp = rad_p[i], ecc = 0, w = 51,
                            a = semi_a[i], q = [q1[i], q2[i]], fp = 1.0, T_s=6305.79,
                            rp_a = rad_p[i] / semi_a[i], limb_dark='quadratic', name='WASP-76b', channel=f"IRAC {2}",
                            hotspot_offset = h_off[i], A_B = a_b[i],  c11 = c_11[i])
    bin_f = bin(test_curve[0], test_curve[1], len(real_time) + 1)
    tt = np.linspace(test_curve[0][0], test_curve[0][-1], len(real_time))
    diff = real_flux - bin_f
    # l = 'average value of resid for Trial '+ str(i+1) + str(np.average(diff))      # uncomment to see average residual value for each trial
    # diffs.append(l)

    if i == len(ttr) - 1:
        ax3.scatter(tt, diff, s=2, color = 'k', label='Average')
        ax4.hist(diff, bins=50, color = 'k', range=(-0.0026, 0.0026), orientation='horizontal')
    elif i+1 in mid_trials:
        ax3.scatter(tt, diff, s=2, color='y', alpha = 0.3)
        #ax4.hist(diff, bins=50, color='y', range=(-0.0026, 0.0026), orientation='horizontal')
    elif i+1 in good_trials:
        ax3.scatter(tt, diff, s = 2, label = 'Trial #' + str(1 + i) + ', Good Fit')
        ax4.hist(diff, bins=50, range=(-0.0026, 0.0026), orientation='horizontal')
    else:
        print('boowomp')

print(diffs)

ax3.axhline(y = 0, color = 'k', alpha = 0.5, ls = '--')
ax3.set_xlabel('Time, days')
ax3.set_ylabel('Difference in flux, ppm')
ax3.set_ylim(-0.003, 0.003)     # Zoomed in to ignore some outliers
plt.subplots_adjust(hspace = 0.0)

ax4.axhline(y = 0, color = 'k', alpha = 0.5, ls = '--')
ax4.set_xlabel('Frequency (Good fits)')

ax1.legend()
ax3.legend()
plt.show()
