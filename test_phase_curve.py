import astro_models
import numpy as np
import matplotlib.pyplot as plt
import kelp_models

# Parameters
t0   = 2458080.626165 #transit time
per  = 1.80988198     #Orbital period.
rp   = 0.10852        #Planet radius (in units of stellar radii).
a    = 4.08           #Semi-major axis (in units of stellar radii).
inc  = 89.623         #Orbital inclination (in degrees).
ecosw= 0.0           #Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
esinw= 0.0           #Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
q1   = 0.01          #Limb darkening coefficient 1, parametrized to range between 0 and 1.
q2   = 0.01          #Limb darkening coefficient 2, parametrized to range between 0 and 1.
fp   = 0.003687      #Planet-to-star flux ratio. NOT LISTED, so I took the average from a table
A    = 0.4           #Amplitude of the first-order cosine term.
B    = 0.0001        #Amplitude of the first-order sine term.
C    = 0.0           #(optional): Amplitude of the second-order cosine term. Default=0.
D    = 0.0           #(optional): Amplitude of the second-order sine term. Default=0.
r2   = None          #(optional): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
r2off= None          #(optional): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.


# astro models
#time1 = np.linspace(t0 - 0.7*per, t0 + 0.7*per, 10000) #Array of times at which to calculate the model.
#norm_flux = astro_models.ideal_lightcurve(time1, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B)

# Data
import pickle

with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)

time = data[1]
flux = data[2] / data[4]
plt.scatter(time, flux, color = 'r', s = 2, label = 'data')

# kelp phase curve
phase_shift = np.radians(42)

kelpt = kelp_models.kelp_transit(time, t0 = 57859.2985794537, per = per, inc = inc, rp = rp, ecc = 0, w = 51, a = a, q = [q1, q2], fp = 1.0, t_secondary = 0,
                                T_s = 6305.79, rp_a = 0.0266, limb_dark = 'quadratic', name = 'WASP-76b',
                                channel = f"IRAC {2}", hotspot_offset = np.radians(-3), phase_shift = phase_shift, A_B = 0.2, c11 = 0.19)

#plt.plot(time1, norm_flux, color = 'k', label = 'batman')
plt.plot(kelpt[0], kelpt[1], color = 'g', label = 'kelp transit')

plt.legend()
plt.show()

# Residuals + Bin data for comparison
from kelp_models import bin
calibrated_binned = bin(kelpt[0], kelpt[1], len(time)+1)
tt = np.linspace(kelpt[0][0], kelpt[0][-1], len(time))
diff = flux - calibrated_binned
plt.scatter(tt, diff, s = 2)
plt.show()