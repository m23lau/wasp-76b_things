from astro_models import kelp_transit
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Raw data
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)

real_time = data[1]    # I need to shift the data over
real_flux = data[2] / data[4]           # Calibrated flux


# Parameters
t0   = 2458080.626165 #transit time
per  = 1.80988198     #Orbital period.
rp   = 0.10852        #Planet radius (in units of stellar radii).
a    = 4.08           #Semi-major axis (in units of stellar radii).
inc  = 89.623         #Orbital inclination (in degrees).
q1   = 0.01          #Limb darkening coefficient 1, parametrized to range between 0 and 1.
q2   = 0.01          #Limb darkening coefficient 2, parametrized to range between 0 and 1.
fp   = 0.003687      #Planet-to-star flux ratio. NOT LISTED, so I took the average from a table
ecc = 0
w = 51
t_secondary = 0
T_s = 6305.79
rp_a = rp / a
limb_dark = 'Quadratic'
name = 'WASP-76b'
channel = f"IRAC {2}"
n_periods = 1
hotspot_offset = np.radians(3)
A_B = 0.0
c11 =  0.18

time = np.linspace(0, per, len(real_time))
kelp = kelp_transit(time, t0, per, inc, rp, ecc, w, a, [q1, q2], fp, t_secondary,
                                T_s, rp_a, limb_dark, name, channel, n_periods, hotspot_offset=0.89, A_B=0.2, c11=0.336)

plt.scatter(real_time, real_flux, s = 3, color = 'r', label = 'data')
plt.plot(kelp[0]+real_time, kelp[1], color = 'g', label = 'model')
plt.legend()
plt.show()

print(real_time[0])