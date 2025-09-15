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
time1 = np.linspace(0, 2*per, 1000) #Array of times at which to calculate the model.
norm_flux = astro_models.ideal_lightcurve(time1, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B)


# kelp transit
kelp = kelp_models.kelp_transit(time1, t0, per, inc, rp, ecc = 0, w = 51, a = a, q = [q1, q2], fp = fp, t_secondary = 0,
                                T_s = 6305.79, rp_a = 0.0266, limb_dark = 'Quadratic', name = 'WASP-76b',
                                channel = f"IRAC {2}", hotspot_offset = np.radians(-3), A_B = 0.0, c11 = 0.18)



plt.plot(time1, norm_flux, color = 'k', label = 'batman')
plt.plot(kelp[0], kelp[1], color = 'g', label = 'kelp')
plt.legend()
plt.show()
