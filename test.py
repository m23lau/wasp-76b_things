from kelp import Filter, Planet, Model
import matplotlib.pyplot as plt
import numpy as np
import astro_models

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

model = Model(hotspot_offset = offset, omega_drag = w_drag, alpha = alpha, C_ml = [[0], [0, c11, 0]], lmax = 1,
              A_B = 0.0, planet = p, filt = filt)

angle = np.linspace(-np.pi, np.pi, 500)
pc = model.phase_curve(angle, omega = 0.0, g = 0.0)
flux = pc[0].flux

plt.plot(angle, flux)
plt.show()

plt.show()

