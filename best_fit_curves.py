import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from kelp_models import kelp_transit, bin

# Scatter data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
real_time = data[1]
real_flux = data[2] / data[4]
ferr = np.abs(real_flux * (0.0002 * np.random.randn(len(real_flux))))
plt.scatter(real_time, real_flux, color = 'r', s = 5, alpha = 0.3, label = 'Raw Data')


# Open csv file with parameters and plot each curve
params = pd.read_csv('MCMC Results - Sheet1.csv')
p_s, a_b, c_11 = params['Phase Shift'], params['Bond Albedo'], params['C11']

for i in range(len(p_s)):
    if i == 1 or i == 11:
        print('womp womp')
    else:
        test_curve = kelp_transit(real_time, t0 = 56107.3541, per = 1.80988198, inc = 89.623, rp = 0.10852, ecc = 0, w = 51,
                             a = 4.08, q = [0.01, 0.01], fp = 1.0, t_secondary = 0, T_s=6305.79, rp_a = 0.0266,
                             limb_dark='quadratic', name='WASP-76b', channel=f"IRAC {2}",
                             hotspot_offset = np.radians(-3), phase_shift = p_s[i], A_B = a_b[i],  c11 = c_11[i])
        plt.plot(test_curve[0], test_curve[1], label = 'Trial #' + str(1 + i))


# Bin data points
bin_t = np.linspace(real_time[0], real_time[-1], 50)
bin_rf = bin(real_time, real_flux, 51)
plt.scatter(bin_t, bin_rf, color = 'k',  label = 'Binned Data')

plt.title('Light Curve')
plt.xlabel('Time, days')
plt.ylabel('Normalized flux, ppm')
plt.legend()
plt.show()


# Plot residuals
res_list = []
for i in range(len(p_s)):
    if i == 1 or i == 11:
        print('boowomp')
    else:
        test_curve = kelp_transit(real_time, t0=56107.3541, per=1.80988198, inc=89.623, rp=0.10852, ecc=0, w=51,
                                  a=4.08, q=[0.01, 0.01], fp=1.0, t_secondary=0, T_s=6305.79, rp_a=0.0266,
                                  limb_dark='quadratic', name='WASP-76b', channel=f"IRAC {2}",
                                  hotspot_offset=np.radians(-3), phase_shift=p_s[i], A_B=a_b[i], c11=c_11[i])
        bin_f = bin(test_curve[0], test_curve[1], len(real_time) + 1)
        tt = np.linspace(test_curve[0][0], test_curve[0][-1], len(real_time))
        diff = real_flux - bin_f
        plt.scatter(tt, diff, s = 2, label = 'Trial #' + str(1 + i))
        res_list.append(np.abs(np.average(diff)))

plt.axhline(y = 0, color = 'k', alpha = 0.5, ls = '--')
plt.title('Residuals')
plt.xlabel('Time, days')
plt.ylabel('Difference in flux, ppm')
plt.legend()
plt.show()
print(np.array(res_list))