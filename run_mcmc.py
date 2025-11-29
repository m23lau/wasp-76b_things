import numpy as np
from mcmc_funcs_121 import pc_model, log_likelihood, log_prob
from mcmc import run_sims


# Define data points for WASP-121b
data = np.load('WASP-121b-exoTEDRF-Order1-R100-detrended-flux.npz')
time = data['time']
wavelengths = data['wavelength']
fluxes = data['detrended_flux']/1e6 + 1
flux_errs = data['detrended_flux_err']/1e6

init_p = np.array([0.2, 0.0, 0.2, 0.5, 1.0])
p_labels = [r'$r_p$', r'$\Delta \phi$', '$A_B$','$C_{11}$', 'Unc']
explore_scale = np.array([0.1, 2.0, 0.2, 0.2, 0.1])


for i in range(1, len(wavelengths)):
    bt, gs, fc = run_sims(init_p, explore_scale, pc_model, log_likelihood, log_prob, time, fluxes[i], flux_errs[i],
                          1000, 2000, 1, p_labels, title=wavelengths[i])