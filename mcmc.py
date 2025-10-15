import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import log_prob
from scipy.optimize import minimize
from emcee import EnsembleSampler
from corner import corner
import pickle
from tqdm import tqdm

# Define data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
ferr = np.abs(flux * (0.00002 * np.random.randn(len(flux))))


# Initial guess
init_p = np.array([57859.31, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.77, 0.24, 0.30])
bounds = [[57855, 57865], [1.8, 1.85], [87.0, 92.0], [0.1, 0.12], [4.0, 4.2], [0.0, 0.02], [0.0, 0.02],
          [-0.1221, 0.0873], [0.6, 0.8], [0.1, 0.5], [0.2, 0.5]]
# soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# Start simulations
n_params = 11
n_walkers = 2 * n_params
p0 = [init_p + 0.01 * np.random.randn(n_params) for i in range(n_walkers)]
sampler = EnsembleSampler(n_walkers, n_params, log_prob, args = (time, flux, ferr))

n_burn, n_steps = 10, 100
p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
sampler.reset()
sampler.run_mcmc(p1, n_steps, progress = True)          # Actual simulation

chain = sampler.get_chain()
guesses = chain.transpose(2, 1, 0)      # Transpose the chain so that each guesses[i] shows each parameter and each
                                        # walker's steps in a row

# Make n subplots, one for each parameter, to see if they converge
xs = np.arange(0, n_steps)
fig, axs = plt.subplots(n_params)
fig.suptitle('Convergence of Walkers')
for i in range(n_walkers):
    for j in range(n_params):
        axs[j].plot(xs, guesses[j][i])

# guesses[i] corresponds to the ith parameter in the list below
p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
                                                  r'$\Delta \phi$', r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']

for k in range(n_params):
    axs[k].set_ylabel(p_labels[k])

axs[10].set_xlabel('Step Number')
plt.show()


# Show corner plot
corner(sampler.flatchain, truths = None, labels = p_labels)
plt.show()


# Print each walker's last step
for i in range(n_params):
    for j in range(n_walkers):
        print(guesses[i][j][-1])
    print()