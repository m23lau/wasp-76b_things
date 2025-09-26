import numpy as np
import matplotlib.pyplot as plt
import kelp_models
from scipy.optimize import minimize
from emcee import EnsembleSampler
from corner import corner
import pickle
from tqdm import tqdm

# Scatter data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
ferr = np.abs(flux * (0.0002 * np.random.randn(len(flux))))
# plt.errorbar(time, flux, yerr = ferr, fmt = '.r', capsize = 0)
# plt.show()

# Create model
def pc_model(params, t):
    """ Model phase curve using kelp_models.kelp_transit
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
    Returns:
         ndarray: Observed flux from planet-star system
    """
    p_s, b_alb, c_11 = params
    new_t, flux = kelp_models.kelp_transit(t, t0 = 56107.3541, per = 1.80988190, inc = 89.623, rp = 0.10852, ecc = 0,
                                    w = 51, a = 4.08, q = [0.01, 0.01], fp = 1.0, t_secondary = 0, T_s = 6305.79,
                                    rp_a = 0.0266, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
                                    hotspot_offset = np.radians(-3), phase_shift = p_s, A_B = b_alb, c11 = c_11)
    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """
    p_s, b_alb, c_11 = params

    if (p_s > np.pi or p_s < -np.pi or b_alb > 1 or b_alb < 0 or c_11 > 1 or c_11 < 0):
        return -np.inf

    else:
        return 0


def log_likelihood(params, t, f, ferr):
    """ Log likelihood, maximum likelihood estimator of parameters
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
        f (ndarray): Flux data points
        ferr (ndarray): Error on flux
    Returns:
        float: Log-likelihood estimate
    """
    return -0.5 * np.sum((pc_model(params, t) - f)**2 / ferr**2)


def log_prob(params, t, f, ferr):
    """ Log probability, the sum of log_likelihood and log_prior
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
        f (ndarray): Flux data points
        ferr (ndarray): Error on flux
    Returns:
        float or -infinity
    """
    lp = log_prior(params)

    if np.isfinite(lp):
        return lp + log_likelihood(params, t, f, ferr)

    else:
        return -np.inf


# Initial guess
init_p = np.array([0.7, 0.2, 0.34])
bounds = [[-np.pi, np.pi], [0.0, 1.0], [0.0, 1.0]]
soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# Start simulations
n_params = 3
n_walkers = 2 * n_params
p0 = [soln.x + 0.1 * np.random.randn(n_params) for i in range(n_walkers)]
sampler = EnsembleSampler(n_walkers, n_params, log_prob, args = (time, flux, ferr))

n_burn, n_steps = 100, 1000
p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
sampler.reset()
sampler.run_mcmc(p1, n_steps, progress = True)          # Actual simulation

chain = sampler.get_chain()
print(chain[-1])            # Print last step of each walker


# See if walkers converge
shift_guesses = np.zeros((n_walkers, n_steps), float)       # These are 2D arrays with n_walkers rows
alb_guesses = np.zeros((n_walkers, n_steps), float)         # and n_steps columns for each paraneter
c_guesses = np.zeros((n_walkers, n_steps), float)

for i in range(n_walkers):
    for j in range(n_steps):
        shift_guesses[i][j] = chain[j][i][0]
        alb_guesses[i][j] = chain[j][i][1]
        c_guesses[i][j] = chain[j][i][2]

for i in range(n_walkers):
    print(shift_guesses[i][-1])
print()
for i in range(n_walkers):
    print(alb_guesses[i][-1])
print()
for i in range(n_walkers):
    print(c_guesses[i][-1])


# Make 3 subplots, one for each parameter
xs = np.arange(0, n_steps)
fig, axs = plt.subplots(3)
fig.suptitle('Convergence of Walkers')
for i in range(6):
    axs[0].plot(xs, shift_guesses[i])
    axs[1].plot(xs, alb_guesses[i])
    axs[2].plot(xs, c_guesses[i])

axs[0].set_ylabel(r'$\Delta \phi$')
axs[1].set_ylabel(r'$A_B$')
axs[2].set_ylabel(r'$C_{11}$')
axs[2].set_xlabel('Step Number')
plt.show()


# Show corner plot
corner(sampler.flatchain, truths = None, labels = [r'$\Delta \phi$', r'$A_B$', r'$C_{11}$'])
plt.show()
