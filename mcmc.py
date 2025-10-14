import numpy as np
import matplotlib.pyplot as plt
import kelp_models
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


# Create model
def pc_model(params, t):
    """ Model phase curve using kelp_models.kelp_transit
    Args:
        params: Parameters to optimize
        t (ndarray): Times to calculate the model
    Returns:
         ndarray: Observed flux from planet-star system
    """
    ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, p_s, b_alb, c_11 = params
    new_t, flux = kelp_models.kelp_transit(t, t0 = ttr, per = pd, inc = incl, rp = rad_p, ecc = 0,
                                    w = 51, a = semi_a, q = [q1, q2], fp = 1.0, t_secondary = 0, T_s = 6305.79,
                                    rp_a = rad_p / semi_a, limb_dark = 'quadratic', name = 'WASP-76b', channel = f"IRAC {2}",
                                    hotspot_offset = h_off, phase_shift = p_s, A_B = b_alb, c11 = c_11)
    f_binned = kelp_models.bin(new_t, flux, len(t) + 1)
    return f_binned


def log_prior(params):
    """ Log prior, assumed probability distribution
    Args:
        params: Parameters to optimize
    Returns:
        -infinity or 0
    """
    ttr, pd, incl, rad_p, semi_a, q1, q2, h_off, p_s, b_alb, c_11 = params

    if (57859. < ttr < 57860. and 0.00 < q1 < 0.02 and 0.00 < q2 < 0.02
            and 0.60 < p_s < 0.80 and 0.10 < b_alb < 0.50 and 0.20 < c_11 < 0.50):
        prior_per = -0.5 * ((pd - 1.80988198) / 0.00000064)**2
        prior_inc = -0.5 * ((incl - 89.623) / 0.034)**2
        prior_rad = -0.5 * ((rad_p - 0.10852) / 0.00096)**2
        prior_a = -0.5 * ((semi_a - 4.08) / 0.06)**2
        prior_offset = -0.5 * ((h_off - np.radians(-3)) / np.radians(10))**2
        return prior_per + prior_inc + prior_rad + prior_a + prior_offset

    else:
        return -np.inf


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
init_p = np.array([57859.31, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.77, 0.24, 0.30])
bounds = [[57855, 57865], [1.8, 1.85], [87.0, 92.0], [0.1, 0.12], [4.0, 4.2], [0.0, 0.02], [0.0, 0.02],
          [-0.1221, 0.0873], [0.6, 0.8], [0.1, 0.5], [0.2, 0.5]]
soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# Start simulations
n_params = 11
n_walkers = 2 * n_params
p0 = [init_p + 0.01 * np.random.randn(n_params) for i in range(n_walkers)]
sampler = EnsembleSampler(n_walkers, n_params, log_prob, args = (time, flux, ferr))

n_burn, n_steps = 100, 1000
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

p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
                                                  r'$\Delta \phi$', r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']
# guesses[i] corresponds to the ith parameter in the list above

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