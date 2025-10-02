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

    if (ttr < 57855 or ttr > 57865 or pd < 1.80 or pd > 1.85 or incl < 87.0 or incl > 92.0 or rad_p < 0.1 or rad_p > 0.12 or
        semi_a < 4.0 or semi_a > 4.2 or q1 < 0.0 or q1 > 0.02 or q2 < 0.0 or q2 > 0.02 or h_off < np.radians(-7) or
        h_off > np.radians(5) or p_s < 0.6 or p_s > 0.8 or b_alb < 0.1 or b_alb > 0.5 or c_11 < 0.2 or c_11 > 0.5):
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
init_p = np.array([57859, 1.83, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.77, 0.24, 0.30])
bounds = [[57855, 57865], [1.8, 1.85], [87.0, 92.0], [0.1, 0.12], [4.0, 4.2], [0.0, 0.02], [0.0, 0.02],
          [-0.1221, 0.0873], [0.6, 0.8], [0.1, 0.5], [0.2, 0.5]]
soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# Start simulations
n_params = 11
n_walkers = 2 * n_params
p0 = [soln.x + 0.01 * np.random.randn(n_params) for i in range(n_walkers)]
sampler = EnsembleSampler(n_walkers, n_params, log_prob, args = (time, flux, ferr))

n_burn, n_steps = 100, 1500
p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
sampler.reset()
sampler.run_mcmc(p1, n_steps, progress = True)          # Actual simulation

chain = sampler.get_chain()
# print(chain[-1])            # Print last step of each walker


# See if walkers converge
transit_guesses = np.zeros((n_walkers, n_steps), float)     # These are 2D arrays with n_walkers rows
period_guesses = np.zeros((n_walkers, n_steps), float)      # and n_steps columns for each paraneter
incline_guesses = np.zeros((n_walkers, n_steps), float)
radius_guesses = np.zeros((n_walkers, n_steps), float)
semiaxis_guesses = np.zeros((n_walkers, n_steps), float)
q1_guesses = np.zeros((n_walkers, n_steps), float)
q2_guesses = np.zeros((n_walkers, n_steps), float)
offset_guesses = np.zeros((n_walkers, n_steps), float)
shift_guesses = np.zeros((n_walkers, n_steps), float)
alb_guesses = np.zeros((n_walkers, n_steps), float)
c_guesses = np.zeros((n_walkers, n_steps), float)

for i in range(n_walkers):
    for j in range(n_steps):
        transit_guesses[i][j] = chain[j][i][0]
        period_guesses[i][j] = chain[j][i][1]
        incline_guesses[i][j] = chain[j][i][2]
        radius_guesses[i][j] = chain[j][i][3]
        semiaxis_guesses[i][j] = chain[j][i][4]
        q1_guesses[i][j] = chain[j][i][5]
        q2_guesses[i][j] = chain[j][i][6]
        offset_guesses[i][j] = chain[j][i][7]
        shift_guesses[i][j] = chain[j][i][8]
        alb_guesses[i][j] = chain[j][i][9]
        c_guesses[i][j] = chain[j][i][10]

# Make n subplots, one for each parameter
xs = np.arange(0, n_steps)
fig, axs = plt.subplots(n_params)
fig.suptitle('Convergence of Walkers')
for i in range(6):
    axs[0].plot(xs, transit_guesses[i])
    axs[1].plot(xs, period_guesses[i])
    axs[2].plot(xs, incline_guesses[i])
    axs[3].plot(xs, radius_guesses[i])
    axs[4].plot(xs, semiaxis_guesses[i])
    axs[5].plot(xs, q1_guesses[i])
    axs[6].plot(xs, q2_guesses[i])
    axs[7].plot(xs, offset_guesses[i])
    axs[8].plot(xs, shift_guesses[i])
    axs[9].plot(xs, alb_guesses[i])
    axs[10].plot(xs, c_guesses[i])

axs[0].set_ylabel(r'$t_0$')
axs[1].set_ylabel('Per')
axs[2].set_ylabel('Inc')
axs[3].set_ylabel(r'$r_p$')
axs[4].set_ylabel('a')
axs[5].set_ylabel('q1')
axs[6].set_ylabel('q2')
axs[7].set_ylabel(r'$\Delta \phi$')
axs[8].set_ylabel(r'$\Delta \phi$')
axs[9].set_ylabel(r'$A_B$')
axs[10].set_ylabel(r'$C_{11}$')
axs[10].set_xlabel('Step Number')
plt.show()


# Show corner plot
# corner(sampler.flatchain, truths = None, labels = [r'$t_0', 'Per', 'Inc', r'$r_p', 'a', 'q1', 'q2',
#                                                   r'$\Delta \phi$', r'$\Delta \phi$', r'$A_B$', r'$C_{11}$'])
# plt.show()


# Print each walker's last step
for i in range(n_walkers):
    print(transit_guesses[i][-1])
print()
for i in range(n_walkers):
    print(period_guesses[i][-1])
print()
for i in range(n_walkers):
    print(incline_guesses[i][-1])
print()
for i in range(n_walkers):
    print(radius_guesses[i][-1])
print()
for i in range(n_walkers):
    print(semiaxis_guesses[i][-1])
print()
for i in range(n_walkers):
    print(q1_guesses[i][-1])
print()
for i in range(n_walkers):
    print(q2_guesses[i][-1])
print()
for i in range(n_walkers):
    print(offset_guesses[i][-1])
print()
for i in range(n_walkers):
    print(shift_guesses[i][-1])
print()
for i in range(n_walkers):
    print(alb_guesses[i][-1])
print()
for i in range(n_walkers):
    print(c_guesses[i][-1])
