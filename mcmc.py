import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import log_prob
# from scipy.optimize import minimize
from emcee import EnsembleSampler
from corner import corner
import pickle

# Define data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
ferr = np.abs(flux * (0.00002 * np.random.randn(len(flux))))


# Initial guess
init_p = np.array([57859.31, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.24, 0.30])
bounds = [[57855, 57865], [1.8, 1.85], [87.0, 92.0], [0.1, 0.12], [4.0, 4.2], [0.0, 0.02], [0.0, 0.02],
          [-0.1221, 0.0873], [0.1, 0.5], [0.2, 0.5]]
# soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# later on, guesses[i] corresponds to the ith parameter in the list below
p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
            r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']

# Start simulations
def run_sims(n_dim, init, x, y, yerr, n_burn, n_steps, n_runs, labels):
    """ Run MCMC n_runs times
    Args:
        n_dim (int): Number of dimensions/parameters
        init (ndarray): List of initial guesses
        x (ndarray): x-values of data points
        y (ndarray): y-values of data points
        yerr (ndarray): Error in y-values of data points
        n_burn (int): Number of burn-in steps
        n_steps (int): Number of steps in each MCMC run
        n_runs (int): Number of times to run MCMC
        labels (list: str): List of labels for corner plot
    Returns:
         tuple: ndarray, ndarray, ndarray
         last steps of each run, guesses for the last run, flatchain used in corner plot
    """
    last_steps = []

    n_walkers = 2 * n_dim
    p0 = [init + 0.01 * np.random.randn(n_dim) for _ in range(n_walkers)]
    sampler = EnsembleSampler(n_walkers, n_dim, log_prob, args = (x, y, yerr))

    while len(last_steps) < n_runs:
        p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
        sampler.reset()
        sampler.run_mcmc(p1, n_steps, progress = True)          # Actual simulation

        chain = sampler.get_chain()


        # Set up next run with last step of previous run
        new_init = np.average(chain[-1], axis=0)
        last_steps.append(new_init)
        p0 = [new_init + 0.01 * np.random.randn(n_dim) for _ in range(n_walkers)]

    last_chain = sampler.get_chain()
    gs = last_chain.transpose(2, 1, 0)           # Transpose the chain so that each guesses[i] shows each parameter and each
                                                 # walker's steps in a row

    # Make n subplots, one for each parameter, to see if they converge
    xs = np.arange(0, n_steps)
    fig, axs = plt.subplots(n_dim)
    fig.suptitle('Convergence of Walkers')
    for i in range(n_walkers):
        for j in range(n_dim):
            axs[j].plot(xs, gs[j][i])

    for k in range(n_dim):
        axs[k].set_ylabel(labels[k])

    axs[-1].set_xlabel('Step Number')
    plt.show()

    # Show corner plot with the last run
    last20_pct = int(n_steps * 0.8)
    fc = last_chain[last20_pct:]
    fc = np.reshape(fc, (len(fc)*n_walkers, n_dim))
    corner(fc, truths=None, labels=labels)
    plt.show()

    return np.array(last_steps), gs, fc

test1, test2, test3 = run_sims(10, init_p, time, flux, ferr, 100, 100, 1, p_labels)
