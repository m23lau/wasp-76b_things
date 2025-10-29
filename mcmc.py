import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import log_prob, log_prior
# from scipy.optimize import minimize
from emcee import EnsembleSampler
from corner import corner
import pickle
# from scipy.stats import mode
# import math
from multiprocessing import Pool
from statistics import stdev


# Define data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
ferr = np.abs(flux * (0.00002 * np.random.randn(len(flux))))


# Initial guess
init_p = np.array([56107, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.24, 0.30])
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
    line = []
    err = []

    n_walkers = 10 * n_dim
    p0 = [init + 0.001 * np.random.randn(n_dim) for _ in range(n_walkers)]

    with Pool() as pool:
        sampler = EnsembleSampler(n_walkers, n_dim, log_prob, args = (x, y, yerr), pool=pool)

        while len(last_steps) < n_runs:
            p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
            sampler.reset()
            sampler.run_mcmc(p1, n_steps, progress = True)          # Actual simulation
            chain = sampler.get_chain()

            # Set up next run with last step of previous run
            new_init = np.average(chain[-1], axis=0)
            last_steps.append(new_init)
            p0 = [new_init + 0.001 * np.random.randn(n_dim) for _ in range(n_walkers)]

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

    # Plot only the last xx% of the steps
    last_pct = int(n_steps * 0.8)
    fig2, axs2 = plt.subplots(n_dim)
    fig2.suptitle('Convergence of Walkers, last ' + str(int((n_steps - last_pct)/n_steps * 100)) + '% of steps')

    skip = 10
    for i in range(n_dim):
        line.append(np.average(gs[i], axis=0)[last_pct::skip])         # line has shape (ndim, nsteps/skip)
        gs_t = gs[i].transpose()[last_pct::skip]                       # transpose gs to get all the walkers in a row
        for j in gs_t:
            err.append(stdev(j))                                    # get stdev of each row of gs' transpose (i.e. gs column)

    line = np.array(line)
    err = np.reshape(err, line.shape)  # reshape err so it has same dimensions as line
    err = np.array(err)
    xs = np.arange(last_pct, n_steps, skip)

    for i in range(len(line)):
        axs2[i].plot(xs, line[i], color='r')
        axs2[i].fill_between(xs, line[i] - err[i], line[i] + err[i], color='r', alpha=0.3)

    for k in range(n_dim):
        axs2[k].set_ylabel(labels[k])
    axs2[-1].set_xlabel('Step Number')

    # Make corner plot with last bit of last run
    fc = last_chain[last_pct:]
    fc = np.reshape(fc, (len(fc)*n_walkers, n_dim))        # Last xx% of the steps for each walker is saved in fc

    corner(fc, truths=None, labels=labels)
    plt.show()

    return np.array(last_steps), gs, fc

ls, gs, fc = run_sims(10, init_p, time, flux, ferr, 1000, 1000, 1, p_labels)


