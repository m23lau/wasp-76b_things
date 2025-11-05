import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import log_prob, log_prior
# from scipy.optimize import minimize
from emcee import EnsembleSampler
from corner import corner
import pickle
# from scipy.stats import mode
import math
from multiprocessing import Pool
from statistics import stdev


def corner_plot(flatchain, truths, labels):
    """ Scans for and takes out stray walkers, then plots a corner plot
    Args:
        flatchain (ndarray): 1D or 2D array that feeds into corner plot
        truths(list: float): List of "True Values" to be highlighted in plot
        labels (list: str): List of labels for plot
    Returns:
         ndarray: Modified flatchain
    """

    go_away = []
    # for i in range(len(flatchain)):
    #     for j in range(len(truths)):
    #         if not math.isclose(flatchain[i][j], truths[j], rel_tol=1e-3):
    #             go_away.append(i)

    for i in range(len(flatchain)):
        if not math.isclose(flatchain[i][0], truths[0], rel_tol=1e-6):
            go_away.append(i)

    newfc = np.delete(flatchain, go_away, axis=0)
    print(len(flatchain), len(newfc))
    print(truths)

    # Plot corner plot before removing strays
    corner(flatchain, truths=truths, labels=labels)
    plt.show()

    # Plot corner plot after removing strays, plus print out the lengths of each array that corner.corner takes
    corner(newfc, truths=truths, labels=labels)
    plt.show()

    return newfc


# Define data points
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
ferr = np.abs(flux * (0.00002 * np.random.randn(len(flux))))


# Initial guess
init_p = np.array([57859.318, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.24, 0.30])
bounds = [[57855, 57865], [1.8, 1.85], [87.0, 92.0], [0.1, 0.12], [4.0, 4.2], [0.0, 0.02], [0.0, 0.02],
          [-0.1221, 0.0873], [0.1, 0.5], [0.2, 0.5]]
# soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# later on, guesses[i] corresponds to the ith parameter in the list below
p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
            r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']

# Start simulations
def run_sims(init, x, y, yerr, n_burn, n_steps, n_runs, labels):
    """ Run MCMC n_runs times
    Args:
        init (ndarray): List of initial guesses
        x (ndarray): x-values of data points
        y (ndarray): y-values of data points
        yerr (ndarray): Error in y-values of data points
        n_burn (int): Number of burn-in steps
        n_steps (int): Number of steps in each MCMC run
        n_runs (int): Number of times to run burn ins
        labels (list: str): List of labels for corner plot
    Returns:
         tuple: ndarray, ndarray, ndarray
         last steps of each run, guesses for the last run, flatchain that will be used in corner plot
    """
    n_dim = len(labels)
    n_walkers = 10 * n_dim
    p0 = [init + 0.001 * np.random.randn(n_dim) for _ in range(n_walkers)]

    with Pool() as pool:
        sampler = EnsembleSampler(n_walkers, n_dim, log_prob, args = (x, y, yerr), pool=pool)
        count = 0
        while count < n_runs:
            p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
            p1_state = p1[0]

            # Set up next run with best step of previous run
            lps = [log_prob(i, x, y, yerr) for i in p1_state]
            max_lp = max(lps)
            new_init = p1_state[lps.index(max_lp)]
            p0 = [new_init + 0.00001 * np.random.randn(n_dim) for _ in range(n_walkers)]
            sampler.reset()
            count += 1

        sampler.reset()
        sampler.run_mcmc(p1, n_steps, progress=True)  # Actual simulation

    last_chain = sampler.get_chain()
    last_steps = np.average(last_chain[-1], axis = 0)
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
    last_pct = int(n_steps * 0.9)
    fig2, axs2 = plt.subplots(n_dim)
    fig2.suptitle('Convergence of Walkers, last ' + str(int((n_steps - last_pct)/n_steps * 100)) + '% of steps')

    skip = int(n_steps * 0.01)
    line = []
    err = []

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
        axs2[i].plot(xs, line[i], color = 'r', marker = 'o')
        axs2[i].fill_between(xs, line[i] - err[i], line[i] + err[i], color='r', alpha=0.3)

    for k in range(n_dim):
        axs2[k].set_ylabel(labels[k])
    axs2[-1].set_xlabel('Step Number')

    # Make corner plot with last bit of last run
    fc = last_chain[last_pct:]
    fc = np.reshape(fc, (len(fc)*n_walkers, n_dim))        # Last xx% of the steps for each walker is saved in fc

    corner_plot(fc, truths=new_init, labels=labels)

    return np.array(last_steps), gs, fc

ls, gs, fc = run_sims(init_p, time, flux, ferr, 1000, 5000, 2, p_labels)



import csv
with open('file.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(p_labels)
    for i in range(len(fc)):
        writer.writerow([fc[i][0], fc[i][1], fc[i][2], fc[i][3], fc[i][4], fc[i][5], fc[i][6], fc[i][7],
                         fc[i][8], fc[i][9]])