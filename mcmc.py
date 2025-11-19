import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import log_prob, corner_plot
# from scipy.optimize import minimize
import emcee as mc
import pickle
from statistics import stdev
from multiprocessing import Pool


# Start simulations
def run_sims(init, explore_scale, logprob, x, y, yerr, n_burn, n_steps, n_runs, labels, title):
    """ Run MCMC n_runs times
    Args:
        init (ndarray): List of initial guesses
        explore_scale (ndarray): Range of scatter for data points
        logprob (callable): Log probability function to use
        x (ndarray): x-values of data points
        y (ndarray): y-values of data points
        yerr (ndarray): Error in y-values of data points
        n_burn (int): Number of burn-in steps
        n_steps (int): Number of steps in each MCMC run
        n_runs (int): Number of times to run burn ins
        labels (list: str): List of labels for corner plot
        title (str or float): Word or number to include in plot titles
    Returns:
         tuple: ndarray, ndarray, ndarray
         last steps of each run, guesses for the last run, flatchain that will be used in corner plot
    """
    if type(title) == np.float64:
        title = round(title, 3)

    n_dim = len(labels)
    n_walkers = 10 * n_dim
    p0 = np.array(init) + (np.random.randn(n_walkers, n_dim) * explore_scale)

    with Pool(processes=6) as pool:
        moves = mc.moves.StretchMove(a=1.8)
        sampler = mc.EnsembleSampler(n_walkers, n_dim, logprob, args = (x, y, yerr), pool=pool, moves=moves)
        count = 0
        while count < n_runs:
            p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
            p1_state = p1[0]

            # Set up next run with best step of previous run
            lps = [logprob(i, x, y, yerr) for i in p1_state]
            max_lp = max(lps)
            new_init = p1_state[lps.index(max_lp)]
            rescale = explore_scale * 0.1
            p0 = new_init + np.random.randn(n_walkers, n_dim) * rescale

            n_burn = n_burn // 2

            af = sampler.acceptance_fraction
            print("acceptance fraction (mean, min, max):", af.mean(), af.min(), af.max())

            stuck_idx = np.where(af == 0.0)[0]
            print("stuck walkers:", stuck_idx)
            logp_counts = np.isfinite(sampler.get_log_prob(flat=False)).sum(axis=0)
            for i in stuck_idx:
                print(i, "finite logp count:", logp_counts[i])
                print("last pos:", sampler.get_chain()[-1, i])

            sampler.reset()
            count += 1

        sampler.reset()
        sampler.run_mcmc(p1, n_steps, progress=True)  # Actual simulation

        # gpt
        af = sampler.acceptance_fraction
        print("acceptance fraction (mean, min, max):", af.mean(), af.min(), af.max())
        tau = sampler.get_autocorr_time(tol=0)
        print("Autocorrelation time per parameter:", tau)
        print("Mean τ:", np.mean(tau))
        print("Run length / mean τ:", n_steps / np.mean(tau))       # should be greater than or equal to 50

        last_chain = sampler.get_chain()
        last_steps = last_chain[-1]
        gs = last_chain.transpose(2, 1, 0)           # Transpose the entire chain so that each guesses[i] shows each parameter and each
                                                     # walker's steps in a row

        # Make n subplots, one for each parameter, to see if they converge
        xs = np.arange(0, n_steps)
        fig, axs = plt.subplots(n_dim)
        fig.suptitle(str(title)+ r'$\mu$' 'm Convergence of Walkers')
        for i in range(n_walkers):
            for j in range(n_dim):
                axs[j].plot(xs, gs[j][i])

        for k in range(n_dim):
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('Step Number')
        plt.savefig(str(title)+'_convergence.pdf')

        # Plot only the last xx% of the steps
        last_pct = int(n_steps * 0.9)
        fig2, axs2 = plt.subplots(n_dim)
        fig2.suptitle(str(title)+ r'$\mu$' 'm Convergence of Walkers, last ' + str(int((n_steps - last_pct)/n_steps * 100)) + '% of steps')

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
        plt.savefig(str(title)+'_convergence_10pct.pdf')

        # Make corner plot with last bit of last run
        fc = last_chain[last_pct:]
        fc = np.reshape(fc, (len(fc)*n_walkers, n_dim))        # Last xx% of the steps for each walker is saved in fc

        corner_plot(fc, labels=labels, title=title)

    return np.array(last_steps), gs, fc


# Define data points for spitzer
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
time, flux = np.delete(time, [1317, 1318, 1420]), np.delete(flux, [1317, 1318, 1420])
ferr = np.abs(flux * (0.001 * np.random.randn(len(flux))))

# Initial guess
# init_p = np.array([57859.318, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.24, 0.30])
init_p = np.array([np.radians(-3), 0.22, 0.45])
explore = np.array([np.radians(2.0), 0.05, 0.05])
# soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
#             r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']
p_labels = [r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']

# ls, gs, fc = run_sims(init_p, explore, log_prob, time, flux, ferr, 100, 100, 1, p_labels)