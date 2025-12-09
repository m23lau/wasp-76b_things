import corner
import numpy as np
import matplotlib.pyplot as plt
from mcmc_funcs import corner_plot
# from scipy.optimize import minimize       # don't technically need this but can uncomment (it'll just take longer)
import emcee as mc
import pickle
from statistics import stdev
from multiprocessing import Pool
from kelp_models import big_fig
from joblib import Parallel, delayed


# Start simulations
def run_sims(init, explore_scale, model, loglike, logprob, t, f, ferr, n_burn, n_steps, n_runs, labels, title):
    """ Run MCMC n_runs times
    Args:
        init (ndarray): List of initial guesses
        explore_scale (ndarray): Range of scatter for data points
        model (callable): Phase curve model to use
        loglike (callable): Log likelihood function to use
        logprob (callable): Log probability function to use
        t (ndarray): Time values of data points
        f (ndarray): Flux values of data points
        ferr (ndarray): Error in f-values of data points
        n_burn (int): Number of burn-in steps
        n_steps (int): Number of steps in each MCMC run
        n_runs (int): Number of times to run burn ins
        labels (list: str): List of labels for corner plot
        title (str or float): Word or number to include in plot titles
    Returns:
         tuple: ndarray, ndarray, ndarray
         best step of the run, guesses for the most recent run, flatchain used in corner plot
    """
    if type(title) == np.float64:
        title = round(title, 3)

    n_dim = len(labels)
    n_walkers = 10 * n_dim
    p0 = np.array(init) + (np.random.randn(n_walkers, n_dim) * explore_scale)

    with Pool() as pool:
        moves = mc.moves.StretchMove(a=1.8)
        sampler = mc.EnsembleSampler(n_walkers, n_dim, logprob, args = (t, f, ferr), pool=pool, moves=moves)
        count = 0
        while count < n_runs:
            p1 = sampler.run_mcmc(p0, n_burn, progress = True)      # Burn-in
            p1_state = p1[0]

            # Set up next run with best step of previous run
            lps = [logprob(i, t, f, ferr) for i in p1_state]
            new_init = p1_state[lps.index(max(lps))]
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

    # Check acceptance fraction and autocorrelation times, make sure chain actually worked
    af = sampler.acceptance_fraction
    print("acceptance fraction (mean, min, max):", af.mean(), af.min(), af.max())   # if min is ~0.0, that means there is at lesat one stuck walker
    tau = sampler.get_autocorr_time(tol=0)
    print("Autocorrelation time per parameter:", tau)
    print("Mean τ:", np.mean(tau))
    print("Run length / mean τ:", n_steps / np.mean(tau))       # should be greater than or equal to 50

    last_chain = sampler.get_chain()

    # Find walker with max log likelihood which will give best fit params, bt
    all_steps = last_chain.reshape(n_steps * n_walkers, n_dim)
    lls = Parallel(n_jobs=8)(delayed(loglike)(i, t, f, ferr) for i in all_steps)       # Change njobs argument for more/less cores
    bt = all_steps[lls.index(max(lls))]

    # Save each step of the chain and its corresponding log-likelihood
    h = np.column_stack((all_steps, lls))
    head = ','
    head = head.join(labels)
    np.savetxt(str(title)+'_entire_chain.txt', h, delimiter=', ', header=head+', log-likelihood')

    # Transpose the entire chain so that each guesses[i] shows each parameter and each walker's steps in a row
    gs = last_chain.transpose(2, 1, 0)

    # Make n subplots, one for each parameter, to see if they converge
    xs = np.arange(0, n_steps)
    fig, axs = plt.subplots(n_dim, sharex=True)
    fig.suptitle(str(title)+ r'$\mu$' 'm Convergence of Walkers')
    for i in range(n_walkers):
        for j in range(n_dim):
            axs[j].plot(xs, gs[j][i])

    for k in range(n_dim):
        axs[k].set_ylabel(labels[k])
    axs[-1].set_xlabel('Step Number')
    plt.savefig(str(title)+'_convergence.pdf')
    plt.close()

    # Plot only the last xx% of the steps
    last_pct = int(n_steps * 0.9)
    fig2, axs2 = plt.subplots(n_dim, sharex=True)
    fig2.suptitle(str(title)+ r'$\mu$' 'm Convergence of Walkers, last ' + str(int((n_steps - last_pct)/n_steps * 100)) + '% of steps')

    skip = int(n_steps * 0.01)
    line = []
    err = []

    for i in range(n_dim):
        line.append(np.average(gs[i], axis=0)[last_pct::skip])         # line has shape (ndim, nsteps/skip)
        gs_t = gs[i].transpose()[last_pct::skip]                       # transpose gs to get all the walkers in a row
        for j in gs_t:
            err.append(stdev(j))             # get stdev of each row of gs' transpose (i.e. gs column)

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
    plt.close()

    # Make corner plot with last bit of last run
    fc = last_chain[last_pct:]
    fc = np.reshape(fc, (len(fc)*n_walkers, n_dim))       # Last xx% of the steps for each walker is saved in fc

    corner_plot(fc, labels=labels, title=title)
    fct = fc.transpose()

    # ndim x 3 array that stores median and +/- 1 sigma fit params
    msarr = np.array([corner.quantile(fct[i], [0.16, 0.50, 0.84]) for i in range(len(fct))])

    fit_params = np.column_stack([bt, msarr])
    np.savetxt(str(title)+'_fitparams.txt', fit_params, delimiter=', ', header = 'best, minus sigma, median, plus sigma')

    # Plot and save phase curve, cut out uncertainty param for input params of big_fig
    phasecurve, med_curve, msigma_curve, psigma_curve = big_fig(t, f, ferr, model, bt[:-1], msarr[:-1].transpose(), title=title)

    pcvals = np.column_stack([t, f, phasecurve, msigma_curve, med_curve, psigma_curve])
    np.savetxt(str(title)+'_pcvals.txt', pcvals, delimiter=', ', header='time, flux, phasecurve, '
                                                                           'minus sigma, median, plus sigma')

    return bt, gs, fc


# Define data points for spitzer
with open('Bestfit_Poly5_v1_autoRun.pkl', 'rb') as f:
    data = pickle.load(f)
time = data[1]
flux = data[2] / data[4]
time, flux = np.delete(time, [1317, 1318, 1420]), np.delete(flux, [1317, 1318, 1420])
ferr = np.abs(flux * (0.001 * np.random.randn(len(flux))))

# Initial guess
# init_p = np.array([57859.318, 1.81, 89.6, 0.108, 4.08, 0.01, 0.01, np.radians(-3), 0.24, 0.30])
init_p = np.array([np.radians(-3), 0.22, 0.45, 1.0])
explore = np.array([np.radians(2.0), 0.05, 0.05, 0.1])
# soln = minimize(lambda p, t, f, ferr: -log_prob(p, t, f, ferr), init_p, args=(time, flux, ferr), method="Powell")

# p_labels = [r'$t_0$', 'Per', 'Inc', r'$r_p$', 'a', 'q1', 'q2',
#             r'$\Delta \phi$', r'$A_B$', r'$C_{11}$']
p_labels = [r'$\Delta \phi$', r'$A_B$', r'$C_{11}$', 'Unc']

from mcmc_funcs import pc_model, log_prob, log_likelihood   # Uncomment below to test mcmc for Spitzer data
# bt, gs, fc = run_sims(init_p, explore, pc_model, log_likelihood, log_prob, time, flux, ferr, 10, 100, 1, p_labels, title='69')
# print(bt)