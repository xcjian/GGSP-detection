from utils import *
from scipy.optimize import minimize
import time
import smom_functions
import warnings
import numpy as np
import pickle
import os

from functools import partial  # To give a second argument to mp.map
import pathos.multiprocessing as mp
from scipy import stats

from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def est_paras_beta_ggsp(p_val, sample_points, graph_basis, time_max_idx, bandwidths, nonlinear_type):
    """
    Estimate the alternative p-value PDF using Beta-GGSP, but do not threshold the parameters.

    Parameters
    ----------
    p_val: p-values, 1D numpy.ndarray
    sample_points: M x 2 numpy.ndarray
        The sample points. Time is in the form of integer stamps {0,1,2,...time_max_idx}.
        The first column is the vertex index and the second column is the time index.
    graph_basis: N x N numpy.ndarray
        The Fourier basis matrix for the graph.
    time_max_idx: int
        The maximum time index.
    bandwidths: dict
        The bandwidth ranges for the graph and time Fourier basis.
        The keys are 'graph_bw_ran' and 'time_bw_ran'.
        The values are lists of bandwidths to be tried.
    nonlinear_type: str
        The choice of nonlinear function on generalized graph signal.
    Returns
    -------
    alpha_hat: numpy.ndarray
        The estimated alpha.
    """

    n_samples = p_val.shape[0]
    n_vertex = graph_basis.shape[0]
    n_time = time_max_idx + 1

    # solve the Beta-GGSP model
    ## load bandwidth ranges
    gft_bw_range = bandwidths['graph_bw_ran']
    tft_bw_range = bandwidths['time_bw_ran']
    gft_bw_best = 0
    tft_bw_best = 0
    alpha_best = 0.5
    BIC_best = np.inf

    for gft_bw in gft_bw_range:
        for tft_bw in tft_bw_range:

            ## calculate the joint Fourier basis
            ### convert time instances to [-1, 1]:
            basis_matrix = basis_construct_ggsp(sample_points, gft_bw, tft_bw, graph_basis, time_max_idx)

            start_time = time.time()
            beta_hat, alpha_hat, negative_logllh = beta_ggsp_solver_scipy(p_val, basis_matrix, nonlinear_type)
            end_time = time.time()

            n_paras = basis_matrix.shape[1]
            BIC = n_paras * np.log(n_samples) + 2 * negative_logllh

            if BIC < BIC_best:
                gft_bw_best = gft_bw
                tft_bw_best = tft_bw
                alpha_best = alpha_hat
                beta_best = beta_hat
                BIC_best = BIC

            print(f"Time elapsed: {end_time - start_time}", "gft_bw: ", gft_bw, "tft_bw: ", tft_bw, "negative_logllh: ", negative_logllh, "BIC:", BIC)

    print(f"Best bandwidths: alpha_gft_bw = {gft_bw_best}, alpha_tft_bw = {tft_bw_best}")
    alpha_hat = alpha_best
    beta_hat = beta_best

    pi0_hat = alpha_hat
    pi0_hat[pi0_hat > 1] = 1 # This won't happen in practice
    f1_p = (alpha_hat * (p_val ** (alpha_hat - 1)) - alpha_hat) / (1 - alpha_hat)
    f1_p_strList = np.array([str(x) for x in f1_p], dtype=str)
    nan_idx = np.where(f1_p_strList == 'nan')[0]

    if len(nan_idx) > 0:
        f1_p[nan_idx] = 0 # assign a valid value instead of nan. The concrete value does not matter
        # since you will always multiply is with 1 - 1 = 0 when computing lfdr.

    # compute pi0 hat for all p values, even those that are not observed
    sample_points_all = np.array([[i, j] for i in range(n_time - 1) for j in range(n_vertex)])
    basis_matrix_all = basis_construct_ggsp(sample_points_all , gft_bw_best, tft_bw_best, graph_basis, time_max_idx)
    if nonlinear_type == 'sigmoid':
        alpha_hat_all = 1 / (1 + np.exp(- basis_matrix_all @ beta_hat))
    else:
        raise ValueError("Unconstructed nonlinear function type, can manually add this in the code if you need.")

    # set the estimated alpha to be at least small_val.
    pi0_hat_all = alpha_hat_all

    return f1_p, pi0_hat, pi0_hat_all


def beta_ggsp_solver_scipy(p_val, basis_matrix, nonlinear_type):
    """
    Solve the beta for GGSP model, using scipy solver.

    Parameters
    ----------
    p_val : 1D numpy.ndarray
        The p values.
    basis_matrix : numpy.ndarray
        The basis matrix.
    nonlinear_type : str
    The choice of nonlinear function on generalized graph signal.
    'softplus' for softplus function.
    'identity' for identity function.
    'exp' for exponential function.
    'quad' for quadratic function.

    Returns
    -------
    beta : numpy.ndarray
        The beta.
    """

    large_number = 1e5
    small_number = 1e-5

    beta_len = basis_matrix.shape[1]
    alpha_len = basis_matrix.shape[0]

    # Define and solve the problem.
    def objective_identity(beta):
        term1 = -np.log(basis_matrix @ beta)
        term2 = -np.multiply(basis_matrix @ beta - 1, np.log(p_val))
        return np.sum(term1 + term2)

    def objective_softplus(beta):
        softplus_val = np.log(1 + np.exp(basis_matrix @ beta))
        term1 = -np.log(softplus_val)
        term2 = -np.multiply(softplus_val - 1, np.log(p_val))
        return np.sum(term1 + term2)

    def objective_exp(beta):
        exp_val = np.exp(basis_matrix @ beta)
        term1 = -np.log(exp_val)
        term2 = -np.multiply(exp_val - 1, np.log(p_val))
        return np.sum(term1 + term2)

    def objective_quad(beta):
        square_val = np.square(basis_matrix @ beta)
        term1 = -np.log(square_val)
        term2 = -np.multiply(square_val - 1, np.log(p_val))
        return np.sum(term1 + term2)

    def objective_sigmoid(beta):
        # capped_val = np.maximum(basis_matrix @ beta, -50)
        square_val = 1 / (1 + np.exp(-basis_matrix @ beta))
        term1 = -np.log(square_val)
        term2 = -np.multiply(square_val - 1, np.log(np.maximum(p_val, 10**(-50))))
        return np.sum(term1 + term2)

    if nonlinear_type == 'identity':
        objective = objective_identity
        constraints = {'type': 'ineq', 'fun': lambda beta: basis_matrix @ beta, 'lb': small_number, 'ub': 1 - small_number}
        # Initial guess for beta
        beta_initial = np.ones(beta_len)
        beta_initial[0] = 0.5 / basis_matrix[0, 0]

    elif nonlinear_type == 'softplus':
        objective = objective_softplus
        constraints = {'type': 'ineq', 'fun': lambda beta: np.log(1 + np.exp(basis_matrix @ beta)), 'lb': small_number, 'ub': 1 - small_number}
        # Initial guess for beta
        beta_initial = np.zeros(beta_len)

    elif nonlinear_type == 'exp':
        objective = objective_exp
        constraints = {'type': 'ineq', 'fun': lambda beta: np.exp(basis_matrix @ beta), 'lb': small_number, 'ub': 1 - small_number}
        # Initial guess for beta
        beta_initial = np.ones(beta_len)
        beta_initial[0] = -0.5 / basis_matrix[0, 0]

    elif nonlinear_type == 'quad':
        objective = objective_quad
        constraints = {'type': 'ineq', 'fun': lambda beta: np.square(basis_matrix @ beta), 'lb': small_number, 'ub': 1 - small_number}
        # Initial guess for beta
        beta_initial = np.ones(beta_len)
        beta_initial[0] = 0.5 / basis_matrix[0, 0]

    elif nonlinear_type == 'sigmoid':
        objective = objective_sigmoid
        # Initial guess for beta
        beta_initial = np.zeros(beta_len)
        beta_initial[0] = 0.5 / basis_matrix[0, 0]

    else:
        raise ValueError("Invalid nonlinear function type")

    # Solve the problem using scipy.optimize.minimize
    # try the following methods: 'trust-constr'
    # result = minimize(objective, beta_initial, constraints=constraints, method='Nelder-Mead')

    # result = minimize(objective, beta_initial, constraints=constraints, method='trust-constr')

    result = minimize(objective, beta_initial, method='Powell')
    beta_optimal = result.x

    # Print the results
    if not result.success:
        print(result.message)

    # Calculate the optimal alpha, depending on the nonlinear function
    if nonlinear_type == 'identity':
        alpha_optimal = basis_matrix @ beta_optimal
    elif nonlinear_type == 'softplus':
        alpha_optimal = np.log(1 + np.exp(basis_matrix @ beta_optimal))
    elif nonlinear_type == 'exp':
        alpha_optimal = np.exp(basis_matrix @ beta_optimal)
    elif nonlinear_type == 'quad':
        alpha_optimal = np.square(basis_matrix @ beta_optimal)
    elif nonlinear_type == 'sigmoid':
        alpha_optimal = np.exp(basis_matrix @ beta_optimal) / (1 + np.exp(basis_matrix @ beta_optimal))
    else:
        raise ValueError("Invalid nonlinear function type")


    alpha_optimal[alpha_optimal < 0] = 10 ** (-5)
    alpha_optimal[alpha_optimal > 1] = 1 - 10 ** (-5) # These won't happen in practice
    # Return the best solution and the corresponding negative log likelihood
    nnll_opt = -beta_log_likelihood(p_val, alpha_optimal)

    return beta_optimal, alpha_optimal, nnll_opt


def parallel_smom(p_val, par_type, dat_path, max_wrk, partition, quant_bits,
                  sensoring_thr):
    """Estimate the alternative p-value PDF for lfdr-sMoM with parallelization.

    Parameters
    ----------
    fd : RadioSpatialField/RadioSpatialFieldEstimated
        The field the lfdrs are to be estimated for.
    par_type : str
        The smom parametrization type. Recommended: 'stan' for standard.
    dat_path : str
        The path to where the data is stored.
    partition : str
        The type of partition for obtaining the p-value vectors.
    quant_bits : int or None
        The number of bits used for quantization.
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring occurs.

    Returns
    -------
    list
        The list with all estimated quantities.
    """
    with open(os.path.join(dat_path, "", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        if partition == 'spatial':
            mom_d = mom_d[
                np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
        del ld_par

    num_wrk = np.min((max_wrk, os.cpu_count() - 1))
    print(f"Starting a parallel pool with {num_wrk} workers.")
    p_val = p_val[np.newaxis, :]
    par_pl = mp.Pool(num_wrk)
    if partition == 'spatial':
        rtns = par_pl.map(partial(
            single_run_smom_spatial, dat_path=dat_path, par_type=par_type,
            quant_bits=quant_bits, sensoring_thr=sensoring_thr), p_val)
    elif partition == 'random':
        rtns = par_pl.map(partial(
            single_run_smom_random, dat_path=dat_path, par_type=par_type,
            quant_bits=quant_bits, sensoring_thr=sensoring_thr), p_val)
    par_pl.close()

    # Setting up null proportion and alternative density
    n_MC = p_val.shape[0]
    n_nodes = p_val.shape[1]

    pi_0_hat = np.zeros((n_MC))
    f1_hat = np.zeros((n_MC, n_nodes))

    # Setting up the variables for storing the results
    mom_lam_hats = np.zeros((n_MC, np.max(mom_k)))
    mom_a_hats = np.zeros((n_MC, np.max(mom_k), np.max(mom_d)))
    mom_p_pdf = np.zeros((n_MC, n_nodes))
    mom_p_cdf = np.zeros((n_MC, n_nodes))
    mom_sel_k = np.zeros((n_MC))
    mom_sel_d = np.zeros((n_MC))
    ex_time = np.zeros(n_MC)
    mom_diff_best = np.zeros(n_MC)
    for mc in np.arange(0, n_MC, 1):
        # Unpacking the results of the spectral method of moments
        mom_a_hats[mc, :, :] = rtns[mc][0]
        mom_lam_hats[mc, :] = rtns[mc][1]
        mom_p_pdf[mc, :] = rtns[mc][2]
        mom_p_cdf[mc, :] = rtns[mc][3]
        mom_diff_best[mc] = rtns[mc][4]
        mom_sel_k[mc] = rtns[mc][5]
        mom_sel_d[mc] = rtns[mc][6]
        ex_time[mc] = rtns[mc][7]
        # Estimating the alternative density and null proportion by Pounds
        # method
        (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
            [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
             np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
             mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
    return (f1_hat, pi_0_hat, mom_lam_hats, mom_a_hats,
            mom_p_pdf, mom_p_cdf, mom_diff_best, mom_sel_k, mom_sel_d,
            ex_time)

def single_run_smom_spatial(p_val, dat_path, par_type, quant_bits,
                            sensoring_thr):
    """Apply smom with spatial partitioning for a single MC run.

    Parameters
    ----------
    p_val : numpy array
        The p-values for this MC run.
    dat_path : str
        The path to where the data is stored.
    par_type : str
        The smom parametrization. Recommended: 'stan' for standard
    quant_bits : int or None
        The number of bits for quantization. If None, then no quantization
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring.

    Returns
    -------
    list
        The quantities estimated by sMoM.
    """
    with open(os.path.join(dat_path, "", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        del ld_par
    # Reading in method of moments parameters
    mom_d = mom_d[np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
    start_time = time.time()
    # Setting up the grid for the goodness-of-fit statistic
    if dis_msr == 'js' or dis_msr == 'kl':
        bin_wdt = 10/p_val.size  # Need to make sure that there are few empty
        # bins.
    elif dis_msr == 'ks' or dis_msr == 'was':
        bin_wdt = 3/p_val.size  # Need as many bins as possible for EDF-based
        # measures
        bin_wdt = 1/1000
    else:
        print("Invalid distance measure")

    grd = np.arange(1e-10, 1, bin_wdt)  # the grid
    grd_cut_off_idx = -1
    bns = np.arange(1e-10-bin_wdt/2, 1+bin_wdt/2, bin_wdt)  # the bin edges

    if quant_bits is not None:
        with open(os.path.join(dat_path, '',
            f'quan_{quant_bits}Bit_sensoring_at_{sensoring_thr}') + '_par.pkl',
            'rb') as input:
            loaded = pickle.load(input)
            (bins, bin_wdt, grd) = loaded[0], loaded[1], loaded[2]


    # Spatial division of the data in tiles
    N = p_val.size
    dim = (int(np.sqrt(N)), int(np.sqrt(N)))

    # Number of tiles along x and y axis
    tls_per_x = (np.floor(np.sqrt(N/mom_d))).astype(int)
    tls_per_y = (np.floor(np.sqrt(N/mom_d))).astype(int)

    # Specification of the tile parameters
    num_tiles = tls_per_x * tls_per_y
    tls_x_len = (np.floor(dim[0]/tls_per_x)).astype(int)
    tls_y_len = (np.floor(dim[1]/tls_per_y)).astype(int)

    # Computation of the edf on the grid for g.o.f
    grd_emp_prob = np.histogram(p_val, bins=bns, density=True)[0]*bin_wdt
    grd_edf = np.cumsum(grd_emp_prob)
    # Initialization of g.o.f measures
    diff_best = np.inf
    sel_d = np.nan
    sel_k = np.nan

    if dis_msr == 'ks':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.max(np.abs((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx]), 1)
    elif dis_msr == 'msd':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.mean(((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx])**2, 1)
    elif dis_msr == 'was':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = wasserstein_distance(
                    grd_edf, grd_cdf[t_idx, :])
            return diff_k
    elif dis_msr == 'kl':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = entropy(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)
            return diff_k
    elif dis_msr == 'js':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = jensenshannon(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)**2
            return diff_k

    for d_idx in np.arange(0, mom_d.size, 1):
        if (not np.isnan(sel_d)) and sel_d < mom_d[d_idx-1]:
            break
        # Setting up the p-values divided into tiles
        p_div = np.zeros((
            num_tiles[d_idx], tls_x_len[d_idx]*tls_y_len[d_idx])) + np.nan
        for tr_idx in np.arange(0, mom_n_tr, 1):
            for spa_div_x_idx in np.arange(0, tls_per_x[d_idx], 1):
                for spa_div_y_idx in np.arange(0, tls_per_y[d_idx], 1):
                    tls_sel_mat = np.zeros(dim, dtype=bool)
                    (tle_starts_at_x_idx, tle_starts_at_y_idx) = (
                        spa_div_x_idx*tls_x_len[d_idx], spa_div_y_idx
                        * tls_y_len[d_idx])
                    tls_sel_mat[
                        tle_starts_at_x_idx:tls_x_len[d_idx]
                        +tle_starts_at_x_idx,
                        tle_starts_at_y_idx:tle_starts_at_y_idx+tls_y_len[
                            d_idx]] = True
                    p_vec = np.random.choice(
                        p_val, size=np.prod(dim),
                        replace=False).reshape(dim)[tls_sel_mat]
                    # Uncomment this to exclude the pile at 0 from the fitting
                    if ~np.all(p_vec == 0):
                        p_vec[np.where(p_vec == 0)] = np.min(
                            p_vec[np.where(p_vec != 0)])
                        p_div[spa_div_x_idx*tls_per_x[d_idx]
                        + spa_div_y_idx, :] = (
                            np.random.permutation(p_vec))
                    # p_div[spa_div_x_idx*tls_per_x + spa_div_y_idx, :] = (
                    #     np.random.permutation(p_vec))
            # Uncomment to exclude tiles with only 0 from the fitting
            tiles_for_fitting = np.where(~np.isnan(np.sum(p_div, 1)))[0]
            # Uncomment to include the pile at0
            # tiles_for_fitting = np.arange(0, num_tiles, 1)

            # Paramter estimation by spectral method of moments
            diff_all_k = np.zeros(mom_k.size)

            for (k_idx, k) in enumerate(mom_k):
                if k < mom_d[d_idx]:  # Size of multivariate vectors limits
                    # the number of mixture components.
                    a_hat, b_hat, w_hat, grd_pdf, grd_cdf = (
                        smom_functions.learnMBM(
                        p_div[tiles_for_fitting, :], tiles_for_fitting.size,
                        mom_d[d_idx], k, grd, mom_reps_eta,
                        gaussian_eta=True))

                    # Goodness of fit
                    diff_k = dis_msr_fct(grd_pdf, grd_cdf)
                    try:
                        min_idx = np.nanargmin(diff_k)
                        diff_all_k[k_idx] = diff_k[min_idx]
                        if diff_all_k[k_idx] - diff_all_k[
                            np.max((k_idx - 1, 0))] <= 0:
                            if diff_all_k[k_idx] - diff_best <= 0:
                                a_hat_win = np.copy(a_hat[min_idx, :, :])
                                b_hat_win = np.copy(b_hat[min_idx, :, :])
                                w_hat_win = np.copy(w_hat[min_idx, :])
                                diff_best = diff_all_k[k_idx]
                                sel_k = k
                                sel_d = mom_d[d_idx]
                        else:
                            break
                    except ValueError:
                        # print(['No valid result for this parametrization'
                        #       ' with averaging!'])
                        failed_at_least_once = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # PDF marginalized over coordinates
        beta_pdf_hat_av = get_pdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
        # CDF marginalized over coordinates
        beta_cdf_hat_av = get_cdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
    # Scaling the weights back to sum to 1
    w_hat_win[np.where(w_hat_win < 0)[0]] = 0
    w_hat_win = w_hat_win/np.sum(w_hat_win)
    # print('\rCompleted run {mc}/{nMC}'.format(mc=mc+1, nMC=mc), end="")
    a_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    b_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    w_hat_rtd = np.zeros((np.max(mom_k)))
    a_hat_rtd[0:sel_k, 0:sel_d] = a_hat_win
    b_hat_rtd[0:sel_k, 0:sel_d] = b_hat_win
    w_hat_rtd[0:sel_k] = w_hat_win
    ex_time = time.time() - start_time
    return (a_hat_rtd, w_hat_rtd, beta_pdf_hat_av,
            beta_cdf_hat_av, diff_best, sel_k, sel_d, ex_time)


def single_run_smom_random(p_val, dat_path, par_type, quant_bits,
                           sensoring_thr):
    """Apply smom with random partitioning for a single MC run.

    Parameters
    ----------
    p_val : numpy array
        The p-values for this MC run.
    dat_path : str
        The path to where the data is stored.
    par_type : str
        The smom parametrization. Recommended: 'stan' for standard
    quant_bits : int or None
        The number of bits for quantization. If None, then no quantization
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring.

    Returns
    -------
    list
        The quantities estimated by sMoM.
    """
    # Reading in method of moments parameters
    with open(os.path.join(dat_path, "", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        del ld_par
    start_time = time.time()

    # Setting up the grid for the goodness-of-fit statistic
    if dis_msr == 'js' or dis_msr == 'kl':
        bin_wdt = 10/p_val.size  # Need to make sure that there are few empty
        # bins.
    elif dis_msr == 'ks' or dis_msr == 'was':
        bin_wdt = 3/p_val.size  # Need as many bins as possible for EDF-based
        # measures
        bin_wdt = 1/1000
    else:
        print("Invalid distance measure")
    grd = np.arange(1e-10, 1, bin_wdt)  # the grid
    grd_cut_off_idx = -1
    bns = np.arange(1e-10-bin_wdt/2, 1+bin_wdt/2, bin_wdt)  # the bin edges

    if quant_bits is not None:
        with open(os.path.join(
            dat_path, '',
            f'quan_{quant_bits}Bit_sensoring_at_{sensoring_thr}') + '_par.pkl',
            'rb') as input:
            loaded = pickle.load(input)
            (bns, bin_wdt, grd) = loaded[0], loaded[1], loaded[2]

    # Spatial division of the data in tiles
    N = p_val.size

    # Specification of the tile parameters
    num_tiles = np.floor(N/mom_d).astype(int)
    N_tilde = num_tiles*mom_d.astype(int)

    # Computation of the edf on the grid for g.o.f
    grd_emp_prob = np.histogram(p_val, bins=bns, density=True)[0]*bin_wdt
    grd_edf = np.cumsum(grd_emp_prob)

    diff_best = np.inf
    sel_d = np.nan
    sel_k = np.nan
    if dis_msr == 'ks':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.max(np.abs((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx]), 1)
    elif dis_msr == 'msd':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.mean(((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx])**2, 1)
    elif dis_msr == 'was':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = wasserstein_distance(
                    grd_edf, grd_cdf[t_idx, :])
            return diff_k
    elif dis_msr == 'kl':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = entropy(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)
            return diff_k
    elif dis_msr == 'js':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = jensenshannon(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)**2
            return diff_k
    for d_idx in np.arange(0, mom_d.size, 1):
        # Setting up the p-values divided into tiles
        if (not np.isnan(sel_d)) and sel_d < mom_d[d_idx-1]:
            break
        for tr_idx in np.arange(0, mom_n_tr, 1):
            shuffled_pval_idx = np.random.permutation(
                np.arange(N))[0:N_tilde[d_idx]].reshape(
                    (num_tiles[d_idx], mom_d[d_idx]))
            p_div = p_val[shuffled_pval_idx]
            p_div[np.where(p_div == 0)] = np.min(p_div[np.nonzero(p_div)])

            # Parameter estimation by spectral method of moments
            diff_all_k = np.zeros(mom_k.size)
            for (k_idx, k) in enumerate(mom_k):
                if k < mom_d[d_idx]:  # Size of multivariate vectors limits
                    # the number of mixture components.
                    a_hat, b_hat, w_hat, grd_pdf, grd_cdf = (
                        smom_functions.learnMBM(p_div, num_tiles[d_idx],
                        mom_d[d_idx], k, grd, mom_reps_eta, gaussian_eta=True))
                    # Goodness of fit
                    diff_k = dis_msr_fct(grd_pdf, grd_cdf)
                    try:
                        min_idx = np.nanargmin(diff_k)
                        diff_all_k[k_idx] = diff_k[min_idx]
                        if (diff_all_k[k_idx]
                            - diff_all_k[np.max((k_idx - 1, 0))] <= 0):
                            if diff_all_k[k_idx] - diff_best <= 0:
                                a_hat_win = np.copy(a_hat[min_idx, :, :])
                                b_hat_win = np.copy(b_hat[min_idx, :, :])
                                w_hat_win = np.copy(w_hat[min_idx, :])
                                diff_best = diff_all_k[k_idx]
                                sel_k = k
                                sel_d = mom_d[d_idx]
                        else:
                            break
                    except ValueError:
                        # print(['No valid result for this parametrization'
                        #       ' with averaging!'])
                        failed_at_least_once = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # PDF marginalized over coordinates
        beta_pdf_hat_av = get_pdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
        # CDF marginalized over coordinates
        beta_cdf_hat_av = get_cdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
    # Scaling the weights back to sum to 1
    w_hat_win[np.where(w_hat_win < 0)[0]] = 0
    w_hat_win = w_hat_win/np.sum(w_hat_win)
    # print('\rCompleted run {mc}/{nMC}'.format(mc=mc+1, nMC=mc), end="")
    a_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    b_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    w_hat_rtd = np.zeros((np.max(mom_k)))
    a_hat_rtd[0:sel_k, 0:sel_d] = a_hat_win
    b_hat_rtd[0:sel_k, 0:sel_d] = b_hat_win
    w_hat_rtd[0:sel_k] = w_hat_win
    ex_time = time.time() - start_time
    return (a_hat_rtd, w_hat_rtd, beta_pdf_hat_av,
            beta_cdf_hat_av, diff_best, sel_k, sel_d, ex_time)

def for_loop_smom(p_val, par_type, dat_path, partition, quant_bits,
                  sensoring_thr):
    """Estimate the alternative p-value PDF for lfdr-sMoM with a for loop (for
    debugging.)

    Parameters
    ----------
    p_val : numpy array
        The p-values.
    par_type : str
        The smom parametrization type. Recommended: 'stan' for standard.
    dat_path : str
        The path to where the data is stored.
    partition : str
        The type of partition for obtaining the p-value vectors.
    quant_bits : int or None
        The number of bits used for quantization.
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring occurs.

    Returns
    -------
    list
        The list with all estimated quantities.
    """
    with open(os.path.join(dat_path, "", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        if partition == 'spatial':
            mom_d = mom_d[
                np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
        del ld_par

    # Setting up null proportion and alternative density
    p_val = p_val[np.newaxis, :]
    n_MC = p_val.shape[0]
    n_nodes = p_val.shape[1]

    pi_0_hat = np.zeros((n_MC))
    f1_hat = np.zeros((n_MC, n_nodes))

    # Setting up the variables for storing the results
    mom_lam_hats = np.zeros((n_MC, np.max(mom_k)))
    mom_a_hats = np.zeros((n_MC, np.max(mom_k), np.max(mom_d)))
    mom_p_pdf = np.zeros((n_MC, n_nodes))
    mom_p_cdf = np.zeros((n_MC, n_nodes))
    mom_sel_k = np.zeros((n_MC), dtype=int)
    mom_sel_d = np.zeros((n_MC), dtype=int)
    ex_time = np.zeros(n_MC)
    mom_diff_best = np.zeros(n_MC)
    if partition == 'spatial':
        for mc in np.arange(0, n_MC, 1):
            (mom_a_hats[mc, :, :],
                mom_lam_hats[mc, :], mom_p_pdf[mc, :],
                mom_p_cdf[mc, :], mom_diff_best[mc],
                mom_sel_k[mc], mom_sel_d[mc],
                ex_time[mc])  = single_run_smom_spatial(
                    p_val[mc, :], dat_path, par_type, quant_bits,
                    sensoring_thr)
            # Estimating the alternative density and null proportion by Pounds
            # method
            (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
                [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
                np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
                mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
            print(f"\rFinished MC run {mc+1}/{n_MC}", end="")
    elif partition == 'random':
        for mc in np.arange(0, n_MC, 1):
            (mom_a_hats[mc, :, :],
                mom_lam_hats[mc, :], mom_p_pdf[mc, :],
                mom_p_cdf[mc, :], mom_diff_best[mc],
                mom_sel_k[mc], mom_sel_d[mc],
                ex_time[mc]) = single_run_smom_random(p_val[mc, :], dat_path,
                par_type, quant_bits, sensoring_thr)
            # Estimating the alternative density and null proportion by Pounds
            # method
            (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
                [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
                np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
                mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
            print(f"\rFinished MC run {mc+1}/{n_MC}", end="")
    print("")
    return (f1_hat, pi_0_hat, mom_lam_hats, mom_a_hats,
            mom_p_pdf, mom_p_cdf, mom_diff_best, mom_sel_k, mom_sel_d,
            ex_time)

def apply_pounds_estimator(par_list, typ='emp'):
    """Estimate the null fraction and the alternative p-value PDF with the
    method proposed by Pounds.

    Parameters
    ----------
    typ : str
        Tells if we use a given PDF or the parameters of an MBM to compute the
        minimal value of the mixture PDF.
    par_list : list
        The list with additional parameters, depends on the typ.

    Returns
    -------
    list
        The estimated null proportion and the estimated PDF under H1.
    """
    pi0 = est_pi0_pounds(typ, par_list)

    # Underestimate the alternative component on the left hand side
    if pi0 < 1:
        f1_p = (par_list[0] - pi0)/(1-pi0)
    else:
        f1_p = np.zeros(par_list[0].size)
    if np.any(f1_p < 0):
        f1_p[np.where(f1_p < 0)] = 0

    return pi0, f1_p

def est_pi0_pounds(typ, par_list):
    """Estimate the null fraction with the method proposed by Pounds.

    Parameters
    ----------
    typ : str
        Tells if we use a given PDF or the parameters of an MBM to compute the
        minimal value of the mixture PDF.
    par_list : list
        The list with additional parameters, depends on the typ.

    Returns
    -------
    float
        The estimated null proportion.
    """
    if typ == 'mbm':
        [_, a, b, w, d, k] = par_list
        if a.ndim == 1:
            a = a[:, np.newaxis]
            b = b[:, np.newaxis]
        d = int(d)
        k = int(k)
        # Overestimate the alternative proportion
        p_grid = np.arange(1/1000, 1, 1/1000)
        # Check if this is an averaged parameter model or not
        pdf = get_pdf_multivariate_mbm(
                p_grid, a[0:k, 0:d], b[0:k, 0:d], w[0:k], d, k)
        pi0 = np.nanmin(pdf)
    elif typ == 'emp':
        pdf = par_list
        pi0 = np.max((0, np.nanmin(pdf)))
    return pi0

def get_pdf_multivariate_mbm(dat, a, b, w, d, k):
    """Returns the PDF values at given data points for a multivariate
    multi-parameter beta distribution model (MBM) with the given
    parameters.

    Parameters
    ----------
    dat : numpy array
        A one-dimensional numpy array with points where the PDF shall be
        evaluated.
    a : numpy array
        A k x d numpy array with the first beta shape parameter.
    b : numpy array
        A k x d numpy array with the second beta shape parameter. We use b = 1
        in the present works.
    w : numpy array
        A one-dimensional (length k) numpy array with the weights for each
        multivariate component
    d : int
        The multivariate dimension.
    k : int
        The number of mixture components.

    Returns
    -------
    numpy array
        The PDF values evaluated at the data in dat.
    """
    pdf = np.zeros((k, d, dat.size))
    use_k_idc = is_k_ok(w)
    use_d_idc = np.ones(a.shape)
    use_d_idc[~use_k_idc] = 0
    # Segmentation fault may occur for super small p-values below ~1e-305.
    dat[np.where(dat<1e-305)] = 1e-305
    for k_idx in np.arange(0, k, 1):
        if use_k_idc[k_idx]:
            for d_idx in np.arange(0, d, 1):
                use_d_idc[k_idx, d_idx] = is_d_ok(
                    a[k_idx, d_idx], b[k_idx, d_idx])
                if use_d_idc[k_idx, d_idx]:
                    pdf[k_idx, d_idx] = (stats.beta.pdf(
                        dat, a[k_idx, d_idx],
                        b[k_idx, d_idx]))
                else:
                    pdf[k_idx, d_idx, :] = np.nan
        else:
            pdf[k_idx, :, :] = np.nan
            use_d_idc[k_idx, :] = 0
    use_k_idc[np.where(use_k_idc)[0]] = np.sum(
        use_d_idc[np.where(use_k_idc)[0], :], 1) > 0
    pdf_av = np.zeros((d, dat.size))
    w_tmp = np.copy(w)
    w_tmp[~use_k_idc] = 0
    w_tmp = w_tmp/np.sum(w_tmp)
    for k_idx in np.where(use_k_idc)[0]:
        pdf_av[np.where(
            use_d_idc[k_idx, :])[0], :] = (pdf_av[np.where(
                use_d_idc[k_idx, :])[0], :] + w_tmp[k_idx] *
                1/np.sum(use_d_idc[k_idx, :]) * pdf[k_idx, np.where(
                    use_d_idc[k_idx, :])[0], :])
    # pdf_av = np.mean(pdf_av, axis=0)
    pdf_av = np.sum(pdf_av, axis=0)
    return pdf_av

def is_k_ok(w_hat):
    """Returns which components of the multivariate beta mixture model can be
    used.

    Parameters
    ----------
    w_hat : numpy array
        The weights for each mixture component.

    Returns
    -------
    numpy array
        A one-dimensional numpy array with indicators if a mixture component is
        save to use or not. If a weight of a mixture component is negative, it
        shall not be used.
    """
    use_k_idc = np.all([w_hat >= 0], axis=0)
    return use_k_idc

def is_d_ok(a_run, b_run):
    """Returns which multivariate entries can be used

    Parameters
    ----------
    a_run : numpy array
        The values of the first beta shape parameters to be checked for
        plausibility.
    b_run : _type_
        The values of the second beta shape parameters to be checked for
        plausibility.

    Returns
    -------
    boolean
        If the given values of the beta shape parameters are ok.
    """
    a_int_max = 10
    a_int_min = 0
    b_int_max = 10
    b_int_min = 0
    return (a_int_max >= a_run and
            a_int_min <= a_run and
            b_int_max >= b_run and
            b_int_min <= b_run)

def get_cdf_multivariate_mbm(dat, a, b, w, d, k):
    """Returns the CDF values at given data points for a multivariate
    multi-parameter beta distribution model (MBM) with the given
    parameters.

    Parameters
    ----------
    dat : numpy array
        A one-dimensional numpy array with points where the CDF shall be
        evaluated.
    a : numpy array
        A k x d numpy array with the first beta shape parameter.
    b : numpy array
        A k x d numpy array with the second beta shape parameter. We use b = 1
        in the present works.
    w : numpy array
        A one-dimensional (length k) numpy array with the weights for each
        multivariate component
    d : int
        The multivariate dimension.
    k : int
        The number of mixture components.

    Returns
    -------
    numpy array
        The CDF values evaluated at the data in dat.
    """
    cdf = np.zeros((k, d, dat.size))
    use_k_idc = is_k_ok(w)
    use_d_idc = np.ones(a.shape)
    use_d_idc[~use_k_idc] = 0
    for k_idx in np.arange(0, k, 1):
        if use_k_idc[k_idx]:
            for d_idx in np.arange(0, d, 1):
                use_d_idc[k_idx, d_idx] = is_d_ok(
                    a[k_idx, d_idx], b[k_idx, d_idx])
                if use_d_idc[k_idx, d_idx]:
                    cdf[k_idx, d_idx, :] = (stats.beta.cdf(
                            dat, a[k_idx, d_idx], b[k_idx, d_idx]))
                else:
                    cdf[k_idx, d_idx, :] = np.nan
        else:
            cdf[k_idx, :, :] = np.nan
            use_d_idc[k_idx, :] = 0
    use_k_idc[np.where(use_k_idc)[0]] = np.sum(
        use_d_idc[np.where(use_k_idc)[0], :], 1)>0
    cdf_av = np.zeros((d, dat.size))
    w_tmp = np.copy(w)
    w_tmp[~use_k_idc] = 0
    w_tmp = w_tmp/np.sum(w_tmp)
    for k_idx in np.where(use_k_idc)[0]:
        cdf_av[np.where(
            use_d_idc[k_idx, :])[0], :] = (cdf_av[np.where(
            use_d_idc[k_idx, :])[0], :]  + w_tmp[k_idx] *
                1/np.sum(use_d_idc[k_idx, :]) * cdf[k_idx, np.where(
            use_d_idc[k_idx, :])[0], :])
    cdf_av = np.sum(cdf_av, axis=0)
    return cdf_av