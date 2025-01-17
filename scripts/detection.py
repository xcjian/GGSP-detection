from para_estimation import *
from utils import *
from scipy import stats
import pandas as pd
import numpy as np


def det_lfdr(alp_levels, lfdr, h_true):
    """
    Detect the alternative hypothesis using the lfdr.

    Parameters
    ----------
    alp_levels : numpy array
        The nominal significance levels.
    lfdr : numpy array
        The estimated lfdr.
    h_true : numpy array
        The true alternative hypothesis. 1 for alternative and 0 for null.

    Returns
    -------
    h_est : list of numpy arrays
        The estimated alternative hypothesis under different nominal levels. 1 for alternative and 0 for null.
    FDR : numpy array
        The false discovery rates under different nominal levels.
    pow : numpy array
        The power under different nominal levels.
    """

    # take the maximum set of lfdr so that the average FDR is controlled at the desired level
    h_est = []
    FDR = []
    pow = []

    # change nan values in lfdr to 1.
    lfdr[np.where(np.isnan(lfdr) == True)] = 1 # This will not happen in the most updated implementation of MHT-GGSP.
    # generate an array so that the i-th element is the average of the smallest i elements of lfdr
    sort_idx = np.argsort(lfdr)
    # lfdr_sorted = lfdr[sort_idx]
    lfdr_sorted = np.array([lfdr[sort_idx[i]] for i in range(len(lfdr))])
    # lfdr_sortavg = np.cumsum(lfdr_sorted) / np.arange(1, len(lfdr_sorted) + 1)
    lfdr_sortavg = np.array([lfdr_sorted[:i + 1].mean() for i in range(len(lfdr))])
    for alp in alp_levels:
        rej_idx = np.where(lfdr_sortavg <= alp)[0]
        h_est_curr = np.zeros(lfdr.shape)
        # set the threshold to be the largest element in lfdr_sortavg that is less than or equal to the desired level
        if len(rej_idx) != 0:
            h_est_curr[sort_idx[rej_idx]] = 1
        h_est.append(h_est_curr)
        FDR.append(np.sum(h_est[-1] * (1 - h_true)) / max(np.sum(h_est[-1]), 1))
        pow.append(np.sum(h_est[-1] * h_true) / max(np.sum(h_true), 1))

    return h_est, FDR, pow

def det_BH(alp_levels, p_values, h_true, sav_path, sav_res):
    """
    Detect the alternative hypothesis using the Benjamini-Hochberg method.
    Parameters
    ----------
    alp_levels : numpy array
        The significance levels.
    p_values : numpy array
        The p-values.
    h_true : numpy array
        The true alternative hypothesis. 1 for alternative and 0 for null.

    Returns
    -------
    h_est : list of numpy arrays
        The estimated alternative hypothesis. 1 for alternative and 0 for null.
    FDR : numpy array
        The false discovery rate.
    pow : numpy array
        The power.
    """
    try:
        res = pd.read_pickle(sav_path)
        h_est = res['h_est']
        FDR = res['FDR']
        pow = res['power']
        print("BH loaded!")
    except FileNotFoundError:
        print("No results found for BH, computing ...", end="")

        n_alp = len(alp_levels)
        h_est = []
        FDR = []
        pow = []

        for alp_idx in range(n_alp):
            alpha = alp_levels[alp_idx]
            h_est_alp = BH_method(p_values, alpha)
            h_est.append(h_est_alp)
            FDR.append(np.sum(h_est_alp * (1 - h_true)) / max(np.sum(h_est_alp), 1))
            pow.append(np.sum(h_est_alp * h_true) / max(np.sum(h_true), 1))

        if sav_res:
            res = pd.DataFrame(
                {"h_est": h_est,
                 "FDR": FDR,
                 "power": pow
                 })
            res.to_pickle(sav_path)
            del res
        print("BH completed!")

    return h_est, FDR, pow

def det_sabha(alp_levels, p_values, q_weights, tau, h_true, sav_path, sav_res):
    """
    Detect the alternative hypothesis using the SABHA method.
    """

    try:
        res = pd.read_pickle(sav_path)
        h_est = res['h_est']
        FDR = res['FDR']
        pow = res['power']
        print("SABHA loaded!")
    except FileNotFoundError:
        print("No results found for SABHA, computing ...", end="")
        n_alp = len(alp_levels)
        h_est = []
        FDR = []
        pow = []

        for alp_idx in range(n_alp):
            alpha = alp_levels[alp_idx]
            h_est_alp = sabha(p_values, q_weights, tau, alpha)
            h_est.append(h_est_alp)
            FDR.append(np.sum(h_est_alp * (1 - h_true)) / max(np.sum(h_est_alp), 1))
            pow.append(np.sum(h_est_alp * h_true) / max(np.sum(h_true), 1))

        if sav_res:
            res = pd.DataFrame(
                {"h_est": h_est,
                 "FDR": FDR,
                 "power": pow
                 })
            res.to_pickle(sav_path)
            del res
        print("SABHA completed!")

    return h_est, FDR, pow

def det_GGSP_adapt(data, alp_levels, config, sav_path, sav_res):
    """
    Combination of the GGSP and AdaPT.
    """

    # --- Load data ---

    p_value = data['p_val']
    x_aux = data['sample_points'] # auxiliary information, in the form of (time, vertex)
    h_true = data['h_true']
    graph_basis = data['graph_basis']
    time_max_idx = data['time_max_idx']

    unit = config['unit']
    graph_bw_ran = config['graph_bw_ran']
    time_bw_ran = config['time_bw_ran']

    alp_levels = np.sort(alp_levels)

    # --- Initialize ---

    h_est = []
    FDR = []
    pow = []

    s_t = 0.45
    max_iter = p_value.size

    try:
        with open(sav_path, 'rb') as f:
            res = pickle.load(f)

        h_est = res['h_est']
        FDR = res['FDR']
        pow = res['power']
        print("GGSP-AdaPT loaded!")
    except FileNotFoundError:
        print("No results found for GGSP-AdaPT, computing ...", end="")

        # --- Iterates ---

        for t in range(max_iter):

            idx_revealed = np.where((s_t < p_value) & (p_value < (1 - s_t)))[0]
            idx_masked = np.where((p_value >= 1 - s_t) | (p_value <= s_t))[0]

            x_revealed = x_aux[idx_revealed, :]
            x_masked = x_aux[idx_masked, :]

            p_val_revealed = p_value[idx_revealed]
            p_val_masked = p_value[idx_masked]
            p_val_prime = np.minimum(p_val_masked, 1 - p_val_masked)

            R_t = np.sum(p_value <= s_t)  # number of p-values <= s_t
            A_t = np.sum(p_value >= 1 - s_t)  # number of p-values >= 1 - s_t
            FDP_hat_t = (1 + A_t) / np.maximum(1, R_t)  # FDP estimate

            alp_achieved = alp_levels[FDP_hat_t <= alp_levels]  # significance levels achieved

            if len(alp_achieved) > 0:
                for alp in alp_achieved:
                    h_est_alp = np.where(p_value <= s_t, 1, 0)
                    h_est.append(h_est_alp)
                    FDR.append(np.sum(h_est_alp * (1 - h_true)) / max(np.sum(h_est_alp), 1))
                    pow.append(np.sum(h_est_alp * h_true) / max(np.sum(h_true), 1))

                # delete the achieved significance levels
                alp_levels = alp_levels[FDP_hat_t > alp_levels]

            # stopping condition
            # all significance levels are achieved
            if len(alp_levels) == 0:
                break

            # arrive at the maximum iteration
            if t == max_iter - 1:
                for alp in alp_levels:
                    h_est_alp = np.where(p_value <= s_t, 1, 0)
                    h_est.append(h_est_alp)
                    FDR.append(np.sum(h_est_alp * (1 - h_true)) / max(np.sum(h_est_alp), 1))
                    pow.append(np.sum(h_est_alp * h_true) / max(np.sum(h_true), 1))

            # -------update s_t--------

            # model update
            if t % unit == 0:

                data_for_model = {'x_revealed': x_revealed, 'x_prime': x_masked, 'sample_points': x_aux,
                                  'graph_basis': graph_basis, 'time_max_idx': time_max_idx,
                                  'p_val_revealed': p_val_revealed, 'p_val_prime': p_val_prime}

                if t == 0:
                    config_for_model = {'graph_bw_range': graph_bw_ran, 'time_bw_range': time_bw_ran}
                else:
                    graph_bw_best = [int(gft_bw_best)]
                    tft_bw_best = [int(tft_bw_best)]

                    config_for_model = {'graph_bw_range': graph_bw_best, 'time_bw_range': tft_bw_best}

                beta_params_revealed, beta_params_prime, gft_bw_best, tft_bw_best = GGSP_adapt_model_update(data_for_model, config_for_model)

                beta_params = np.zeros(p_value.size)
                beta_params[idx_revealed] = beta_params_revealed
                beta_params[idx_masked] = beta_params_prime

            # determine the constant c (such that at least reveal one more p-value)
            lfdr_masked = p_val_prime ** (1 - beta_params[idx_masked])
            c = np.max(lfdr_masked) - 10 ** (-15)

            # update s_t
            s_x_c = c ** (1 / (1 - beta_params))
            s_t = np.minimum(s_t, s_x_c)

        # Flip the results, because the results are stored in the reverse order
        FDR.reverse()
        pow.reverse()
        h_est.reverse()

        if sav_res:
            res = {"h_est": h_est,
                 "FDR": FDR,
                 "power": pow}
            with open(sav_path, 'wb') as f:
                pickle.dump(res, f)
            del res

    return h_est, FDR, pow

def est_lfdr_beta_ggsp(data, data_info, para_config, sav_path, sav_res):
    """Estimate the lfdrs using the MHT-GGSP model.

        Parameters
        ----------
        data : dict
            p_val : n-dimensional numpy array
            sample_points : n x 2 numpy array. Each row is the (time, vertex) index of the corresponding p-value.
            Both time and vertex index start from 0.
        data_info : dict
            The graph and time information.
            'graph_basis' : N X N numpy array. Graph Fourier basis, columns listed in the increasing order of frequency.
            'time_max_idx' : int. If the time index is 0,1, ..., T, then time_max_idx = T + 1.
        para_config : dict
            The configuration of the parameters during MLE computation.
            'bandwidths' : dict. The bandwidths for the graph and time dimensions.
                'graph_bw_ran': numpy array. The range of the graph bandwidth. e.g., np.arange(1, 10)
                'time_bw_ran': numpy array. The range of the time bandwidth. e.g., np.arange(1, 5)
            'nonlinear_type' : str. The type of the nonlinear function. In this paper we only use 'sigmoid'.
        res_path : str
            The path to where the results are to be stored
        sav_res : boolean
            If the results are to be saved

        Returns
        -------
        numpy array
            The estimated lfdrs and density values.
        """
    try:
        res = pd.read_pickle(sav_path)
        lfdr = res['lfdr_hat']
        f_p = res['f_p_hat']
        f1_p = res['f1_p_hat']
        pi0 = res['pi0_hat']
        est_time = res['est_time']
        print("MHT-GGSP loaded!")
    except FileNotFoundError:
        print("No results found for MHT-GGSP, computing ...", end="")

        p_val = data['p_val']
        sample_points = data['sample_points']
        graph_basis = data_info['graph_basis']
        time_max_idx = data_info['time_max_idx']
        bandwidths = para_config['bandwidths']
        nonlinear_type = para_config['nonlinear_type']

        start_time = time.time()
        f1_p, pi0, _ = est_paras_beta_ggsp(p_val, sample_points, graph_basis, time_max_idx, bandwidths, nonlinear_type)
        end_time = time.time()
        est_time = end_time - start_time

        f_p = pi0 * stats.uniform.pdf(p_val) + (1 - pi0) * f1_p
        lfdr = pi0 * stats.uniform.pdf(p_val) / f_p
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                {"f_p_hat": f_p,
                 "f1_p_hat": f1_p,
                 "lfdr_hat": lfdr,
                 "pi0_hat": pi0,
                    "est_time": est_time
                 })
            res.to_pickle(sav_path)
            del res
        print("MHT-GGSP completed!")
    return lfdr, f_p, f1_p, pi0, est_time

def est_lfdr_beta_ggsp_reg(data, para_config, sav_path, sav_res):
    """Estimate the lfdrs using MHT-GGSP_reg. This method is based on MHT-GGSP, but marginally regularized by Storey's pi0_hat.

        Parameters
        ----------
        data : dict
            p_val : n-dimensional numpy array
            pi0_ggsp : n-dimensional numpy array.
                This is the pi0_hat estimated by MHT-GGSP.
        para_config : dict
            The configuration of the parameters.
            'tau_ggsp' : float. The threshold used by Storey's method to estimate the proportion of null.
        res_path : str
            The path to where the results are to be stored
        sav_res : boolean
            If the results are to be saved

        Returns
        -------
        numpy array
            The estimated lfdrs and density values.
        """
    try:
        res = pd.read_pickle(sav_path)
        lfdr = res['lfdr_hat']
        f_p = res['f_p_hat']
        f1_p = res['f1_p_hat']
        pi0 = res['pi0_hat']
        est_time = res['est_time']
        print("MHT-GGSP_reg loaded!")
    except FileNotFoundError:
        print("No results found for MHT-GGSP_reg, computing ...", end="")

        p_val = data['p_val']
        tau = para_config['tau_ggsp']

        start_time = time.time()
        pi0 = data['pi0_ggsp']
        pi0_avg = np.sum(p_val > tau) / p_val.size / (1 - tau) # storey's pi0
        # adjust the pi0_hat (equivalently the beta_paras)
        pi0 = pi0 / np.mean(pi0) * pi0_avg
        pi0[pi0 >= 1] = 1 # should not exceed 1
        f1_p = (p_val ** (pi0 - 1) - 1) * pi0 / (1 - pi0)
        ## consider the case where beta_paras = 1
        beta_paras_one_idx = np.where(pi0 == 1)[0]
        f1_p[beta_paras_one_idx] = -np.log(p_val[beta_paras_one_idx])

        end_time = time.time()
        est_time = end_time - start_time

        f_p = pi0 * stats.uniform.pdf(p_val) + (1 - pi0) * f1_p
        lfdr = pi0 * stats.uniform.pdf(p_val) / f_p
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                {"f_p_hat": f_p,
                 "f1_p_hat": f1_p,
                 "lfdr_hat": lfdr,
                 "pi0_hat": pi0,
                    "est_time": est_time
                 })
            res.to_pickle(sav_path)
            del res
        print("MHT-GGSP_reg completed!")
    return lfdr, f_p, f1_p, pi0, est_time

def est_lfdr_beta_ggsp_margin(data, data_info, para_config, sav_path, sav_res):
    """Estimate the lfdrs using the Beta-GGSP contextual model, regulated by storey's pi0_hat.

        Parameters
        ----------
        data : dict
            p_val : numpy array
            sample_points : numpy array
        data_info : dict
            The information of the data.
        para_config : dict
            The configuration of the parameters.
        res_path : str
            The path to where the results are to be stored
        sav_res : boolean
            If the results are to be saved

        Returns
        -------
        numpy array
            The estimated lfdrs.
        """
    try:
        res = pd.read_pickle(sav_path)
        lfdr = res['lfdr_hat']
        f_p = res['f_p_hat']
        f1_p = res['f1_p_hat']
        pi0 = res['pi0_hat']
        est_time = res['est_time']
        print("Beta-GGSP loaded!")
    except FileNotFoundError:
        print("No results found for Beta-GGSP, computing ...", end="")

        p_val = data['p_val']
        null_pdf = para_config['null_pdf']

        if null_pdf == 'uniform':
            args = {}
        elif null_pdf == 'var_compare':
            args = {'var_test': para_config['var_test'], 'var_null': para_config['var_null'], 'df': para_config['df']}
        elif null_pdf == 'alpha': # suppose that the null p-value pdf is alpha * p ^ (alpha - 1).
            tau = para_config['tau_ggsp']
            null_alpha = null_alpha_estimate(p_val, tau)
            args = {'null_alpha': null_alpha}

        start_time = time.time()
        beta_paras = data['pi0_ggsp']
        pi0 = np.mean(beta_paras)
        f_p = np.zeros(p_val.size)
        f1_p = np.zeros(p_val.size) # will not be used
        for i in range(p_val.size):
            p_val_curr = p_val[i]
            f_p[i] = pi0 + np.mean((1 - beta_paras) * (p_val_curr ** (beta_paras - 1) - 1) * beta_paras / (1 - beta_paras))

        end_time = time.time()
        est_time = end_time - start_time

        lfdr = pi0 * stats.uniform.pdf(p_val) / f_p
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                {"f_p_hat": f_p,
                 "f1_p_hat": f1_p,
                 "lfdr_hat": lfdr,
                 "pi0_hat": pi0,
                    "est_time": est_time
                 })
            res.to_pickle(sav_path)
            del res
        print("Beta-GGSP completed!")
    return lfdr, f_p, f1_p, pi0, est_time

def est_lfdr_beta_ggsp_cens(data, data_info, para_config, sav_path, sav_res):
    """Estimate the lfdrs using the MHT-GGSP_cens method.

        Parameters
        ----------
        data : dict
            p_val : numpy array
            sample_points : numpy array
        data_info : dict
            The information of the data.
        para_config : dict
            The configuration of the parameters.
        res_path : str
            The path to where the results are to be stored
        sav_res : boolean
            If the results are to be saved

        Returns
        -------
        numpy array
            The estimated lfdrs.
        """
    try:
        res = pd.read_pickle(sav_path)
        lfdr = res['lfdr_hat']
        f_p = res['f_p_hat']
        f1_p = res['f1_p_hat']
        pi0 = res['pi0_hat']
        est_time = res['est_time']
        print("MHT-GGSP_cens loaded!")
    except FileNotFoundError:
        print("No results found for MHT-GGSP_cens, computing ...", end="")

        p_val = data['p_val']
        sample_points = data['sample_points']
        graph_basis = data_info['graph_basis']
        time_max_idx = data_info['time_max_idx']
        bandwidths = para_config['bandwidths']
        nonlinear_type = para_config['nonlinear_type']

        # step1: separate out the non-zero p-values
        cens_threshold = 10 ** (-4)
        cens_indices = np.where(p_val <= cens_threshold)[0]
        noncens_indices = np.where(p_val > cens_threshold)[0]
        sample_points_noncens = sample_points[np.where(p_val > cens_threshold), :][0, :, :]
        p_val_noncens = p_val[np.where(p_val > cens_threshold)]
        proportion_cens = cens_indices.size / p_val.size

        start_time = time.time()
        f1_p_noncens, pi0_noncens, pi0_cont_all = est_paras_beta_ggsp(p_val_noncens, sample_points_noncens, graph_basis, time_max_idx, bandwidths, nonlinear_type)
        end_time = time.time()
        est_time = end_time - start_time

        # compute and correct pi0 for all p-values

        pi0 = np.zeros(p_val.size)
        pi0[noncens_indices] = pi0_noncens
        pi0_censored = cens_threshold / proportion_cens
        pi0[cens_indices] = pi0_censored
        # storey's correction for pi0
        pi0_avg = np.sum(p_val > 0.5) / p_val.size / (1 - 0.5)
        pi0 = pi0 / np.mean(pi0) * pi0_avg
        pi0[pi0 >= 1] = 1 # should not exceed 1
        beta_paras = pi0

        # correct f1_p_noncens

        beta_paras_nonzero = beta_paras[noncens_indices]
        f1_p_noncens = (p_val_noncens ** (beta_paras_nonzero - 1) - 1) * beta_paras_nonzero / (1 - beta_paras_nonzero)
        ## consider the case where beta_paras = 1
        beta_paras_one_idx = np.where(beta_paras_nonzero == 1)[0]
        f1_p_noncens[beta_paras_one_idx] = -np.log(p_val_noncens[beta_paras_one_idx])

        # compute the f1_p of p-values
        # Note: proportion_cens = (1-pi_{0,i}) * w_i.
        f1_p = np.zeros(p_val.size)

        ## calculate the f1_p for the zero p-values
        ## Note: if we set lfdr as pi0_censored, then we do not need to do this.
        F_p_zero = proportion_cens
        F1_p_nonzero = (cens_threshold ** (beta_paras_nonzero) - cens_threshold * beta_paras_nonzero) / (1 - beta_paras_nonzero)
        F1_p_nonzero[beta_paras_one_idx] = cens_threshold * (1 - np.log(cens_threshold))
        F1_p_zero = (F_p_zero - np.mean(pi0) * cens_threshold - np.sum((1 - pi0[noncens_indices]) * F1_p_nonzero) / p_val.size) / (np.sum(1 - pi0[cens_indices]) / p_val.size)
        F1_p_zero = min(F1_p_zero, 1)
        # F1_p_zero = (proportion_cens - pi0_censored * cens_threshold) / (1 - pi0_censored) / proportion_cens
        f1_p_zero = F1_p_zero / cens_threshold

        f1_p[noncens_indices] = f1_p_noncens
        f1_p[cens_indices] = f1_p_zero
        # f1_p[cens_indices] = 0.5 # a dummy value, not used.


        f_p = pi0 * stats.uniform.pdf(p_val) + (1 - pi0) * f1_p
        lfdr = pi0 * stats.uniform.pdf(p_val) / f_p
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                {"f_p_hat": f_p,
                 "f1_p_hat": f1_p,
                 "lfdr_hat": lfdr,
                 "pi0_hat": pi0,
                    "est_time": est_time
                 })
            res.to_pickle(sav_path)
            del res
        print("MHT-GGSP_cens completed!")
    return lfdr, f_p, f1_p, pi0, est_time



def est_paras_ggsp_cens(p_val, censor_threshold, sample_points, graph_basis, time_max_idx, bandwidths, nonlinear_type, null_pdf, **args):
    """
    Estimate the parameters for the censoring method.
    """

    # step1: separate out the non-zero p-values
    censored_indices = np.where(p_val <= censor_threshold)[0]
    noncensored_indices = np.where(p_val > censor_threshold)[0]
    sample_points_noncensored = sample_points[noncensored_indices, :]
    p_val_noncensored = p_val[noncensored_indices]
    proportion_censored = censored_indices.size / p_val.size

    # step2: estimate the parameters for the non-censored p-values
    start_time = time.time()
    f1_p, pi0_non_cens, pi0_cont_all = est_paras_beta_ggsp(p_val_noncensored, sample_points_noncensored,
                                                                                   graph_basis, time_max_idx,
                                                                                   bandwidths, nonlinear_type, null_pdf,
                                                                                   **args)
    # beta_paras = pi0_cont_all
    end_time = time.time()
    est_time = end_time - start_time

    # step3: compute and correct parameters for all p-values
    # pi0:
    pi0 = np.zeros(p_val.size)
    pi0[noncensored_indices] = pi0_non_cens
    pi0_censored = censor_threshold / proportion_censored
    pi0[censored_indices] = pi0_censored
    # storey's pi0
    pi0_avg = np.sum(p_val > 0.5) / p_val.size / (1 - 0.5)
    pi0 = pi0 / np.mean(pi0) * pi0_avg
    pi0[pi0 >= 1] = 1  # should not exceed 1
    beta_paras = pi0
    # f1_p:
    f1_p = np.zeros(p_val.size)
    beta_paras_noncensored = beta_paras[noncensored_indices]
    f1_p_noncensored = (p_val_noncensored ** (beta_paras_noncensored - 1) - 1) * beta_paras_noncensored / (
            1 - beta_paras_noncensored)
    ## consider the case where beta_paras = 1
    beta_paras_one_idx = np.where(beta_paras_noncensored == 1)[0]
    f1_p_noncensored[beta_paras_one_idx] = -np.log(p_val_noncensored[beta_paras_one_idx])

    F_p_censored = proportion_censored
    F1_p_noncensored = (censor_threshold ** (
        beta_paras_noncensored) - censor_threshold * beta_paras_noncensored) / (
                               1 - beta_paras_noncensored)
    F1_p_noncensored[beta_paras_one_idx] = censor_threshold * (1 - np.log(censor_threshold))
    F1_p_censored = (F_p_censored - np.mean(pi0) * censor_threshold - np.sum(
        (1 - pi0[noncensored_indices]) * F1_p_noncensored) / p_val.size) / (
                                np.sum(1 - pi0[censored_indices]) / p_val.size)
    F1_p_censored = min(F1_p_censored, 1)
    f1_p_censored = F1_p_censored / censor_threshold

    f1_p[noncensored_indices] = f1_p_noncensored
    f1_p[censored_indices] = f1_p_censored

    '''
    # Theoretical CDF:
    Theo_CDF = np.zeros(p_val.size)
    for i, p in enumerate(p_val):
        mix_alt_part = np.zeros(p_val.size)
        mix_alt_part[noncensored_indices] = (p ** (beta_paras_noncensored) - p * beta_paras_noncensored) / (
                1 - beta_paras_noncensored)
        mix_alt_part[noncensored_indices[beta_paras_one_idx]] = p * (1 - np.log(p))
        if p <= censor_threshold:
            mix_alt_part[censored_indices] = f1_p_censored * p
        else:
            mix_alt_part[censored_indices] = F1_p_censored + (p - censor_threshold) * (1 - F1_p_censored) / (
                        1 - censor_threshold)
        mix_alt_part = mix_alt_part * (1 - pi0)
        Theo_CDF[i] = np.mean(pi0) * p + np.mean(mix_alt_part)
    '''

    return f1_p, pi0, pi0_cont_all, est_time

def est_lfdr_smom(data, res_path, sav_res, par_lst, partition):
    """Estimate the lfdrs using lfdr-sMoM.

    Details: See [Goelz2022TSIPN].

    Parameters
    ----------
    data: dict
        p_val : numpy array
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par_lst : list
        The additional parameters.
    partition : str
        The partition to be used for finding the p-value vectors.

    Returns
    -------
    list
        A list with the estimated lfdrs, f_p, f1_p, pi0 and the execution time.
    """
    try:
        if partition == 'spatial':
            res = pd.read_pickle(res_path)
        elif partition == 'random':
            res = pd.read_pickle(res_path)
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        if partition == 'spatial':
            print("sMoM_s loaded!")
        elif partition == 'random':
            print("sMoM loaded!")
    except FileNotFoundError:
        [dat_path, max_wrk, par_type, quant_bits, sensoring_thr] = par_lst
        if partition == 'spatial':
            print("No results found for lfdr-sMoM_s, computing ...")
        elif partition == 'random':
            print("No results found for lfdr-sMoM, computing ...")

        p_val = data['p_val']
        n_nodes = p_val.shape[0]
        # ------ Applying smom -------
        # Uncomment for for loop, useful for debugging
        #(f1_p, pi0, smom_lam, smom_a, f_p, F_p, diff_edf_est_cdf, sel_k,
        #   sel_d, ex_time) = for_loop_smom(p_val, par_type, dat_path, partition,
        #                           quant_bits, sensoring_thr)
        # Uncomment for parallelization
        (f1_p, pi0, smom_lam, smom_a, f_p, F_p, diff_edf_est_cdf, sel_k, sel_d,
          ex_time) = parallel_smom(p_val, par_type, dat_path, max_wrk, partition,
                                  quant_bits, sensoring_thr)

        # ------ Applying smom -------
        f1_p[np.where(np.isnan(f1_p))] = np.inf

        f_p = (np.transpose(np.tile(pi0, (n_nodes, 1)))
               + (1-np.transpose(np.tile(pi0, (n_nodes, 1)))) * f1_p)

        lfdr = (np.transpose(np.tile(pi0, (n_nodes, 1))))/f_p
        lfdr[np.where(np.isnan(lfdr))] = 0
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "a_k_hat": [smom_a],
                     "sel_k": [sel_k],
                     "sel_d": [sel_d],
                     "pi_k_hat": [smom_lam],
                     "diff_edf_cdf": [diff_edf_est_cdf],
                     "F_p_hat": [F_p],
                     "ex_time": [ex_time]
                     })
            if partition == 'spatial':
                res.to_pickle(res_path)
            elif partition == 'random':
                res.to_pickle(res_path)
            del res

    lfdr = lfdr.flatten()
    f_p = f_p.flatten()
    f1_p = f1_p.flatten()
    pi0 = pi0.flatten()

    return lfdr, f_p, f1_p, pi0, ex_time

def est_lfdr_smom_gtpi0(data, res_path, sav_res, par_lst, partition):
    """Estimate the lfdrs using lfdr-sMoM.

    Details: See [Goelz2022TSIPN].

    Parameters
    ----------
    data: dict
        p_val : numpy array
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par_lst : list
        The additional parameters.
    partition : str
        The partition to be used for finding the p-value vectors.

    Returns
    -------
    list
        A list with the estimated lfdrs, f_p, f1_p, pi0 and the execution time.
    """
    try:
        if partition == 'spatial':
            res = pd.read_pickle(res_path)
        elif partition == 'random':
            res = pd.read_pickle(res_path)
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        if partition == 'spatial':
            print("sMoM_s loaded!")
        elif partition == 'random':
            print("sMoM loaded!")
    except FileNotFoundError:
        [dat_path, max_wrk, par_type, quant_bits, sensoring_thr] = par_lst
        if partition == 'spatial':
            print("No results found for lfdr-sMoM_s, computing ...")
        elif partition == 'random':
            print("No results found for lfdr-sMoM, computing ...")

        p_val = data['p_val']
        pi0 = data['pi0_gt']
        f_p = data['f_p']
        f1_p = data['f1_p']

        n_nodes = p_val.shape[0]
        # ------ Applying smom -------
        # use the result of smom to estimate fp


        lfdr = pi0/(pi0 + (1-pi0)*f1_p)
        # lfdr = pi0 / f_p
        lfdr[np.where(np.isnan(lfdr))] = 0
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     })
            if partition == 'spatial':
                res.to_pickle(res_path)
            elif partition == 'random':
                res.to_pickle(res_path)
            del res

    lfdr = lfdr.flatten()
    f_p = f_p.flatten()
    f1_p = f1_p.flatten()
    pi0 = pi0.flatten()

    return lfdr, f_p, f1_p, pi0

def proportion_matching(data, tau, alp_levels, h_true, res_path, sav_res):
    """
    Proportion matching method.

    Parameters
    ----------
    data : dict
        p_val : numpy array
        sample_points : numpy array. The first column is the time index and the second column is the vertex index.
    tau : dict
        The information of the data.
    alp_levels : dict
        The significance levels.

    Returns
    -------
    numpy array
        The estimated lfdrs.
    """

    try:
        res = pd.read_pickle(res_path)
        h_est = res['h_est']
        FDR = res['FDR']
        power = res['power']
        print("Proportion matching loaded!")
    except FileNotFoundError:
        print("No results found for proportion matching, computing ...", end="")

        # load the data.
        p_val = data['p_val']
        sample_points = data['sample_points']
        n_vertex = np.unique(sample_points[:, 1]).shape[0]

        # estimate r0 on each vertex.
        for i in range(n_vertex):
            idx = np.where(sample_points[:, 1] == i)[0]
            p_val_i = p_val[idx]
            r0_i_hat = np.minimum((np.sum(p_val_i > tau) / p_val_i.shape[0]) / (1 - tau), 1)
            m_i = len(idx)
            if i == 0:
                r0_i_hat_vec = r0_i_hat
                m_i_vec = m_i
            else:
                r0_i_hat_vec = np.vstack((r0_i_hat_vec, r0_i_hat))
                m_i_vec = np.vstack((m_i_vec, m_i))

        # estimate the overall r0.
        r0_hat = np.sum(r0_i_hat_vec * m_i_vec) / np.sum(m_i_vec)

        # estimate the global slope.
        beta_hat = ((1 / alp_levels) - r0_hat) / (1 - r0_hat)

        # derive the local BH significance levels.
        alp_corrected = 1 / ((1 - r0_i_hat_vec) * beta_hat + r0_i_hat_vec)
        # the (i,j)-th element is the corrected significance level under the j-th nominal level on the i-th vertex.

        # perform BH locally.
        n_alp = len(alp_levels)
        h_est = []
        for alp_idx in range(n_alp):
            h_est_alp = np.zeros(p_val.shape[0])
            for i in range(n_vertex):
                alpha = alp_corrected[i, alp_idx]
                idx = np.where(sample_points[:, 1] == i)[0]
                p_val_i = p_val[idx]
                h_est_i = BH_method(p_val_i, alpha)
                h_est_alp[idx] = h_est_i
            h_est.append(h_est_alp)

        # compute the FDR and power.
        FDR = []
        power = []

        for alp_idx in range(n_alp):
            h_est_alp = h_est[alp_idx]
            FDR.append(np.sum(h_est_alp * (1 - h_true)) / max(np.sum(h_est_alp), 1))
            power.append(np.sum(h_est_alp * h_true) / max(np.sum(h_true), 1))

        if sav_res:
            res = pd.DataFrame(
                {"h_est": h_est,
                 "FDR": FDR,
                 "power": power
                 })
            res.to_pickle(res_path)
            del res
        print("Proportion matching completed!")

    return h_est, FDR, power
def BH_method(p_vals, alp_level):
    """
    Benjamini-Hochberg method.

    Parameters
    ----------
    p_vals : numpy array
        The p-values.
    alp_level : numpy array
        The significance level.

    Returns
    -------
    numpy array
        The estimated hypotheses.
    """
    n = len(p_vals)
    p_vals_sorted = np.sort(p_vals)
    idx = np.argsort(p_vals)
    h_est = np.zeros(n)

    p_lower_idx = np.where(p_vals_sorted <= alp_level * (np.arange(1, n + 1) / n))
    if len(p_lower_idx[0]) != 0:
        p_threshold = p_vals_sorted[np.max(p_lower_idx)]
        h_est[np.where(p_vals <= p_threshold)] = 1

    return h_est

def sabha(p_vals, q_weights, tau, alp_level):
    """
    SABHA method.

    Parameters
    ----------
    p_vals : numpy array
        The p-values.
    q_weights : numpy array
        The weights.
    tau : cut-off value
    alp_level : numpy array
        The significance level.

    Returns
    -------
    numpy array
        The estimated hypotheses.
    """
    n = len(p_vals)

    # The following codes are translated from R code.

    p_vals[p_vals > tau] = np.inf
    # khat < - max(c(0, which(sort( as.matrix(qhat * pvals)) <= alpha * (1:length(pvals)) / length(pvals))))
    khat = np.max([0, np.max(np.where(np.sort(q_weights * p_vals) <= alp_level * (np.arange(1, n + 1) / n)))])

    # which(qhat*pvals<=alpha*khat/length(pvals))
    h_est = np.zeros(n)
    h_est[np.where(q_weights * p_vals <= alp_level * khat / n)] = 1

    return h_est