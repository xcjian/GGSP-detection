import argparse
import numpy as np
import pickletools

from utils import *
from detection import *
import parameters as par
import networkx as nx
import matplotlib.pyplot as plt

# -------------------
# pass the parameters to the script
# -------------------

# Initialize the parser
parser = argparse.ArgumentParser()

# Add the dataset argument with choices
parser.add_argument('--dataset', type=str, default='communication', choices=['communication'],
                    help='The dataset to be used for the experiments.')
parser.add_argument('--vary_snr', action='store_true', default=True,
                    help='vary the SNR levels')
parser.add_argument('--no_vary_snr', action='store_false', dest='vary_snr',
                    help='Do not vary the SNR levels')
parser.add_argument('--repeat_time', type=int, default=20,
                    help='The number of times to repeat the experiments.')
parser.add_argument('--alp_levels', type=float, nargs='+', default=[0.02, 0.05, 0.07] + list(np.linspace(0.10, 1, 19)),
                    help='The nominal FDR levels.')
parser.add_argument('--sav_res', action='store_true', default=True,
                    help='Save the results.')
parser.add_argument('--no_sav_res', action='store_false', dest='sav_res',
                    help='Do not save the results.')
parser.add_argument('--sav_plots', action='store_true', default=True,
                    help='Save the plots.')
parser.add_argument('--no_sav_plots', action='store_false', dest='sav_plots',
                    help='Do not save the plots.')
parser.add_argument('--instance_inspect', type=int, nargs='+', default=[0, 1, 2],
                    help='The instance to inspect the detection results.')
parser.add_argument('--combinatorial', action='store_true', default=True,
                    help='Use combinatorial graph Laplacian.')
parser.add_argument('--no_combinatorial', action='store_false', dest='combinatorial',
                    help='Do not use combinatorial graph Laplacian.')
parser.add_argument('--method_names', type=str, nargs='+', default=['MHT-GGSP', 'MHT-GGSP_cens', 'MHT-GGSP_reg', 'lfdr-sMoM', 'Prop-matching', 'BH', 'FDR-smoothing','SABHA', 'AdaPT'],
                    help='The methods to be compared.')
parser.add_argument('--graph_bw_ran', type=int, nargs='+', default=np.arange(5, 30, 2).tolist(),
                    help="Range of graph bandwidth values.")
parser.add_argument('--time_bw_ran', type=int, nargs='+', default=np.arange(0, 6, 1).tolist(),
                    help="Range of time bandwidth values.")
parser.add_argument('--tau_ggsp', type=float, default=0.5,
                    help="The Storey threshold for GGSP method.")
parser.add_argument('--graph_bw_ran_cens', type=int, nargs='+', default=np.arange(1, 30, 2).tolist(),
                    help="Range of graph bandwidth values for MHT-GGSP_cens method.")
parser.add_argument('--time_bw_ran_cens', type=int, nargs='+', default=np.arange(0, 3, 1).tolist(),
                    help="Range of time bandwidth values for MHT-GGSP_cens method.")
parser.add_argument('--tau_pmatch', type=float, default=0.3,
                    help="The threshold for the proportion-matching method.")

# Parse the arguments
args = parser.parse_args()

# -------------------
# call dataset
# -------------------
dataset = args.dataset
vary_snr = args.vary_snr
repeat_time = args.repeat_time
alp_levels = np.array(args.alp_levels)
sav_res = args.sav_res
sav_plots = args.sav_plots
instance_inspect = args.instance_inspect
combinatorial = args.combinatorial

method_names = args.method_names

# Set the parameters for MHT-GGSP
graph_bw_ran = np.array(args.graph_bw_ran)
time_bw_ran = np.array(args.time_bw_ran)
tau_ggsp = args.tau_ggsp

# Set the parameters for MHT-GGSP_cens
graph_bw_ran_cens = np.array(args.graph_bw_ran_cens)
time_bw_ran_cens = np.array(args.time_bw_ran_cens)

# Set the parameters for the proportion-matching method
tau_pmatch = args.tau_pmatch

#-------------------
# Set configurations
#-------------------
config = get_config(dataset, vary_snr)
data_path = config['data_path']
res_path = config['res_path']

#Set plotting parameters
method_names_all = config['method_names']
line_styles = config['line_styles']
colors = config['colors']
markers = config['markers']

method_indices = [i for i, method in enumerate(method_names_all) if method in method_names]
method_names = [method_names_all[i] for i in method_indices]
line_styles = [line_styles[i] for i in method_indices]
colors = [colors[i] for i in method_indices]
markers = [markers[i] for i in method_indices]

# -------------------
FDR_vary_summary = {}
Power_vary_summary = {}
noise_levels = np.zeros(len(data_path))
for method in method_names:
    FDR_vary_summary[method] = []
    Power_vary_summary[method] = []

n_noise_levels = len(data_path)
for noise_idx in range(n_noise_levels):
    start_time = time.time()

    curr_data_path = data_path[noise_idx]
    curr_res_path = res_path[noise_idx]

    curr_noise_level = np.load(os.path.join(os.path.dirname(curr_data_path), 'noise_std.npy'))
    noise_levels[noise_idx] = curr_noise_level

    FDR_summary = {}
    Power_summary = {}
    for method in method_names:
        FDR_summary[method] = []
        Power_summary[method] = []

    selected_thresholds = np.zeros(repeat_time)

    # Run detection experiments
    for repeat_idx in range(repeat_time):
        print(f"Repeat time: {repeat_idx}")

        # load data
        with open(os.path.join(curr_data_path, f'repeat_{repeat_idx}.pkl'), 'rb') as f:
            data_dict = pickle.load(f)

        p_values = data_dict['p_values']
        graph_adj = data_dict['graph_adj']
        node_coords = data_dict['node_coords']
        hypotheses = data_dict['hypotheses']
        center_coords = data_dict['center_coords']

        if combinatorial:
            graph_adj[graph_adj > 0] = 1

        # generate mask
        mask = np.ones(p_values.shape, dtype=bool)
        mask_indices = np.nonzero(mask)
        p_values_obs = p_values[mask_indices]
        h_true = hypotheses[mask_indices]

        # -------------------
        # MHT-GGSP
        # -------------------
        if 'MHT-GGSP' in method_names:
            # obtain the graph Fourier basis
            n_vertex = graph_adj.shape[0]
            g_Lap = np.diag(graph_adj @ np.ones(n_vertex)) - graph_adj
            g_freq, g_fb = graph_spectral_decomp(g_Lap)

            sample_points = np.column_stack(mask_indices)
            dat_ggsp = {'p_val': p_values_obs, 'sample_points': sample_points}
            dat_info_ggsp = {'graph_basis': g_fb, 'time_max_idx': p_values.shape[0]}
            para_config_ggsp = {'bandwidths': {'graph_bw_ran': graph_bw_ran, 'time_bw_ran': time_bw_ran},
                                 'nonlinear_type': 'sigmoid'}

            # estimate the lfdrs
            filename_ggsp = 'MHT-GGSP' + f'_repeat_{repeat_idx}.pkl'
            sav_path = os.path.join(curr_res_path, filename_ggsp)
            lfdr_ggsp, _, f1_p_ggsp, pi0_ggsp, est_time_direct= est_lfdr_beta_ggsp(dat_ggsp, dat_info_ggsp, para_config_ggsp, sav_path, sav_res)
            # detect the hypotheses
            h_est_ggsp, FDR_ggsp, pow_ggsp = det_lfdr(alp_levels, lfdr_ggsp, h_true)

            # Record the results
            FDR_summary['MHT-GGSP'].append(FDR_ggsp)
            Power_summary['MHT-GGSP'].append(pow_ggsp)

            print(f"MHT-GGSP: FDR = {FDR_ggsp[0]}, Power = {pow_ggsp[0]}", f"Time = {est_time_direct}")

        # -------------------
        # MHT-GGSP_reg
        # -------------------
        if 'MHT-GGSP_reg' in method_names:

            # obtain the graph Fourier basis
            n_vertex = graph_adj.shape[0]
            g_Lap = np.diag(graph_adj @ np.ones(n_vertex)) - graph_adj
            g_freq, g_fb = graph_spectral_decomp(g_Lap)

            sample_points = np.column_stack(mask_indices)
            try:
                dat_ggsp_reg = {'p_val': p_values_obs, 'f1_p': f1_p_ggsp, 'pi0_ggsp': pi0_ggsp}
            except:
                raise ValueError('Please run MHT-GGSP first to get the f1_p and pi0_ggsp.')

            dat_info_ggsp_reg = {'graph_basis': g_fb, 'time_max_idx': p_values.shape[0]}
            para_config_ggsp_reg = {'bandwidths': {'graph_bw_ran': graph_bw_ran, 'time_bw_ran': time_bw_ran},
                                 'nonlinear_type': 'sigmoid', 'tau_ggsp': tau_ggsp}

            # estimate the lfdrs
            filename_ggsp_reg = 'MHT-GGSP_reg' + f'_repeat_{repeat_idx}.pkl'
            sav_path = os.path.join(curr_res_path, filename_ggsp_reg)
            lfdr_ggsp_reg, _, _, _, est_time_ggsp_reg= est_lfdr_beta_ggsp_reg(dat_ggsp_reg, para_config_ggsp_reg, sav_path, sav_res)
            # detect the hypotheses
            h_est_ggsp_reg, FDR_ggsp_reg, pow_ggsp_reg = det_lfdr(alp_levels, lfdr_ggsp_reg, h_true)

            # Record the results
            FDR_summary['MHT-GGSP_reg'].append(FDR_ggsp_reg)
            Power_summary['MHT-GGSP_reg'].append(pow_ggsp)

            print(f"MHT-GGSP_reg: FDR = {FDR_ggsp_reg[0]}, Power = {pow_ggsp_reg[0]}", f"Time = {est_time_ggsp_reg}")

        # -------------------
        # MHT-GGSP_cens
        # -------------------
        if 'MHT-GGSP_cens' in method_names:
            # obtain the graph Fourier basis
            n_vertex = graph_adj.shape[0]
            g_Lap = np.diag(graph_adj @ np.ones(n_vertex)) - graph_adj
            g_freq, g_fb = graph_spectral_decomp(g_Lap)

            sample_points = np.column_stack(mask_indices)
            dat_ggsp_cens = {'p_val': p_values_obs, 'sample_points': sample_points}
            dat_info_ggsp_cens = {'graph_basis': g_fb, 'time_max_idx': p_values.shape[0]}
            para_config_ggsp_cens = {'bandwidths': {'graph_bw_ran': graph_bw_ran_cens, 'time_bw_ran': time_bw_ran_cens},
                                 'nonlinear_type': 'sigmoid'}

            # estimate the lfdrs
            filename_ggsp_cens = 'MHT-GGSP_cens' + f'_repeat_{repeat_idx}.pkl'
            sav_path = os.path.join(curr_res_path, filename_ggsp_cens)
            lfdr_ggsp_cens, _, _, pi0_ggsp_cens, est_time_ggsp_cens= est_lfdr_beta_ggsp_cens(dat_ggsp_cens, dat_info_ggsp_cens, para_config_ggsp_cens, sav_path, sav_res)
            # detect the hypotheses
            h_est_ggsp_cens, FDR_ggsp_cens, pow_ggsp_cens = det_lfdr(alp_levels, lfdr_ggsp_cens, h_true)

            # Record the results
            FDR_summary['MHT-GGSP_cens'].append(FDR_ggsp_cens)
            Power_summary['MHT-GGSP_cens'].append(pow_ggsp_cens)

            print(f"MHT-GGSP_cens: FDR = {FDR_ggsp_cens[0]}, Power = {pow_ggsp_cens[0]}", f"Time = {est_time_ggsp_cens}")

        # -------------------
        # lfdr-sMoM
        # -------------------
        if 'lfdr-sMoM' in method_names:
            dat_smom = {'p_val': p_values_obs}
            N_QUAN_BITS = None
            SENSORING_LAM = 1
            par_lst  =[curr_data_path, 50, 'stan', N_QUAN_BITS, SENSORING_LAM]
            # Read in parameters for the spectral method of moments
            sMoM_k, sMoM_d, sMoM_n_tr, sMoM_reps_eta, sMoM_dis_msr = par.get_par_smom(
                curr_data_path)

            # estimate the lfdrs
            partition = "random"
            sav_path = os.path.join(curr_res_path, f'lfdr-smom-sen_repeat_{repeat_idx}.pkl')
            lfdr_smom, _, _, _,_ = est_lfdr_smom(dat_smom, sav_path, sav_res, par_lst, partition)
            # detect the hypotheses
            lfdr_smom = lfdr_smom.flatten()
            h_est_smom, FDR_smom, pow_smom = det_lfdr(alp_levels, lfdr_smom, h_true)

            # Record the results
            FDR_summary['lfdr-sMoM'].append(FDR_smom)
            Power_summary['lfdr-sMoM'].append(pow_smom)

            print(f"sMoM: FDR = {FDR_smom[0]}, Power = {pow_smom[0]}")

        # -------------------
        # Proportion-matching
        # -------------------
        if 'Prop-matching' in method_names:
            sample_points = np.column_stack(mask_indices)
            dat_pmatch = {'p_val': p_values_obs, 'sample_points': sample_points}
            sav_path = os.path.join(curr_res_path, f'pmatch-sen_repeat_{repeat_idx}.pkl')
            h_est_pmatch, FDR_pmatch, pow_pmatch = proportion_matching(dat_pmatch, tau_pmatch, alp_levels, h_true, sav_path, sav_res)

            # Record the results
            FDR_summary['Prop-matching'].append(FDR_pmatch)
            Power_summary['Prop-matching'].append(pow_pmatch)

            print(f"Proportion-matching: FDR = {FDR_pmatch[0]}, Power = {pow_pmatch[0]}")

        # -------------------
        # BH method
        # -------------------
        if 'BH' in method_names:
            sav_path = os.path.join(curr_res_path, f'bh-sen_repeat_{repeat_idx}.pkl')
            h_est_BH, FDR_BH, pow_BH = det_BH(alp_levels, p_values_obs, h_true, sav_path, sav_res)

            # Record the results
            FDR_summary['BH'].append(FDR_BH)
            Power_summary['BH'].append(pow_BH)

            print(f"BH: FDR = {FDR_BH[0]}, Power = {pow_BH[0]}")

        # -------------------
        # FDR-smoothing
        # -------------------
        if 'FDR-smoothing' in method_names:
            # This result is calculated by smoothfdr repo. So just load it.
            res_path_smoothfdr = os.path.join(curr_res_path, f'smoothfdr-sen_repeat_{repeat_idx}.pkl')
            with open(res_path_smoothfdr, 'rb') as f:
                smoothfdr_result = pickle.load(f, encoding='latin1')
                print("smoothfdr loaded!")
            h_est_smoothfdr = smoothfdr_result['h_est']
            FDR_smoothfdr = smoothfdr_result['FDR']
            pow_smoothfdr = smoothfdr_result['pow']

            # Record the results
            FDR_summary['FDR-smoothing'].append(FDR_smoothfdr)
            Power_summary['FDR-smoothing'].append(pow_smoothfdr)

        # -------------------
        # SABHA
        # -------------------
        if 'SABHA'in method_names:
            # directly load the results from SABHA
            res_path_sabha = os.path.join(curr_res_path, f'sabha-sen-repeat_{repeat_idx}.csv')
            sabha_res = pd.read_csv(res_path_sabha)
            FDR_sabha = sabha_res['FDR'].to_numpy()
            pow_sabha = sabha_res['pow'].to_numpy()

            # Record the results
            FDR_summary['SABHA'].append(FDR_sabha)
            Power_summary['SABHA'].append(pow_sabha)

        # -------------------
        # AdaPT
        # -------------------
        if 'AdaPT' in method_names:
            # directly read the results from AdaPT
            res_path_adapt = os.path.join(curr_res_path, f'adapt-sen-repeat_{repeat_idx}.csv')
            adapt_fd = pd.read_csv(res_path_adapt)
            FDR_adapt  = adapt_fd['FDR'].to_numpy()
            pow_adapt = adapt_fd['power'].to_numpy()

            # Record the results
            FDR_summary['AdaPT'].append(FDR_adapt)
            Power_summary['AdaPT'].append(pow_adapt)

    # -------------------
    # Plot the results
    # -------------------

    # Calculate the average FDR and power, only for alpha_level <= 0.30
    max_show_alpha = 0.31
    snr_show_alpha_idx = 3
    FDR_summary_avg = {}
    Power_summary_avg = {}
    for key in method_names:
        FDR_summary_avg[key] = np.mean(np.array(FDR_summary[key]), axis=0)
        Power_summary_avg[key] = np.mean(np.array(Power_summary[key]), axis=0)

        FDR_vary_summary[key].append(FDR_summary_avg[key][snr_show_alpha_idx])
        Power_vary_summary[key].append(Power_summary_avg[key][snr_show_alpha_idx])

    # Plot FDR versus alpha_level
    idx = np.where(alp_levels <= max_show_alpha)[0][-1] + 1
    plt.figure()
    for key_idx, key in enumerate(FDR_summary_avg.keys()):
        plt.plot(alp_levels[:idx], FDR_summary_avg[key][:idx], label=key, linestyle=line_styles[key_idx], color=colors[key_idx], marker=markers[key_idx])
        # Name x-axis and y-axis
        plt.xlabel('nominal FDR level')
        plt.ylabel('empirical FDR')

    # Add a x=y line
    x = np.linspace(alp_levels[0], alp_levels[idx - 1], 100)
    plt.plot(x, x, linestyle='-', color='k', label=r'$\alpha$')
    plt.legend()
    if sav_plots:
        plt.savefig(os.path.join(curr_res_path, 'empirical_FDR.pdf'))
    plt.show()

    # Plot power versus alpha_level
    plt.figure()
    for key_idx, key in enumerate(Power_summary_avg.keys()):
        plt.plot(alp_levels[:idx], Power_summary_avg[key][:idx], label=key, linestyle=line_styles[key_idx], color=colors[key_idx], marker=markers[key_idx])
        # Name x-axis and y-axis
    plt.xlabel('nominal FDR level')
    plt.ylabel('empirical power')
    plt.legend()
    if sav_plots:
        plt.savefig(os.path.join(curr_res_path, 'empirical_power.pdf'))
    plt.show()

    # plot detection example.
    n_instance = p_values.shape[0]
    alp_level_inspect = 0.20
    for inspect_instance in instance_inspect:

        # true hypotheses
        null_vertices = np.where(hypotheses[inspect_instance, :] == 0)[0]
        alt_vertices = np.where(hypotheses[inspect_instance, :] == 1)[0]

        # detected result
        significance_index = np.where(np.abs(alp_levels - alp_level_inspect) < 10 ** (-5))[0][0]
        h_est_ggsp_reshape = h_est_ggsp[significance_index].reshape(n_instance, n_vertex)
        null_vertices_ggsp = np.where(h_est_ggsp_reshape[inspect_instance, :] == 0)[0]
        alt_vertices_ggsp = np.where(h_est_ggsp_reshape[inspect_instance, :] == 1)[0]

        # Create a graph
        G = nx.from_numpy_array(graph_adj)

        # Assign colors based on detection results
        colors_illus = []
        for node in G.nodes:
            if node in null_vertices:
                if node in null_vertices_ggsp:
                    colors_illus.append('lightblue')  # null and not detected
                else:
                    colors_illus.append('red')  # null but detected
            elif node in alt_vertices:
                if node in alt_vertices_ggsp:
                    colors_illus.append('purple')  # alternative and detected
                else:
                    colors_illus.append('blue')  # alternative but not detected

        # Plot the graph
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = {i: (node_coords[i, 1], node_coords[i, 0]) for i in range(n_vertex)}  # Use node_coords for positions

        nx.draw(G, pos, with_labels=False, node_color=colors_illus, node_size=100,
                ax=ax)  # Adjust node_size to be smaller

        ax.set_title(f'Instance {inspect_instance} Detection Results')

        if sav_plots:
            plt.savefig(os.path.join(curr_res_path, f'instance_{inspect_instance}_detection.pdf'))

        plt.show()

    print('noise level:', noise_idx, 'iteration time:', time.time() - start_time)

# Plot the FDR and power for different noise levels
plt.figure()
for key_idx, key in enumerate(FDR_vary_summary.keys()):
    plt.plot(noise_levels, FDR_vary_summary[key], label=key, linestyle=line_styles[key_idx], color=colors[key_idx], marker=markers[key_idx])
    # Name x-axis and y-axis
plt.gca().invert_xaxis()
plt.xlabel('noise level')
plt.ylabel('empirical FDR')
plt.legend()
if sav_plots:
    plt.savefig(os.path.join(os.path.dirname(res_path[0]), 'FDR_vary.pdf'))
plt.show()

plt.figure()
for key_idx, key in enumerate(Power_vary_summary.keys()):
    plt.plot(noise_levels, Power_vary_summary[key], label=key, linestyle=line_styles[key_idx], color=colors[key_idx], marker=markers[key_idx])
    # Name x-axis and y-axis
plt.gca().invert_xaxis()
plt.xlabel('noise level')
plt.ylabel('empirical power')
plt.legend()
if sav_plots:
    plt.savefig(os.path.join(os.path.dirname(res_path[0]), 'Power_vary.pdf'))
plt.show()

print('ok')