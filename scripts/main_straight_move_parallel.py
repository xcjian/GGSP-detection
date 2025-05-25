"""
Copyright (c) 2025, Xingchao Jian (Nanyang Technological University), Martin Goelz (TU Darmstadt)
All rights reserved.

This code is licensed under the MIT License.
You may obtain a copy of the License at https://opensource.org/licenses/MIT

This code is for testing the asymptotic properties of the algorithm.
"""

import argparse
import numpy as np

from utils import *
from detection import *
from para_estimation import compute_pi0
import parameters as par
import networkx as nx
import matplotlib.pyplot as plt
import load_and_save as ls
import pickle
from multiprocessing import Pool
import itertools

# -------------------
# pass the parameters to the script
# -------------------

# Initialize the parser
parser = argparse.ArgumentParser()

# Add the dataset argument with choices
parser.add_argument('--dataset', type=str, default='straight_move', choices=['straight_move'],
                    help='The dataset to be used for the experiments.')
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

# Parse the arguments
args = parser.parse_args()

# -------------------
# call dataset
# -------------------
dataset = args.dataset
alp_levels = np.array(args.alp_levels)
sav_res = args.sav_res
sav_plots = args.sav_plots
instance_inspect = args.instance_inspect
combinatorial = args.combinatorial
noise_level = 1
n_vertex = 300

#-------------------
# Set configurations
#-------------------
config = get_config(dataset)
data_path = config['data_path']
res_path = config['res_path']
repeat_time = config['repeat_time']

# Set the parameters for MHT-GGSP
graph_bw_ran = config['graph_bw_ran']
time_bw_ran = config['time_bw_ran']

# Number of workers for the parallelization
num_wrk = np.min((2, os.cpu_count() - 1))  # Change first value if you want to

#Set plotting parameters
method_names = ['MHT-GGSP', 'MHT-GGSP-oracle']
line_styles = ['--', '--']
colors = ['r', 'tab:brown']
markers = ['D', 'o']

# -------------------
T_levels = np.zeros(len(data_path))
n_T = len(data_path)

def detection_task(T_repeat_idx):

    FDR_summary = {
    'MHT-GGSP': np.zeros((repeat_time, n_T, len(alp_levels))),
    'MHT-GGSP-oracle': np.zeros((repeat_time, n_T, len(alp_levels)))
    }
    Power_summary = {
    'MHT-GGSP': np.zeros((repeat_time, n_T, len(alp_levels))),
    'MHT-GGSP-oracle': np.zeros((repeat_time, n_T, len(alp_levels)))
    }

    # Run a single detection experiment

    T_idx, repeat_idx = T_repeat_idx

# for T_idx in range(n_T):
    start_time = time.time()

    curr_data_path = data_path[T_idx]
    curr_res_path = res_path[T_idx]

    curr_T_level = np.load(os.path.join(os.path.dirname(curr_data_path), 'T.npy'))
    # T_levels[T_idx] = curr_T_level

    # load data
    with open(os.path.join(curr_data_path, f'repeat_{repeat_idx}.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    
    if not os.path.exists(curr_res_path):
        os.makedirs(curr_res_path, exist_ok=True)

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
    shadow_faded_signal = data_dict['shadow_faded_signal']


    # -------------------
    # Oracle-MHT-GGSP
    # -------------------
    n_vertex = graph_adj.shape[0]
    n_instance = p_values.shape[0]
    gamma_dim = shadow_faded_signal.shape[2]
    sample_points = np.column_stack(mask_indices)
    null_threshold = 0.2

    # Compute gamma for all (v, t)
    shadow_faded_signal = data_dict['shadow_faded_signal']
    grid_wid = 100
    gamma_graph = np.zeros((n_instance, n_vertex, gamma_dim))
    for k in range(gamma_dim):
        for t in range(n_instance):
            gamma_grid_ = np.reshape(shadow_faded_signal[t, :, k], (grid_wid, grid_wid))
            for i in range(n_vertex):
                # find the index of the node in the grid
                x_idx = int(node_coords[i, 0])
                y_idx = int(node_coords[i, 1])
                gamma_graph[t, i, k] = gamma_grid_[y_idx, x_idx]

    # compute the lfdrs
    dat_ora = {'p_val': p_values_obs, 
                'sample_points': sample_points,
                'gamma_graph': gamma_graph,
                'null_threshold': null_threshold,
                'noise_level': noise_level
                }
    filename_ora = 'MHT-GGSP-oracle' + f'_repeat_{repeat_idx}.pkl'
    sav_path = os.path.join(curr_res_path, filename_ora)
    lfdr_ora, pi0_ora, fp_ora, time_ora = lfdr_oracle(dat_ora, sav_path, sav_res)
    
    # detect the hypotheses
    h_est_ora, FDR_ora, pow_ora = det_lfdr(alp_levels, lfdr_ora, h_true)

    # Record the results
    FDR_summary['MHT-GGSP-oracle'] = np.array(FDR_ora)
    Power_summary['MHT-GGSP-oracle'] = np.array(pow_ora)

    print(f"MHT-GGSP-oracle: FDR = {FDR_ora[0]}, Power = {pow_ora[0]}", f"Time = {time_ora}")

    # -------------------
    # MHT-GGSP
    # -------------------

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

    # save the algorithm properties
    K1_, K2_ = ls.ld_alg_properties(sav_path)
    
    # Record the results
    FDR_summary['MHT-GGSP'] = np.array(FDR_ggsp)
    Power_summary['MHT-GGSP'] = np.array(pow_ggsp)

    print(f"MHT-GGSP: FDR = {FDR_ggsp[0]}, Power = {pow_ggsp[0]}", f"Time = {est_time_direct}")

    return FDR_summary, Power_summary, K1_, K2_, curr_T_level


# Create all possible pairs
pairs = list(itertools.product(list(range(n_T)), list(range(repeat_time))))

# Parallel processing
with Pool(processes=num_wrk) as pool:  # Adjust processes as needed
    results = pool.map(detection_task, pairs)

# Reconstruct results in order if needed
FDR_result_summary = {}
Power_result_summary = {}
for method in method_names:
    FDR_result_summary[method] = np.zeros((n_T, repeat_time, len(alp_levels)))
    Power_result_summary[method] = np.zeros((n_T, repeat_time, len(alp_levels)))
# noise_result_summary = np.zeros((n_T, repeat_time))
K1 = np.zeros((n_T, repeat_time))
K2 = np.zeros((n_T, repeat_time))
T_levels = np.zeros((n_T, repeat_time))
for (x, y), result in zip(pairs, results):
    FDR_, Pow_, K1_, K2_, T_ = result
    for method in method_names:
        FDR_result_summary[method][x, y, :] = FDR_[method]
        Power_result_summary[method][x, y, :] = Pow_[method]
    
    K1[x, y] = K1_
    K2[x, y] = K2_
    T_levels[x, y] = T_
T_levels = np.mean(T_levels, axis = 1)

sav_property_path = os.path.dirname(res_path[0]) + '/res_properties.pkl'
alg_properties = {
    'K1': K1,
    'K2': K2,
    'FDR_summary': FDR_result_summary,
    'Power_summary': Power_result_summary,
    'T_levels': T_levels
}

# if not os.path.exists(sav_property_path):
with open(sav_property_path, 'wb') as f:
    pickle.dump(alg_properties, f)

# plot

## plot K1 and K2
K1_mean = np.mean(K1, axis = 1)
K2_mean = np.mean(K2, axis = 1)
### K1:
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))
ax1.plot(T_levels * n_vertex, K1_mean, label = 'K1')
ax1.set_xlabel(r'$I$')
ax1.set_ylabel(r'$K_1$')
ax1.legend()

### K2:
ax2.plot(T_levels * n_vertex, K2_mean, label = 'K2')
ax2.set_xlabel(r'$I$')
ax2.set_ylabel(r'$K_2$')
ax2.legend()

plt.tight_layout()
if sav_plots:
    res_fig_folder = os.path.dirname(sav_property_path)
    plt.savefig(res_fig_folder + '/bandwidths.pdf')
# plt.show()

## plot FDR
disp_alp_idx = 1
plt.figure()
for idx, method in enumerate(method_names):
    FDR_mean = np.mean(FDR_result_summary[method][:, :, disp_alp_idx], axis = 1)
    plt.plot(T_levels * n_vertex, FDR_mean, label = method, linestyle=line_styles[idx], color=colors[idx], marker=markers[idx])
    
plt.xlabel(r'$I$')
plt.ylabel(r'$empirical FDR$')
plt.legend()
if sav_plots:
    res_fig_folder = os.path.dirname(sav_property_path)
    plt.savefig(res_fig_folder + '/FDR.pdf')
# plt.show()

## plot power
disp_alp_idx = 1
plt.figure()
for idx, method in enumerate(method_names):
    pow_mean = np.mean(Power_result_summary[method][:, :, disp_alp_idx], axis = 1)
    plt.plot(T_levels * n_vertex, pow_mean, label = method, linestyle=line_styles[idx], color=colors[idx], marker=markers[idx])
    
plt.xlabel(r'$I$')
plt.ylabel(r'$empirical power$')
plt.legend()
if sav_plots:
    res_fig_folder = os.path.dirname(sav_property_path)
    plt.savefig(res_fig_folder + '/pow.pdf')
# plt.show()

print('ok')