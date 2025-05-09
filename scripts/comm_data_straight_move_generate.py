"""
Copyright (c) 2025, Xingchao Jian (Nanyang Technological University), Martin Goelz (TU Darmstadt)
All rights reserved.

This code is licensed under the MIT License.
You may obtain a copy of the License at https://opensource.org/licenses/MIT

Run this file to generate the data for the communication scenario with fast fading.
"""
# %% setup: import packages
import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

# %% setup: what will become a package
import field_handling as fd_hdl
import detectors as det
import utils
import time
import pickle
import shutil

# %% setup: import custom files
import parameters as par
import paths

from sklearn.neighbors import kneighbors_graph

# %% setup: user input

# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument('--FD_SCEN', type=str, default='scC_TSPIN',
                    choices=['scA_TSIPN','scA_CISS','scA_ICASSP','scB_TSIPN','scB_ICASSP','scB_CISS','scC_TSPIN','scC_CISS','scC_ICASSP'],
                    help='The field scenario to be used for simulation.')
parser.add_argument('--SEN_CFG', type=str, default='cfg2_TSIPN', choices=['cfg2_TSIPN'],
                    help='The sensor configuration to be used for simulation.')
parser.add_argument('--noise_level', type=float, default=1.0,
                    help='The noise levels to be used for simulation.')
parser.add_argument('--repeat_times', type=int, default=60,
                    help='The number of times to repeat the experiment.')
parser.add_argument('--T_range', type=int, nargs='+', default=list(range(10, 85, 5)),
                    help='The number of instances.')
parser.add_argument('--step_size_min', type=float, default=0.25,
                    help='The step_size under the largest T.')
# what really affects K_2 here, is how long does the transmitters travel during the entire time.
args = parser.parse_args()

# Number of workers for the parallelization
num_wrk = np.min((50, os.cpu_count() - 1))  # Change first value if you want to
# use more than 50 cores if available.

# simulation settings
FD_SCEN = args.FD_SCEN
SEN_CFG = args.SEN_CFG
noise_level = args.noise_level
repeat_times = args.repeat_times
T_range = np.array(args.T_range)
step_size_min = args.step_size_min

move_type = 'straight'

#--------------------------------
# generate the p-value processes
#--------------------------------

counter = 0
for T_idx, T in enumerate(T_range):

    dat_dir = paths.get_path_to_dat(FD_SCEN)
    res_dir = paths.get_path_to_res(FD_SCEN)

    dat_dir = os.path.join(dat_dir, 'straight_move_data', f'T_{T_idx}')
    res_dir = os.path.join(res_dir, 'straight_move_res', f'T_{T_idx}')

    total_move = step_size_min * T_range.max()
    step_size_ = total_move / T 

    for repeat_idx in range(repeat_times):

        # %% setup: create directories to store the results

        # create directories to store the results, if they don't exist
        dat_path = os.path.join(dat_dir, f'repeat_{repeat_idx}')
        res_path = os.path.join(res_dir, f'repeat_{repeat_idx}')
        try:
            os.makedirs(dat_path)
        except FileExistsError:
            None
        try:
            if not (SEN_CFG == 'custom' or SEN_CFG == 'non-spatial'):
                os.makedirs(os.path.join(dat_path, SEN_CFG))
        except FileExistsError:
            None
        try:
            os.makedirs(res_path)
        except FileExistsError:
            None
        try:
            os.makedirs(os.path.join(res_path, SEN_CFG))
        except FileExistsError:
            None

        # if there already are data, we do not want to overwrite them
        if os.path.exists(os.path.join(os.path.dirname(dat_path), 'data_for_py', f'repeat_{repeat_idx}.pkl')):
            print('Results already exist for this configuration. Skipping...')
            shutil.rmtree(dat_path)
            continue
        # %% setup: read in parameters
        # Read in field parameters
        fd_dim, n_MC, n_sam, n_src, pi0_des, sha_fa, fast_fa, prop_env = par.get_par_fd_scen(
            FD_SCEN, dat_path, T = T) # can manually change the parameters in this function.

        # Read in sensor parameters
        n_sen, n_sam_per_sen, var_sen_loc, sen_hom = par.get_par_sen_cfg(
                SEN_CFG, dat_path)

        if np.max(n_sam_per_sen) > n_sam:
            # We cannot use more samples per sensor than there are samples for each
            # grid point.
            print("Change the sensor configuration! Cannot use more samples per sensor"
                  " than there are samples per grid point")
            sys.exit()

        # Read in (and save) noise standard deviation
        np.save(dat_path + '/noise_std.npy', noise_level)

        # save the parameters of the moving transmitters
        x_cen = [20, 50] 
        y_cen = [10, 50]
        control_paras = {
            'move_type': move_type,
            'step_size': step_size_,
            'initial_pos': np.array([x_cen, y_cen]).transpose(),
            'move_direction': [0, 0]
        }
        with open(dat_path + '/control_paras.pkl', 'wb') as f:
            pickle.dump(control_paras, f)

        print(f'Running scenario {FD_SCEN} in sensor cfg {SEN_CFG}!')
        # %% setup: reading in or creating the data
        # if 'fd' in globals() and 'est_fd' in globals():
        #     print('Fields already loaded')
        # else:
        fd, est_fd = fd_hdl.rd_in_fds(FD_SCEN, SEN_CFG, dat_path)

        res_path = os.path.join(res_path, SEN_CFG)

        # %% save the data

        # construct the graph from coordinates with k-NN
        n_neighbors = 10
        coordinates = est_fd.sen_cds[0, :, :]
        knn_graph = kneighbors_graph(coordinates, n_neighbors, mode='connectivity',
                                     include_self=False)
        knn_graph = knn_graph.toarray()
        knn_graph = knn_graph + knn_graph.T
        knn_graph[knn_graph > 0] = 1
        edges = np.argwhere(knn_graph == 1)
        # set the edge weights to be the exponential of the negative Euclidean distance
        dists = np.linalg.norm(coordinates[edges[:, 0], :] - coordinates[edges[:, 1], :], axis=1)
        edge_weights = np.exp(- dists / dists.std())
        knn_graph[edges[:, 0], edges[:, 1]] = edge_weights
        Laplacian = np.diag(np.sum(knn_graph, axis=1)) - knn_graph

        # calculate the Graph Fourier Basis
        eigval, eigvec = np.linalg.eigh(Laplacian)
        # sort the eigenvalues and eigenvectors
        idx = eigval.argsort()
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]


        data_dict = {'p_values': est_fd.p, 'graph_adj': knn_graph, 'node_coords': est_fd.sen_cds[0,:,:], 'hypotheses': est_fd.r_tru.astype(int),
                     'center_coords': fd.cen, 'z_values': est_fd.z, 'shadow_faded_signal': fd.X_sf}
        py_path = os.path.join(os.path.dirname(dat_path),  'data_for_py')
        if not os.path.exists(py_path):
            os.makedirs(py_path)
        with open(os.path.join(py_path, f'repeat_{repeat_idx}.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)


        # save data to .csv files
        # R_path = os.path.join(os.path.dirname(dat_path), 'data_for_R')
        # if not os.path.exists(R_path):
        #     os.makedirs(R_path)
        # np.savetxt(os.path.join(R_path, f'p_values_{repeat_idx}.csv'), est_fd.p, delimiter=',')
        # np.savetxt(os.path.join(R_path, f'graph_adj_{repeat_idx}.csv'), knn_graph, delimiter=',')
        # np.savetxt(os.path.join(R_path, f'node_coords_{repeat_idx}.csv'), est_fd.sen_cds[0,:,:], delimiter=',')
        # np.savetxt(os.path.join(R_path, f'hypotheses_{repeat_idx}.csv'), est_fd.r_tru.astype(int), delimiter=',')
        # np.savetxt(os.path.join(R_path, f'z_values_{repeat_idx}.csv'), est_fd.z, delimiter=',')

        # delete the large files.
        shutil.rmtree(dat_path)
        os.remove(os.path.join(os.path.dirname(dat_path), f'repeat_{repeat_idx}.pkl'))

        #-----plot the spatial distribution of hypotheses-----
        instance_plot = 0
        grid_coords = np.array([[i, j] for i in range(fd_dim[0]) for j in range(fd_dim[1])])
        plt.figure()
        plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c=fd.r_tru[instance_plot, :], s=5)
        plt.colorbar()
        plt.title('Hypotheses')
        plt.show()



        print(f'Finished repeat {repeat_idx}', f'T: {T}')

    # save the noise level
    np.save(os.path.join(dat_dir, 'T.npy'), T)
    counter += 1

print('ok')