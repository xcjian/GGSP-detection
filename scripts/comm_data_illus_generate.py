# %% setup: import packages
import sys
import os

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
import paths as paths

from sklearn.neighbors import kneighbors_graph

# %% setup: user input
# For example plots: which MC run shall be shown?
mc = 0

# noise levels (standard error)
noise_levels = np.array([1.5])
repeat_times = 1  # number of times to repeat the experiment

FD_SCEN = "scC_TSPIN"
SEN_CFG = "cfg2_TSIPN"

#--------------------------------
# generate the p-value processes
#--------------------------------
dataset = 'communication_illus'
counter = 0

for noise_std in noise_levels:

    dat_dir = paths.get_path_to_dat(FD_SCEN)
    res_dir = paths.get_path_to_res(FD_SCEN)

    dat_dir = os.path.join(dat_dir, 'communication_illus_data')
    res_dir = os.path.join(res_dir, 'communication_illus_res')

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
            FD_SCEN, dat_path)  # can manually change the parameters in this function.

        # Read in sensor parameters
        n_sen, n_sam_per_sen, var_sen_loc, sen_hom = par.get_par_sen_cfg(
            SEN_CFG, dat_path)

        if np.max(n_sam_per_sen) > n_sam:
            # We cannot use more samples per sensor than there are samples for each
            # grid point.
            print("Change the sensor configuration! Cannot use more samples per sensor"
                  " than there are samples per grid point")
            sys.exit()

        # Read in parameters for the spectral method of moments
        sMoM_k, sMoM_d, sMoM_n_tr, sMoM_reps_eta, sMoM_dis_msr = par.get_par_smom(
            dat_path)

        # Read in parameters for the spatial partitioning.
        (spa_var_bw_grid, spa_var_ker_vec, spa_var_sthr_grid,
         spa_var_pi0_max) = par.get_par_spa_var(dat_path)

        # Read in (and save) noise standard deviation
        np.save(dat_path + '/noise_std.npy', noise_std)

        print(f'Running scenario {FD_SCEN} in sensor cfg {SEN_CFG}!')
        # %% setup: reading in or creating the data
        # if 'fd' in globals() and 'est_fd' in globals():
        #     print('Fields already loaded')
        # else:
        fd, est_fd = fd_hdl.rd_in_fds(FD_SCEN, SEN_CFG, dat_path)

        # %% quantization
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

        # save the ground truth on the entire 100x100 grid.
        hypotheses_all = np.zeros((n_MC, ) + fd_dim)
        for instance in range(n_MC):
            hypotheses_all[instance, :, :] = fd.r_tru[instance, :].reshape(fd_dim)

        data_dict = {'p_values': est_fd.p, 'graph_adj': knn_graph, 'node_coords': est_fd.sen_cds[0, :, :],
                     'hypotheses': est_fd.r_tru.astype(int), 'hypotheses_all': hypotheses_all.astype(int),
                     'center_coords': fd.cen, 'z_values': est_fd.z, 'shadow_faded_signal': fd.X_sf}
        py_path = os.path.join(os.path.dirname(dat_path), 'data_for_py')
        if not os.path.exists(py_path):
            os.makedirs(py_path)
        with open(os.path.join(py_path, f'repeat_{repeat_idx}.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)

        # save data to .csv files
        R_path = os.path.join(os.path.dirname(dat_path), 'data_for_R')
        if not os.path.exists(R_path):
            os.makedirs(R_path)
        np.savetxt(os.path.join(R_path, f'p_values_{repeat_idx}.csv'), est_fd.p, delimiter=',')
        np.savetxt(os.path.join(R_path, f'graph_adj_{repeat_idx}.csv'), knn_graph, delimiter=',')
        np.savetxt(os.path.join(R_path, f'node_coords_{repeat_idx}.csv'), est_fd.sen_cds[0, :, :], delimiter=',')
        np.savetxt(os.path.join(R_path, f'hypotheses_{repeat_idx}.csv'), est_fd.r_tru.astype(int), delimiter=',')
        np.savetxt(os.path.join(R_path, f'z_values_{repeat_idx}.csv'), est_fd.z, delimiter=',')

        # delete the large files.
        shutil.rmtree(dat_path)
        os.remove(os.path.join(os.path.dirname(dat_path), f'repeat_{repeat_idx}.pkl'))

        print(f'Finished repeat {repeat_idx}', f'Noise level: {noise_std}')

    # save the noise level
    np.save(os.path.join(dat_dir, 'noise_std.npy'), noise_std)
    counter += 1

print('ok')