import matplotlib.pyplot as plt
import numpy as np

from utils import *
from detection import *
import networkx as nx

# -------------------
# call dataset
# -------------------
dataset = 'communication_illus'
vary_snr = True

#-------------------
# Set configurations
#-------------------
config = get_config(dataset, vary_snr)
data_path = config['data_path']
res_path = config['res_path']
repeat_time = config['repeat_time']
alp_levels = config['alp_levels']
sav_res = config['sav_res']
sav_plots = config['sav_plots']
instance_inspect = config['instance_inspect']
null_pdf = config['null_pdf']

# Set the parameters for the GGSP method
graph_bw_ran = config['graph_bw_ran']
time_bw_ran = config['time_bw_ran']
combinatorial = config['combinatorial']

# Set the parameters for the GGSP-AdaPT method
# unit = config['unit_adapt']

# Set the parameters for the proportion-matching method
tau_pmatch = config['tau_pmatch']

# Set the parameters for the SABHA method
tau_sabha = config['tau_sabha']

#Set plotting parameters
method_names = config['method_names']
line_styles = config['line_styles']
colors = config['colors']
markers = config['markers']

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

    FDR_summary = {'MHT-GGSP': [], 'lfdr-sMoM': [], 'Prop-matching': [], 'BH': [], 'FDR-smoothing': [], 'SABHA': [], 'AdaPT': []}
    Power_summary = {'MHT-GGSP': [], 'lfdr-sMoM': [], 'Prop-matching': [], 'BH': [], 'FDR-smoothing': [], 'SABHA': [], 'AdaPT': []}
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
        hypotheses_all = data_dict['hypotheses_all']
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

        # obtain the graph Fourier basis
        n_vertex = graph_adj.shape[0]
        g_Lap = np.diag(graph_adj @ np.ones(n_vertex)) - graph_adj
        g_freq, g_fb = graph_spectral_decomp(g_Lap)

        sample_points = np.column_stack(mask_indices)
        dat_ggsp = {'p_val': p_values_obs, 'sample_points': sample_points}
        dat_info_ggsp = {'graph_basis': g_fb, 'time_max_idx': p_values.shape[0]}
        para_config_ggsp = {'bandwidths': {'graph_bw_ran': graph_bw_ran, 'time_bw_ran': time_bw_ran},
                             'nonlinear_type': 'softplus_dash', 'null_pdf': null_pdf}

            # estimate the lfdrs
        filename_ggsp = 'beta-ggsp-' + para_config_ggsp['nonlinear_type'] + f'_repeat_{repeat_idx}.pkl'
        sav_path = os.path.join(curr_res_path, filename_ggsp)
        lfdr_ggsp, _, _, _, est_time_direct= est_lfdr_beta_ggsp(dat_ggsp, dat_info_ggsp, para_config_ggsp, sav_path, sav_res)
        # detect the hypotheses
        h_est_ggsp, FDR_ggsp, pow_ggsp = det_lfdr(alp_levels, lfdr_ggsp, h_true)
        print(f"GGSP: FDR = {FDR_ggsp[0]}, Power = {pow_ggsp[0]}", f"Time = {est_time_direct}")


    # -------------------
    # Plot the results
    # -------------------

    # plot detection example.
    n_instance = p_values.shape[0]
    alp_level_inspect = 0.2

    for inspect_instance in instance_inspect:

        # true hypotheses
        null_vertices = np.where(hypotheses_all[inspect_instance, :].flatten() == 0)[0]
        alt_vertices = np.where(hypotheses_all[inspect_instance, :].flatten() == 1)[0]

        null_vertices_graph = np.where(hypotheses[inspect_instance, :] == 0)[0]
        alt_vertices_graph = np.where(hypotheses[inspect_instance, :] == 1)[0]

        # detected result
        significance_index = np.where(np.abs(alp_levels - alp_level_inspect) < 10 ** (-5))[0][0]
        h_est_ggsp_reshape = h_est_ggsp[significance_index].reshape(n_instance, n_vertex)
        null_vertices_ggsp = np.where(h_est_ggsp_reshape[inspect_instance, :] == 0)[0]
        alt_vertices_ggsp = np.where(h_est_ggsp_reshape[inspect_instance, :] == 1)[0]

        # Create the grid for background
        grid_wid = 100
        grid_coordinates = [(x, y) for x in range(grid_wid) for y in range(grid_wid)]
        ground_truth_grid = np.zeros((grid_wid, grid_wid, 3))  # RGB grid

        # Plot ground truth null and alternative vertices on the grid
        for grid_idx in range(len(grid_coordinates)):
            x, y = grid_coordinates[grid_idx]
            if grid_idx in null_vertices:
                ground_truth_grid[x, y] = [0.678, 0.847, 0.902]  # Light blue for null
            elif grid_idx in alt_vertices:
                ground_truth_grid[x, y] = [0.529, 0.808, 1]  # Deeper blue for alternative

        # Create a graph
        G = nx.from_numpy_matrix(graph_adj)

        # Assign colors based on detection results
        colors_illus = []
        edgecolors = []
        for node in G.nodes:
            if node in null_vertices_graph:
                if node in null_vertices_ggsp:
                    colors_illus.append('lightblue')  # null and not detected
                else:
                    colors_illus.append('red')  # null but detected
            elif node in alt_vertices_graph:
                if node in alt_vertices_ggsp:
                    colors_illus.append('purple')  # alternative and detected
                else:
                    colors_illus.append('blue')  # alternative but not detected
            edgecolors.append('black')  # Add black boundaries to all nodes

        # Plot the background grid
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(ground_truth_grid, origin='lower', extent=(0, grid_wid, 0, grid_wid))

        # Overlay the graph with detected null and alternative vertices
        pos = {i: (node_coords[i, 0], node_coords[i, 1]) for i in range(n_vertex)}
        nx.draw(G, pos, with_labels=False, node_color=colors_illus, node_size=70, edgecolors=edgecolors,ax=ax)

        # Plot the transmitter coordinates
        for t in range(center_coords.shape[1]):
            transmitter_coords = center_coords[inspect_instance, t, :]  # Coordinates of the t-th transmitter
            ax.scatter(transmitter_coords[0], transmitter_coords[1], s=300, marker='*', color = 'orange', zorder=5,
                       label=f'Transmitter {t + 1}')

        # Set the title and show the plot
        ax.set_title(f'Instance {inspect_instance} Detection Results')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')

        if sav_plots:
            plt.savefig(os.path.join(curr_res_path, f'instance_{inspect_instance}_detection.pdf'))

        plt.show()

    print('noise level:', noise_idx, 'iteration time:', time.time() - start_time)



print('ok')