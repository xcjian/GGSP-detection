"""
Copyright (c) 2025, Xingchao Jian (Nanyang Technological University), Martin Goelz (TU Darmstadt)
All rights reserved.

This code is licensed under the MIT License.
You may obtain a copy of the License at https://opensource.org/licenses/MIT

Run this file to produce the figures for the example in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pickle
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Plot the illustration of the Bayesian model.
"""
# ------------------------------------------------------------
# read in the data
# ------------------------------------------------------------
dataset = 'communication_illus'
vary_snr = False
save_fig = True

config = get_config(dataset, vary_snr)
data_path = config['data_path']
res_path = config['res_path']

repeat_idx = 0
curr_data_path = data_path[0]
res_path = res_path[0]

with open(os.path.join(curr_data_path, f'repeat_{repeat_idx}.pkl'), 'rb') as f:
    data_dict = pickle.load(f)

p_values = data_dict['p_values']
graph_adj = data_dict['graph_adj']
node_coords = data_dict['node_coords']
hypotheses = data_dict['hypotheses']
center_coords = data_dict['center_coords']
shadow_faded_signal = data_dict['shadow_faded_signal']

inspect_instance = 5
center_coord = center_coords[inspect_instance, 0, :]
n_vertices = node_coords.shape[0]
n_instances = center_coords.shape[0]
gamma_dim = shadow_faded_signal.shape[2]

# ------------------------------------------------------------
# Compute gamma (minimum distance) for both 100 x 100 grid and graph, for all instances
# ------------------------------------------------------------
grid_wid = 100
gamma_grid = np.zeros((n_instances, grid_wid, grid_wid, gamma_dim))
gamma_graph = np.zeros((n_instances, n_vertices, gamma_dim))
for k in range(gamma_dim):
    for t in range(n_instances):
        gamma_grid[t, :, :, k] = np.reshape(shadow_faded_signal[inspect_instance, :, k], (grid_wid, grid_wid))
        for i in range(n_vertices):
            # find the index of the node in the grid
            x_idx = int(node_coords[i, 0])
            y_idx = int(node_coords[i, 1])
            gamma_graph[t, i, k] = gamma_grid[t, y_idx, x_idx, k]

gamma_norm_graph = np.linalg.norm(gamma_graph, axis=2)
# ------------------------------------------------------------
# Plot gamma(v,t) on the graph for one instance.
# ------------------------------------------------------------

# Create the graph from the adjacency matrix
G = nx.from_numpy_array(graph_adj)
pos = {i: (node_coords[i, 0], node_coords[i, 1]) for i in range(node_coords.shape[0])}

def plot_gam_combined(gamma_grid, gamma_graph, transmitter_coords, plot_val_range, plot_pow, gamma_dim, save_fig=False, res_path="."):
    """
    Plot gamma(v,t) as 2 subfigures in a single horizontal figure with a shared color bar on the left.
    """
    # Normalize gamma for color mapping
    norm = plt.Normalize(vmin=plot_val_range[0], vmax=plot_val_range[1])
    cmap = plt.cm.Blues

    # Create a figure with horizontal subplots
    fig, axs = plt.subplots(1, gamma_dim, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1], 'wspace': -0.2})

    for k in range(gamma_dim):
        gam_single_grid = gamma_grid[inspect_instance, :, :, k] ** plot_pow
        gam_single_graph = gamma_graph[inspect_instance, :, k] ** plot_pow
        transmitter_coord = transmitter_coords[inspect_instance, k, :]

        # Plot the graph with distances represented as colors
        ax = axs[k]
        im = ax.imshow(gam_single_grid, origin='lower', extent=(0, grid_wid, 0, grid_wid), cmap=cmap, norm=norm)

        # Draw the graph with nodes colored by gamma
        nx.draw(G, pos, with_labels=False, node_color=gam_single_graph, cmap=cmap, node_size=70, ax=ax,
                vmin=plot_val_range[0], vmax=plot_val_range[1], edgecolors='k')

        # Plot the transmitter coordinates
        ax.scatter(transmitter_coord[0], transmitter_coord[1], s=600, marker='*', color='orange', zorder=5, label=f'Transmitter {k + 1}')

    # Add a shared color bar on the left
    cax = fig.add_axes([0.11, 0.20, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=15)

    # Save the combined figure if required
    if save_fig:
        fig_path = os.path.join(res_path, 'combined_gamma_plot.pdf')
        plt.savefig(fig_path, bbox_inches='tight')

    plt.show()

# Example usage
plot_pow = 1 / 2
plot_val_range = np.array([gamma_grid[inspect_instance, :, :, :].min(), gamma_grid[inspect_instance, :, :, :].max()]) ** plot_pow
plot_gam_combined(gamma_grid, gamma_graph, center_coords, plot_val_range, plot_pow, gamma_dim, save_fig=True, res_path=res_path)


# ------------------------------------------------------------
# plot the empirical distribution of p-values under different ranges of gamma's norm
# ------------------------------------------------------------

# Define the gamma value bins
gamval_bins = [0.2, 0.35, 0.5]

# Create a 1x2 figure
fig, axs = plt.subplots(1, len(gamval_bins) - 1, figsize=(14, 6), gridspec_kw={'wspace': 0.14})

# Plot histograms for p-values in different distance ranges

for i, (lower, upper) in enumerate(zip(gamval_bins[:-1], gamval_bins[1:])):
    # Select p-values and hypotheses within the current distance range for all instances
    p_vals_in_range_all = []
    hypotheses_in_range_all = []

    for t in range(n_instances):
        indices = (gamma_norm_graph[t, :] >= lower) & (gamma_norm_graph[t, :] < upper)
        p_vals_in_range_all.extend(p_values[t, indices])
        hypotheses_in_range_all.extend(hypotheses[t, indices])

    p_vals_in_range_all = np.array(p_vals_in_range_all)
    hypotheses_in_range_all = np.array(hypotheses_in_range_all)

    # Filter p-values under alternative hypotheses
    alt_p_vals_in_range = p_vals_in_range_all[hypotheses_in_range_all == 1]

    # Plot histogram with frequencies
    axs[i].hist(alt_p_vals_in_range, bins=20, color='blue', edgecolor='black', density=True, range=(0, 1))
    axs[i].set_xlim((0, 1))
    axs[i].set_xlabel('Alternative $p$-value', fontsize=18)
    axs[i].set_ylabel('Frequency', fontsize=18)

    # Compute the proportion of null hypotheses
    proportion_null = np.sum(hypotheses_in_range_all == 0) / len(hypotheses_in_range_all)
    axs[i].set_title(r'empirical $\pi_0$: ' + f'{proportion_null:.2f}', fontsize=18)

    # Make x-axis tick labels larger
    axs[i].tick_params(axis='x', labelsize=15)  # Adjust fontsize for x-axis ticks
    # Make y-axis tick labels larger
    axs[i].tick_params(axis='y', labelsize=15)  # Adjust fontsize for y-axis ticks

if save_fig:
    fig_path = os.path.join(res_path, 'p_value_distribution_combined.pdf')
    plt.savefig(fig_path, bbox_inches='tight')
plt.show()


print('Done!')