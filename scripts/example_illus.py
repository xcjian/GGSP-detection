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

def plot_gam(gam_single_grid, gam_single_graph, transmitter_coord, range, power):
    """
    Plot the field and graph with gamma values on it.
    This function is used to plot one dimension of gamma.
    :param gam_single_grid: 2D array of gamma values on the grid
    :param gam_single_graph: 1D array of gamma values on the graph
    :param transmitter_coord: Coordinates of the transmitter
    :param range: Range of gamma values on the plot bar
    :param power: Power to raise gamma values to for better visualization
    """

    gam_single_grid = gam_single_grid ** power
    gam_single_graph = gam_single_graph ** power

    # Normalize gamma for color mapping
    norm = plt.Normalize(vmin=range[0], vmax=range[1])
    cmap = plt.cm.Blues

    # Plot the graph with distances represented as colors
    fig, ax = plt.subplots(figsize=(12, 12))

    # Normalize distances for color mapping
    im = ax.imshow(gam_single_grid, origin='lower', extent=(0, grid_wid, 0, grid_wid), cmap=cmap, norm=norm)

    # Draw the graph with nodes colored by gamma
    nx.draw(G, pos, with_labels=False, node_color=gam_single_graph, cmap=cmap, node_size=70, ax=ax, vmin=range[0], vmax=range[1],
            edgecolors='k')

    # Add color bar
    # Create a divider for the color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=20)

    # Plot the transmitter coordinates
    ax.scatter(transmitter_coord[0], transmitter_coord[1], s=300, marker='*', color = 'orange', zorder=5,
                   label=f'Transmitter {t + 1}')

plot_pow = 1/2
plot_val_range = np.array([gamma_grid[inspect_instance, :, :, :].min(), gamma_grid[inspect_instance, :, :, :].max()])
plot_val_range = plot_val_range ** plot_pow
for k in range(gamma_dim):
    plot_gam(gamma_grid[inspect_instance, :, :, k], gamma_graph[inspect_instance, :, k], center_coords[inspect_instance, k, :], plot_val_range, plot_pow)
    # Set the title and show the plot
    # ax.set_title('Minimum Distance from Each Vertex to Either Transmitter')
    if save_fig:
        fig_path = os.path.join(res_path, f'gamma_dim{k}.pdf')
        plt.savefig(fig_path)
    plt.show()

# ------------------------------------------------------------
# plot the empirical distribution of p-values under different ranges of gamma's norm
# ------------------------------------------------------------

# Define the gamma value bins
gamval_bins = [0.2, 0.35, 0.5]
# bin_labels = ['0.1-0.3', '0.3-0.5']

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
    plt.hist(alt_p_vals_in_range, bins=20, color='blue', edgecolor='black', density=True, range=(0, 1))
    plt.xlim((0, 1))
    plt.xlabel('Alternative $p$-value')
    plt.ylabel('Frequency')

    # Compute the proportion of null hypotheses
    proportion_null = np.sum(hypotheses_in_range_all == 0) / len(hypotheses_in_range_all)
    plt.title(r'empirical $\pi_0$: ' + f'{proportion_null:.2f}')

    if save_fig:
        fig_path = os.path.join(res_path, f'p_dist_{i}.pdf')
        plt.savefig(fig_path)
    plt.show()


print('Done!')