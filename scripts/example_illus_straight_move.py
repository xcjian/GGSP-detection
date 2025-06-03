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
from scipy import stats
from para_estimation import compute_pi0



"""
Plot the illustration of the Bayesian model, using the straight move data.
"""
# ------------------------------------------------------------
# read in the data
# ------------------------------------------------------------
dataset = 'straight_move_illus'
vary_snr = False
save_fig = True

config = get_config(dataset, vary_snr)
data_path = config['data_path']
res_path = config['res_path']

repeat_idx = 0
T_idx = 0 # T = 20, and move 2 step each instance. 
curr_data_path = data_path[T_idx]
res_path = res_path[T_idx]

with open(os.path.join(curr_data_path, f'repeat_{repeat_idx}.pkl'), 'rb') as f:
    data_dict = pickle.load(f)

p_values = data_dict['p_values']
graph_adj = data_dict['graph_adj']
node_coords = data_dict['node_coords']
hypotheses = data_dict['hypotheses']
center_coords = data_dict['center_coords']
shadow_faded_signal = data_dict['shadow_faded_signal']

inspect_instance = [0, 18]
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
        gamma_grid[t, :, :, k] = np.reshape(shadow_faded_signal[t, :, k], (grid_wid, grid_wid))
        for i in range(n_vertices):
            # find the index of the node in the grid
            x_idx = int(node_coords[i, 0])
            y_idx = int(node_coords[i, 1])
            gamma_graph[t, i, k] = gamma_grid[t, y_idx, x_idx, k]

gamma_norm_graph = np.linalg.norm(gamma_graph, axis=2)
# ------------------------------------------------------------
# Plot gamma(v,t) on the graph for instances.
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
    n_cols = len(inspect_instance)
    # Adjusted parameters for minimal whitespace
    fig, axs = plt.subplots(
        gamma_dim, 
        n_cols,
        figsize=(8 * n_cols, 8 * gamma_dim),  # More balanced dimensions
        gridspec_kw={
            'width_ratios': [1] * n_cols,
            'wspace': -0.1,  # Reduced horizontal spacing
            'hspace': -0.1   # Reduced vertical spacing
        }
    )

    for k in range(gamma_dim):
        for t_idx, t in enumerate(inspect_instance):
            gam_single_grid = gamma_grid[t, :, :, k] ** plot_pow
            gam_single_graph = gamma_graph[t, :, k] ** plot_pow
            transmitter_coord = transmitter_coords[t, :, k]

            # Plot the graph with distances represented as colors
            ax = axs[k, t_idx]
            im = ax.imshow(gam_single_grid, origin='lower', extent=(0, grid_wid, 0, grid_wid), cmap=cmap, norm=norm)

            # Draw the graph with nodes colored by gamma
            nx.draw(G, pos, with_labels=False, node_color=gam_single_graph, cmap=cmap, node_size=70, ax=ax,
                    vmin=plot_val_range[0], vmax=plot_val_range[1], edgecolors='k')

            # Plot the transmitter coordinates
            ax.scatter(transmitter_coord[0], transmitter_coord[1], s=600, marker='*', color='orange', zorder=5, label=f'Transmitter {k + 1}')

            # Add instance label at bottom of each column
            if k == gamma_dim - 1:  # Only for bottom row
                # Place text manually below the subplot
                ax.text(0.5, 0.02, f'$t = {t}$', 
                    transform=ax.transAxes,
                    ha='center', 
                    va='top',
                    fontsize=25)

    # Tighten the layout first
    plt.tight_layout(pad=1.0)

    # Add a shared color bar on the top
    cax = fig.add_axes([0.15, 0.92, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=25)

    # Save the combined figure if required
    if save_fig:
        fig_path = os.path.join(res_path, 'gamma_vt.pdf')
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()

# Example usage
plot_pow = 1 / 3
plot_val_range = np.array([gamma_grid[inspect_instance, :, :, :].min(), gamma_grid[inspect_instance, :, :, :].max()]) ** plot_pow
plot_gam_combined(gamma_grid, gamma_graph, center_coords, plot_val_range, plot_pow, gamma_dim, save_fig=True, res_path=res_path)


# ------------------------------------------------------------
# plot the empirical distribution of p-values under different ranges of gamma's norm
# ------------------------------------------------------------

# Define the bins separately
gam_pval_bins = [0.2, 0.35, 0.5]  # For p-value distribution plots
gam_pi0_bins = [0.2, 0.25, 0.3, 0.35]  # For π₀ calculation

# ------------------------------------------------------------
# First compute and print π₀ values
# ------------------------------------------------------------
print("\nEmpirical π₀ by γ ranges:")
for i, (lower, upper) in enumerate(zip(gam_pi0_bins[:-1], gam_pi0_bins[1:])):
    hypotheses_in_range_all = []
    
    for t in range(n_instances):
        indices = (gamma_norm_graph[t,:] >= lower) & (gamma_norm_graph[t,:] < upper)
        hypotheses_in_range_all.extend(hypotheses[t,indices])
    
    hypotheses_in_range_all = np.array(hypotheses_in_range_all)
    proportion_null = np.sum(hypotheses_in_range_all == 0)/len(hypotheses_in_range_all)
    print(f"γ ∈ [{lower:.2f}, {upper:.2f}): π₀ = {proportion_null:.3f}")

# ------------------------------------------------------------
# Plot alternative p-value distributions
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(gam_pval_bins)-1))

for i, (lower, upper) in enumerate(zip(gam_pval_bins[:-1], gam_pval_bins[1:])):
    # Select p-values and hypotheses within current range
    p_vals_in_range_all = []
    hypotheses_in_range_all = []
    
    for t in range(n_instances):
        indices = (gamma_norm_graph[t,:] >= lower) & (gamma_norm_graph[t,:] < upper)
        p_vals_in_range_all.extend(p_values[t,indices])
        hypotheses_in_range_all.extend(hypotheses[t,indices])
    
    p_vals_in_range_all = np.array(p_vals_in_range_all)
    hypotheses_in_range_all = np.array(hypotheses_in_range_all)
    alt_p_vals_in_range = p_vals_in_range_all[hypotheses_in_range_all == 1]
    
    # Plot histogram
    ax.hist(alt_p_vals_in_range, bins=20, color=colors[i], alpha=0.7,
           edgecolor='black', density=True, range=(0,1),
           label=f'{lower}≤γ<{upper}')

ax.set_xlim(0, 1)
ax.set_xlabel('Alternative $p$-value', fontsize=20)
ax.set_ylabel('Density', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.legend(fontsize=20, title='γ ranges', title_fontsize=20)
ax.set_title('Alternative $p$-value Distributions by γ Ranges', fontsize=20)

if save_fig:
    fig_path = os.path.join(res_path, 'alt_pval_dist_overlap.pdf')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.show()


# ------------------------------------------------------------
# plot pi_0 value over (v, t)
# ------------------------------------------------------------

# Compute pi_0 values for (v, t)
null_threshold = 0.2


# Initialize storage only for inspected instances
pi0_grid_inspect = np.zeros((len(inspect_instance), grid_wid, grid_wid))
pi0_graph_inspect = np.zeros((len(inspect_instance), n_vertices))

# Compute pi0 only for inspected instances
for idx, t in enumerate(inspect_instance):
    # Compute for grid points
    for i in range(grid_wid):
        for j in range(grid_wid):
            gam_val = gamma_grid[t, i, j, :]
            pi0_grid_inspect[idx, i, j] = compute_pi0(gam_val, null_threshold)
    
    # Compute for graph nodes
    for v in range(n_vertices):
        x_idx = int(node_coords[v, 0])
        y_idx = int(node_coords[v, 1])
        pi0_graph_inspect[idx, v] = pi0_grid_inspect[idx, y_idx, x_idx]

# Create combined figure
n_cols = len(inspect_instance)
fig, axs = plt.subplots(1, n_cols, figsize=(8*n_cols, 8), 
                       gridspec_kw={'width_ratios': [1] * n_cols, 'wspace': -0.1})

# Normalization for color mapping
norm = plt.Normalize(vmin=0, vmax=1)
cmap = plt.cm.viridis

for idx, t in enumerate(inspect_instance):
    ax = axs[idx] if n_cols > 1 else axs  # Handle single column case
    
    # Plot grid background
    im = ax.imshow(pi0_grid_inspect[idx], origin='lower', 
                  extent=(0, grid_wid, 0, grid_wid),
                  cmap=cmap, norm=norm, alpha=0.7)
    
    # Overlay graph nodes
    nx.draw(G, pos, ax=ax, with_labels=False,
           node_color=pi0_graph_inspect[idx], cmap=cmap,
           node_size=70, vmin=0, vmax=1, 
           edgecolors='k')
    
    # Mark ALL transmitters for this time instance
    for k in range(gamma_dim):
        ax.scatter(center_coords[t, k, 0], center_coords[t, k, 1],
                  s=600, marker='*', color='orange', zorder=5,
                  label=f'Transmitter {k+1}' if idx == 0 else "")
    
    # Add time label
    ax.text(0.5, 0.02, f'$t = {t}$', transform=ax.transAxes,
           ha='center', va='top', fontsize=25)

# Add single colorbar on the top
cax = fig.add_axes([0.15, 0.92, 0.7, 0.03]) # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize=25)

# Add legend for transmitters (only once)
if gamma_dim > 1:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 0.9))

plt.tight_layout()

if save_fig:
    fig_path = os.path.join(res_path, 'pi0_vt.pdf')
    plt.savefig(fig_path, bbox_inches='tight')

plt.show()


# ------------------------------------------------------------
# plot pi_0 as a function of gamma (optimized for symmetry)
# ------------------------------------------------------------

# Define gamma range and resolution
gam_dom_min = 0
gam_dom_max = 0.5  # Hard-coded maximum as requested
n_points = 50
null_threshold = 0.2

# Create gamma values
gamma_vals = np.linspace(gam_dom_min, gam_dom_max, n_points)

# Initialize pi0 matrix (will fill lower triangle first)
pi0_matrix = np.zeros((n_points, n_points))

# Compute only lower triangular part (including diagonal)
for i in range(n_points):
    for j in range(i+1):  # Only go up to diagonal
        gam_val = np.array([gamma_vals[i], gamma_vals[j]])
        pi0_matrix[i, j] = compute_pi0(gam_val, null_threshold)
    
    # Print progress
    if (i+1) % 10 == 0:
        print(f"Completed {i+1}/{n_points} rows")

# Mirror lower triangle to upper triangle
pi0_matrix = np.triu(pi0_matrix.T, 1) + np.tril(pi0_matrix)

# Create the heatmap plot
plt.figure(figsize=(10, 8))
im = plt.imshow(pi0_matrix, origin='lower',
               extent=[gam_dom_min, gam_dom_max, gam_dom_min, gam_dom_max],
               cmap='viridis', aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('π₀ value', fontsize=25)
cbar.ax.tick_params(labelsize=25)

# Add title only (no axis labels)
plt.title(r'$\pi_0$ as function of γ values', fontsize=25)

# Remove axis labels
plt.xlabel('')
plt.ylabel('')
plt.xticks([gam_dom_min, gam_dom_max], [str(gam_dom_min), str(gam_dom_max)], fontsize=25)# 0 at start, 0.5 at end
plt.yticks([gam_dom_min, gam_dom_max], [str(gam_dom_min), str(gam_dom_max)], fontsize=25)# 0 at start, 0.5 at end

if save_fig:
    fig_path = os.path.join(res_path, 'pi0_vs_gamma.pdf')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)

plt.show()

print('Done!')