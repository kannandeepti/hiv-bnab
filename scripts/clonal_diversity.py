""" Script to visualize clonal diversity of B cells over time in GC / EGC."""

import os
import sys
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
import matplotlib.animation as animation 
import seaborn as sns
import colorsys
import functools

import hiv_code
from hiv_code.simulation import Simulation
from hiv_code import utils

import scripts
from scripts import SWEEP_DIR, PLOT_DIR
from scripts.analyze_sweep import epitope_colors, darken_color

slide_width = 11.5
half_slide_width = 5.67
aspect_ratio = 5/7
cm_in_inch = 2.54
pres_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 1.5,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'pdf.fonttype' : 42,
                  'ps.fonttype' : 42,
                  'figure.titlesize' : 20,
                  'figure.labelsize' : 20,
                  'axes.titlesize': 20,
                  'axes.labelsize': 18,
                  'legend.fontsize': 18,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'text.usetex': True,
                  'figure.figsize': [half_slide_width, half_slide_width * aspect_ratio],
                  'font.family': 'sans-serif',
                  'font.size' : 14,
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 4,
                  'xtick.major.size': 5,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 1.5,
                  'xtick.minor.width' : 1,
                  'xtick.minor.size' : 4,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 4,
                  'ytick.major.size': 5,
                  'ytick.major.width': 1.5,
                   'ytick.minor.width' : 1,
                  'ytick.minor.size' : 4,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}

paper_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 1.0,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'pdf.fonttype' : 42,
                  'ps.fonttype' : 42,
                  'figure.titlesize' : 8,
                  'figure.labelsize' : 8,
                  'axes.titlesize': 8,
                  'axes.labelsize': 8,
                  'legend.fontsize': 7,
                  'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'text.usetex': False,
                  'figure.figsize': [5.08 / cm_in_inch, 4.25 / cm_in_inch],
                  'font.family': 'sans-serif',
                   'font.sans-serif' : 'Arial',
                  'font.size' : 8,
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 2,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.5,
                  'xtick.minor.width' : 0.3,
                  'xtick.minor.size' : 2.5,
                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 2,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.5,
                   'ytick.minor.width' : 0.3,
                  'ytick.minor.size' : 2.5,
                  'ytick.minor.right':False,
                  'lines.linewidth':1.5}
plt.rcParams.update(paper_params)

PLOT_DIR = PLOT_DIR / "for_paper"

def extract_lineage_data(param_dir):
    """ Save a data structure that captures the number of clones in each lineage
    as a function of time in each GC, and for other B cell types. """
    lineage = {}
    replicates = [expdir for expdir in param_dir.iterdir() if expdir.is_dir()]
    params = utils.read_json(replicates[0]/'parameters.json')
    n_lineages = params["n_naive_precursors"]
    n_gc = params["n_gc"]
    history = utils.expand(utils.read_pickle(replicates[0]/'history.pkl'))
    time = np.arange(history['gc']['num_above_aff'].shape[0]) * params['tspan_dt']
    lineage['time'] = time
    ep_boundaries = np.concatenate(([0], history['ep_to_lineage']))
    ep_boundaries[-1] = n_lineages
    print(ep_boundaries)
    lineage['ep_boundaries'] = ep_boundaries
    cell_dummy_array = np.zeros((len(time), len(replicates), n_lineages))
    fields = ["memory_egc"]
    for field in fields:
        lineage[field] = np.copy(cell_dummy_array)
    lineage['gc'] = np.zeros((len(time), len(replicates), n_gc, n_lineages))

    for j, expdir in enumerate(replicates):
        history = utils.expand(utils.read_pickle(expdir/'history.pkl'))
        lineage['gc'][:, j, :, :] = history['gc']['num_by_lineage'] #lineage data from each GC
        for field in fields:
            lineage[field][:, j, :] = history[field]['num_by_lineage'] #lineage data for other cell types
    
    #write summary files to directory with this unique parameter set
    #utils.write_pickle(lineage, param_dir/'lineage.pkl')
    return lineage

# Function to create a consistent color map based on epitope
def create_color_map(ep_boundaries):
    """ Create an array assigning each naive precursor lineage to a color. Precursors
    that target the same epitope will be different shades of the same color. There are 8
    unique shades that repeat across distinct lineages."""

    #Assign each lineage to an epitope
    n_lineages = ep_boundaries[-1]
    print(n_lineages)
    epitope_labels = np.zeros(n_lineages, dtype=int)
    n_ep = len(ep_boundaries) - 1
    for i in range(n_ep):
        epitope_labels[ep_boundaries[i]:ep_boundaries[i+1]] = i
        
    # Base colors for the three epitopes
    base_colors = epitope_colors(n_ep) 
    shades = [sns.light_palette(base_colors[i], n_colors=8, reverse=False) for i in range(3)]  # 8 shades per epitope
    
    # Assign a color to each lineage
    color_map = []
    for i, epitope in enumerate(epitope_labels):
        # For each epitope, assign a specific shade (shade index can be i % 8)
        color = shades[epitope][i % 8]
        color_map.append(color)
    
    return np.array(color_map)

def plot_egc_pie_charts(lineage_data, ep_boundaries, time_points, 
                              replicate_indices, title):
    """ Plot pie charts at different `time_points` in simulation showing clonal
    diversity in example EGCs from simulation replicates indetified in `replicate_indices`."""

    n_time_points = len(time_points)
    n_rows = len(replicate_indices)
    fig_width = 11.5 / cm_in_inch
    fig, axs = plt.subplots(n_rows, n_time_points, figsize=(fig_width, fig_width / n_time_points * n_rows))
    
    # Set up color palette (for expanded clones) and grey for small clones
    color_map = create_color_map(ep_boundaries)
    
    for row_idx, replicate_idx in enumerate(replicate_indices):
        for col_idx, t in enumerate(time_points):
            # Get the clone data for the specific time point
            clone_data = lineage_data[t, replicate_idx, :]

            # Identify expanded clones (> 1) and lump others together
            expanded_clone_mask = clone_data > 1
            expanded_clones = clone_data[expanded_clone_mask]
            expanded_colors = color_map[expanded_clone_mask]

            lumped_small_clones = np.sum(clone_data[~expanded_clone_mask])

            # Data for pie chart: expanded clones and lumped small clones
            sizes = np.append(expanded_clones, lumped_small_clones)

            # Define the colors: Use palette for expanded clones, grey for small clones
            colors = list(expanded_colors) + ['lightgrey']

            # Create the pie chart
            axs[row_idx, col_idx].pie(sizes, colors=colors, startangle=90)
            axs[row_idx, col_idx].axis('equal')  # Ensure pie chart is circular

            # Add "Day" label only in the first row
            if row_idx == 0:
                axs[row_idx, col_idx].set_title(f"Day {t+1}")
            else:
                axs[row_idx, col_idx].set_title("")

            # Add EGC labels 
            axs[row_idx, 0].set_ylabel(f"EGC {replicate_idx + 1}", rotation=0, labelpad=15, va='center')

    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'egc_clonal_pie_charts_{title}.pdf')

# Function to plot clonal dominance for multiple rows of pie charts
def plot_gc_pie_charts(lineage_data, ep_boundaries, time_points, 
                                        replicate_indices, gc_indices, title):
    """ Plot pie charts at different `time_points` in simulation showing clonal
    diversity in example GCs identified by `replicate_indices` and `gc_indices`."""

    n_time_points = len(time_points)
    n_rows = len(replicate_indices)
    fig_width = 11.5 / cm_in_inch
    fig, axs = plt.subplots(n_rows, n_time_points, figsize=(fig_width, fig_width / n_time_points * n_rows))
        
    # Set up color palette (for expanded clones) and grey for small clones
    color_map = create_color_map(ep_boundaries)
    
    for row_idx, (replicate_idx, gc_idx) in enumerate(zip(replicate_indices, gc_indices)):
        for col_idx, t in enumerate(time_points):
            # Get the clone data for the specific time point, replicate, and GC
            clone_data = lineage_data[t, replicate_idx, gc_idx, :]
            
            expanded_clone_mask = clone_data > 1
            expanded_clones = clone_data[expanded_clone_mask]
            expanded_colors = color_map[expanded_clone_mask]
            
            lumped_small_clones = np.sum(clone_data[~expanded_clone_mask])
            
            sizes = np.append(expanded_clones, lumped_small_clones)
            colors = list(expanded_colors) + ['lightgrey']
            
            # Create the pie chart
            axs[row_idx, col_idx].pie(sizes, colors=colors, startangle=90)
            axs[row_idx, col_idx].axis('equal')  # Ensure pie chart is circular
            
            # Add "Day" label only in the first row
            if row_idx == 0:
                axs[row_idx, col_idx].set_title(f"Day {t+1}")
            else:
                axs[row_idx, col_idx].set_title("")
        
        # Add GC labels on the far left for each row
        axs[row_idx, 0].set_ylabel(f"Rep {replicate_idx + 1}\nGC {gc_idx+1}", rotation=0, labelpad=15, va='center')

    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'gc_clonal_pie_charts_{title}.pdf')

def plot_epitope_frequencies_violin(lineage_data, ep_boundaries, time_points, title):
    """ Calculate distribution of frequencies of GC B cells that target each epitope
    and make violin plots across 2000 independent GCs at different time points."""

    n_time_points, n_replicates, n_gcs, n_lineages = lineage_data.shape
    lineage_data = lineage_data.reshape(n_time_points, n_replicates * n_gcs, n_lineages)
    total_gcs = n_replicates * n_gcs
    n_ep = len(ep_boundaries) - 1
    
    # Prepare to plot
    fig, axs = plt.subplots(1, len(time_points), figsize=(slide_width, 4), sharey=True)

    for idx, t in enumerate(time_points):
        # Get the data for the current time point
        data_at_time_t = lineage_data[t, :, :]
        # Initialize lists to store frequencies for each epitope
        epitope_frequencies = np.zeros((total_gcs, n_ep))
        
        # Iterate over GCs and calculate frequencies for each epitope
        for gc in range(total_gcs):
            total_clones_in_gc = np.sum(data_at_time_t[gc, :])
            for i in range(n_ep):
                if total_clones_in_gc > 0:
                    clones_targeting_epitope = data_at_time_t[gc, ep_boundaries[i]:ep_boundaries[i+1]]
                    epitope_frequencies[gc, i] = np.sum(clones_targeting_epitope) / total_clones_in_gc
                else:
                    epitope_frequencies[gc, i] = 0. # If no clones in GC, set frequency to 0
        
        # Create a violin plot for the current time point
        sns.violinplot(data=[epitope_frequencies[:, i] for i in range(n_ep)], palette=epitope_colors(n_ep), ax=axs[idx])
        axs[idx].set_title(f"Day {t + 1}")
        axs[idx].set_xticklabels(['ep 1', 'ep 2', 'ep 3'])
        if idx == 0:
            axs[idx].set_ylabel('Frequency')

    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'epitope_frequencies_{title}.png')
    #epitope frequencies at the final time point
    return epitope_frequencies

def plot_dominant_clone_frequencies(lineage_data, time, day_400_stats, title):
    """ Calculate distribution of frequencies of the most dominant clone across 2000 
    independent GCs at different time points and plot as a function of time."""

    n_time_points, n_replicates, n_gcs, n_lineages = lineage_data.shape
    lineage_data = lineage_data.reshape(n_time_points, n_replicates * n_gcs, n_lineages)
    total_gcs = n_replicates * n_gcs
    n_ep = len(day_400_stats.keys())
    colors = epitope_colors(n_ep, ep_per_ag=3)
    # Prepare to plot
    fig, ax = plt.subplots()
    for i in range(n_ep):
        ep_data = lineage_data[:, day_400_stats[f'ep_{i + 1}']['indices'], :]
        dominant_clones = np.max(ep_data, axis=2)
        total_clones = np.sum(ep_data, axis=2)
        dominant_frequencies = dominant_clones / total_clones #shape (n_time_points, total_gcs)
        mean_frequencies = np.mean(dominant_frequencies, axis=1)   # Shape: (n_time_points,)
        q25_frequencies = np.percentile(dominant_frequencies, 25, axis=1)  # 25th percentile
        q75_frequencies = np.percentile(dominant_frequencies, 75, axis=1) #75th percentile
        ax.plot(time, mean_frequencies, color=colors[i], label=f'ep {i+1}')
        ax.fill_between(time, q25_frequencies, q75_frequencies, color=colors[i], alpha=0.3, label=None)

    ax.legend()
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Dominant clone frequency')
    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'dominant_clone_frequencies_by_epitope_{title}.png')

def plot_time_clone_became_dominant(lineage_data, time, day_400_stats, title, 
                                    num_traj=5, dominance_thresh=0.5):
    """ For each GC, determine the time point at which the most dominant clone at
    day 400 took over the GC (i.e. reached > 50% of the population). Plot distribution of these
    time points as a violin plot for 3 subsets of GCs where epitope 1, 2, or 3 wins.
    
    Also in a separate plot, plot the frequency of the winning clone over time averaged over GCs.
    """

    n_time_points, n_replicates, n_gcs, n_lineages = lineage_data.shape
    lineage_data = lineage_data.reshape(n_time_points, n_replicates * n_gcs, n_lineages)
    total_gcs = n_replicates * n_gcs
    n_ep = len(day_400_stats.keys())
    colors = epitope_colors(n_ep, ep_per_ag=3)

    fig, ax = plt.subplots()
    time_of_dominance = []
    for i in range(n_ep):
        ep_data = lineage_data[:, day_400_stats[f'ep_{i + 1}']['indices'], :]
        ep_dom_clone_dynamics = np.zeros((n_time_points, ep_data.shape[1]))
        time_of_dominance_ep = []
        for gc_idx in range(ep_data.shape[1]):
            #find index of clone that eventually wins by day 400
            win_idx = np.argmax(ep_data[-1, gc_idx, :])
            #frequency of winning clone within GC over time
            freq_dynamics = ep_data[:, gc_idx, win_idx] / np.sum(ep_data[:, gc_idx, :], axis=1)
            ep_dom_clone_dynamics[:, gc_idx] = freq_dynamics
            #time where frequency of winning clone reaches > dominance_thresh
            time_idx = np.searchsorted(freq_dynamics, dominance_thresh)
            time_of_dominance_ep.append(time_idx)
        mean_frequencies = np.mean(ep_dom_clone_dynamics, axis=1) #(n_time_pts, )
        trajs = ep_dom_clone_dynamics.shape[1]
        traj_inds = np.random.randint(0, trajs, size=num_traj)
        ax.plot(time[0:200], ep_dom_clone_dynamics[0:200, traj_inds], lw=.3, color=colors[i], label='_nolegend_')
        #q25_frequencies = np.percentile(ep_dom_clone_dynamics, 25, axis=1)  # 25th percentile
        #q75_frequencies = np.percentile(ep_dom_clone_dynamics, 75, axis=1) #75th percentile
        ax.plot(time[0:200], mean_frequencies[0:200], color=darken_color(colors[i], 0.1), label=f'ep {i+1}')
        #ax.fill_between(time[0:200], q25_frequencies[0:200], q75_frequencies[0:200], color=colors[i], alpha=0.3, label=None)
        time_of_dominance.append(time_of_dominance_ep)
    
    ax.legend()
    ax.set_xlabel('Time (days)')
    ax.set_title('Winning clone frequency')
    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'winning_clone_frequencies_by_epitope_{title}.pdf')

    #fig, ax = plt.subplots()
    #sns.violinplot(data=time_of_dominance, palette=colors, ax=ax)
    #ax.set_xticklabels(['ep 1', 'ep 2', 'ep 3'])
    #ax.set_ylabel('Time to dominate (days)')
    #fig.tight_layout()
    #plt.savefig(PLOT_DIR/f'winning_clone_time_to_dominate_thresh_{dominance_thresh}_{title}.png')

def plot_egc_clonal_diversity(lineage_data, ep_boundaries, title):
    """ Make a violin plot showing distribution of frequency of each clone within each epitope class
    at day 400 in EGC. """   
    in_class_freq_data = []
    n_ep = len(ep_boundaries) - 1
    for i in range(n_ep):
        ep_data = lineage_data[-1, :, ep_boundaries[i]:ep_boundaries[i+1]]
        # Mask for expanded clones (replace non-expanded with zero)
        mask = ep_data > 1
        expanded_clones = np.where(mask, ep_data, 0)
        # Calculate frequency of each clone within this class
        clone_sums = np.sum(expanded_clones, axis=1, keepdims=True)
        ep_frequencies = np.divide(expanded_clones, clone_sums, where=clone_sums > 0)
        dominant_clones = np.max(ep_frequencies, axis=1)
        in_class_freq_data.append(dominant_clones.flatten())
    
    #make violin plot 
    fig, ax = plt.subplots()
    sns.violinplot(data=in_class_freq_data, palette=epitope_colors(n_ep), ax=ax)
    ax.set_xticklabels([f'ep {i}' for i in range(1, len(in_class_freq_data) + 1)])
    ax.set_title('Dominant clone frequency \nwithin epitope class in EGC')
    fig.tight_layout()
    plt.savefig(PLOT_DIR/f'egc_within_class_clonal_diversity_d400_{title}.pdf')

def make_lineage_plots(param_dir, title):
    """ Save lineage data, read from the data and make violin / pie chart plots illustrating
    clonal dynamics as a function of time. """
    lineage = extract_lineage_data(param_dir)
    time_points_to_plot = [2, 9, 99, 399]
    epitope_frequencies = plot_epitope_frequencies_violin(lineage['gc'], lineage['ep_boundaries'], 
                                                          time_points_to_plot, title)
    # calculate fraction of GC's across all replicates where each epitope wins
    n_gcs, n_ep = epitope_frequencies.shape
    winning_class = []
    for gc in range(n_gcs):
        winning_class.append(np.argmax(epitope_frequencies[gc, :]))
    winning_class = np.array(winning_class)
    day_400_stats = {} #% of GCs where each epitope "wins"
    replicate_idxs = []
    gc_idxs = []
    for i in range(n_ep):
        day_400_stats[f'ep_{i + 1}'] = {}
        day_400_stats[f'ep_{i + 1}']['fraction_won'] = np.sum(winning_class == i) / n_gcs
        #extract the replicate index and GC index of a GC where this epitope won
        index = np.where(winning_class == i)[0]
        day_400_stats[f'ep_{i + 1}']['indices'] = index
        #choose an example GC where this epitope won and record its replicate/GC index
        replicate_idxs.append(int(np.floor(index[3] / 200)))
        gc_idxs.append(int(index[2] % 200))
        print(f'Number of GCs dominanted by epitope {i+1}: {np.sum(winning_class == i) / n_gcs}')
    
    #plot clonal dominance stratefied winner of GC
    plot_time_clone_became_dominant(lineage['gc'], lineage['time'], day_400_stats, title)
    plot_egc_clonal_diversity(lineage['memory_egc'], lineage['ep_boundaries'], title)

    #plot pie charts
    time_indices_gc = [6, 29, 36, 44, 59, 99]  # Replace with your specific time points
    time_indices_egc = [6, 29, 44, 59, 99, 199]  # Replace with your specific time points
    plot_gc_pie_charts(lineage['gc'], lineage['ep_boundaries'], time_indices_gc, 
                            replicate_idxs, gc_idxs, title)
    plot_egc_pie_charts(lineage['memory_egc'], lineage['ep_boundaries'], time_indices_egc, 
                         [0, 1, 2], title)

if __name__ == "__main__":
    param_dir = sys.argv[1]
    title = sys.argv[2]
    make_lineage_plots(SWEEP_DIR/param_dir, title)
    #extract_lineage_data(SWEEP_DIR/param_dir) #NOT WORKING UGH