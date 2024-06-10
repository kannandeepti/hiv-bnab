""" Script to analyze a parameter sweep and generate some plots."""

import os
import sys
from pathlib import Path
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
import colorsys
import functools

import hiv_code
from hiv_code.simulation import Simulation
from hiv_code import utils

slide_width = 11.5
half_slide_width = 5.67
aspect_ratio = 5/7
pres_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 1.5,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'pdf.fonttype' : 42,
                  'ps.fonttype' : 42,
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
plt.rcParams.update(pres_params)

SWEEP_DIR = Path('/home/gridsan/dkannan/git-remotes/gc_dynamics/sweeps')
PLOT_DIR = Path('/home/gridsan/dkannan/git-remotes/gc_dynamics/plots')

def darken_color(color, factor):
    """
    Darkens the given color by the specified factor.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0, l - factor), s)

def map_simdir_to_param(sweepdir):
    """ Create a DataFrame which contains the relevant parameter values from the simulation
    (read from the sweep.json files) along with the path to the directory containing all 
    the simulation replicates for that parameter set.
    
    """
    dfs = []
    for param_dir in sweepdir.iterdir():
        if param_dir.is_dir():
            #read the first json file -> make each key a column in the DataFrame
            swept_params = utils.read_json(param_dir/f"{param_dir.name}_0.json")
            #collapse replicates -> write summary file with relevant cell counts / concentrations
            collapse_replicates(param_dir)
            mapping_dict = {}
            for key in swept_params:
                if key == "E1hs":
                    n_ep = swept_params["n_conserved_epitopes"] + swept_params["n_ag"] * swept_params["n_variable_epitopes"]
                    for i in range(n_ep):
                        mapping_dict[f"E{i+1}h"] = np.round(swept_params["E1hs"][i], decimals=1)
                elif key != "experiment_dir":
                    mapping_dict[key] = swept_params[key]
            #path to directory containing replicates of the same parameter set
            mapping_dict["path"] = param_dir
            dfs.append(mapping_dict)

    df = pd.DataFrame(dfs)
    return df

def collapse_replicates(param_dir):
    """ Record data from all replicates in one data structure, results, which we write
    to a pickle file."""

    #loop through subdirectories of each path, extract concentrations and average them
    #other summary statistics? 
    #write to a summary file
    results = {}
    replicates = [expdir for expdir in param_dir.iterdir() if expdir.is_dir()]
    nreplicates = len(replicates)
    if nreplicates == 0:
        return
    history = utils.expand(utils.read_pickle(replicates[0]/'history.pkl'))
    parameters = utils.read_json(replicates[0]/'parameters.json')
    n_ep = parameters["n_conserved_epitopes"] + parameters["n_ag"] * parameters["n_variable_epitopes"]
    time = np.arange(history['gc']['num_above_aff'].shape[0]) * parameters['tspan_dt']
    results['time'] = time
    #make arrays thata are time x n_replicates x n_epitopes 
    dummy_array = np.zeros((len(time), nreplicates, n_ep))
    fields = ["plasma_gc", "memory_gc", "plasma_egc", "memory_egc", "gc"]
    for field in fields:
        results[field] = np.copy(dummy_array)
    concentrations = ["ab_conc", "ab_ka", "titer"]
    for conc in concentrations:
        results[conc] = np.copy(dummy_array)
    mean_fn = functools.partial(np.mean, axis=1)
    
    for j, expdir in enumerate(replicates):
        history = utils.expand(utils.read_pickle(expdir/'history.pkl'))
        ab_conc = history['conc']['ab_conc']
        ka = history['conc']['ab_ka']

        for i in range(n_ep):
            #record concentration dynamics from each replicate
            results["ab_conc"][:, j, i] = ab_conc[:, i]
            results["ab_ka"][:, j, i] = ka[:, i]
            results["titer"][:, j, i] = (ka * ab_conc)[:, i]
            #count number of B cells above affinity 10^6
            for field in fields[:-1]:
                results[field][:, j, i] = history[field]['num_above_aff'][:, 0, i, 0]
            #count mean # GC B cells that target each epitope in each replicate
            results["gc"][:, j, i] = mean_fn(history['gc']['num_above_aff'][:, :, 0, i, 0])
    
    #write summary files to directory with this unique parameter set
    utils.write_pickle(results, param_dir/'conc_cells.pkl')

def plot_averages(df):
    """ Read from summary file and make plots. """
    #first figure out which parameters were swept, based on number of unique values in each column
    params = list(df.columns)
    params.remove('path')
    df_excl_path = df[params]
    unique_counts = np.array(df_excl_path.nunique())
    swept_params = np.array(params)[unique_counts > 1]
    bcell_types = ["plasma_gc", "plasma_egc"]
    concentrations = ["ab_conc", "ab_ka"]
    conc_titles = ['[Ab] (nM)', '$K_a$ (nM$^{-1}$)']
    n_ep = int(df['n_conserved_epitopes'].iloc[0] + df['n_variable_epitopes'].iloc[0] * df['n_ag'].iloc[0])
    stats_list = []
    print(swept_params)
    mean_fn = functools.partial(np.mean, axis=1)
    for path, mat in df.groupby('path'):
        plotname = ''
        for key in swept_params:
            plotname += f'_{key}_{mat[key].iloc[0]}'
        plottitle = ''
        nparams = len(swept_params)
        for i in range(nparams):
            plottitle += f'{swept_params[i]}={mat[swept_params[i]].iloc[0]}, '
        
        results = utils.read_pickle(Path(path)/'conc_cells.pkl')
        nreplicates = results["plasma_gc"].shape[1]
        plottitle += f'replicates={nreplicates}'
        time = results["time"]
        
        """ PLOT PLASMA B CELL NUMBER, [Ab], Ka OVER TIME """
        fig = plt.figure(figsize=(9.5, 5.75))
        plot_idx = 1
        colors = sns.color_palette("Set2", n_ep)
        for bcell in bcell_types:
            plt.subplot(2, 2, plot_idx)
            for i in range(n_ep):
                plt.plot(time, results[bcell][:, :, i], lw=.3, color=colors[i], label='_nolegend_')
                plt.plot(time, mean_fn(results[bcell][:, :, i]), color=darken_color(colors[i], 0.2), lw=2)
            plt.yscale('log')
            plt.title(f'{bcell} bcells with $K_a > 10^6M$')
            plt.ylim([1e0, 1e7])
            plt.legend([f'ep{i+1}' for i in range(n_ep)], loc="lower right")
            plot_idx += 1
        for j, conc in enumerate(concentrations):
            plt.subplot(2, 2, plot_idx)
            for i in range(n_ep):
                plt.plot(time, results[conc][:, :, i], lw=.3, color=colors[i], label='_nolegend_')
                plt.plot(time, mean_fn(results[conc][:, :, i]), color=darken_color(colors[i], 0.2), lw=2)
            plt.yscale('log')
            plt.title(conc_titles[j])
            plt.ylim([1e-5, 1e3])
            plt.legend([f'ep{i+1}' for i in range(n_ep)], loc="lower right")
            plot_idx += 1
        fig.suptitle(plottitle)
        fig.supxlabel('time (days)')
        fig.tight_layout()
        plt.savefig(PLOT_DIR/f"bcells_concs{plotname}.png")
        
        """ Compute mean GC size / titer against each epitope at end of simulation """
        summary_stats = {'path' : path}
        for i in range(n_ep):
            summary_stats[f'ep{i+1}_titer_d400'] = np.mean(results["titer"][-1, :, i])
        summary_stats['gc_size_d400'] = np.sum(np.mean(results["gc"][-1, :, :], axis=0)) #take last day, sum over n_ep
        stats_list.append(summary_stats)
    
    stats_df = pd.DataFrame(stats_list)
    df = pd.merge(df, stats_df, on='path', how='left')
    return df

if __name__ == "__main__":
    sweep_dir = sys.argv[1]
    df = map_simdir_to_param(SWEEP_DIR/sweep_dir)
    df = plot_averages(df)
    df.to_csv((SWEEP_DIR/sweep_dir)/'sweep_map.csv', index=False)