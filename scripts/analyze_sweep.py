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
from scripts import SWEEP_DIR, PLOT_DIR

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
                  'figure.titlesize' : 20,
                  'figure.labelsize' : 20,
                  'axes.titlesize': 20,
                  'axes.labelsize': 18,
                  'legend.fontsize': 18,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'text.usetex': False,
                  'figure.figsize': [half_slide_width, half_slide_width * aspect_ratio],
                  'font.family': 'sans-serif',
                   'font.sans-serif' : 'Helvetica',
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

def darken_color(color, factor):
    """
    Darkens the given color by the specified factor.
    """
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0, l - factor), s)

def epitope_colors(n_ep, ep_per_ag=3, n_ag=2):
    """ Return the color sequence for a given number of epitopes """
    if ep_per_ag == 3:
        colors = sns.color_palette("Set2", 6)
        if n_ep == 3:
            return colors[0:3]
        elif n_ep == 4:
            order = [0, 4, 1, 2]
        elif n_ep == 5:
            order = [0, 1, 4, 3, 2]
        elif n_ep == 6:
            order = [0, 1, 2, 4, 3, 5]
        else:
            return colors
        return [colors[i] for i in order]
    elif ep_per_ag == 6:
        n_conserved_ep = n_ag * ep_per_ag - n_ep
        n_variable_ep = ep_per_ag - n_conserved_ep
        colors = sns.color_palette("Paired", 12)
        dom_ag = colors[-1::-2] 
        subdom_ag = colors[-2::-2] 
        variable_ep = np.concatenate((dom_ag[0:n_variable_ep], subdom_ag[0:n_variable_ep]))
        conserved_ep = dom_ag[(len(dom_ag) - n_conserved_ep):]
        if len(variable_ep) == 0:
            return conserved_ep
        elif len(conserved_ep) == 0:
            return variable_ep
        else:
            return np.concatenate((variable_ep, conserved_ep))
    else:
        return ValueError("ep_per_ag must either be 3 or 6.")

def resort_directories(sweep_dir):
    """ Make sure all expirement directories within sweep_i directories have the same parameters.
    If not, resort the directories so that this structure is maintained. """

    sweep_dir = Path(sweep_dir)
    mapping_df = map_simdir_to_param(SWEEP_DIR/sweep_dir)
    group_keys = [col for col in mapping_df.columns if col not in ['path', 'seed']]
    dfs = []
    for param_dir in sweep_dir.iterdir():
        if param_dir.is_dir():
            for exp_dir in param_dir.iterdir():
                if exp_dir.is_dir():
                    #extract parameters used in this simulation
                    params = utils.read_json(exp_dir/f"parameters.json")
                    mapping_dict = {}
                    for key in params:
                        if key == "E1hs":
                            n_ep = params["n_conserved_epitopes"] + params["n_ag"] * params["n_variable_epitopes"]
                            for i in range(n_ep):
                                mapping_dict[f"E{i+1}h"] = np.round(params["E1hs"][i], decimals=1)
                        elif key == "naive_target_fractions":
                            for i in range(len(params[key])):
                                #naive target fraction (ntf)
                                mapping_dict[f"ntf{i+1}"] = params["naive_target_fractions"][i]
                        elif key == "f_ag":
                            for i in range(len(params[key])):
                                #naive target fraction (ntf)
                                mapping_dict[f"f_ag{i+1}"] = params["f_ag"][i]
                        elif key != "experiment_dir" and key != "updated_params_file":
                            value = params[key]
                            mapping_dict[key] = tuple(value) if isinstance(value, list) else value
                    #Identify which sweep directory this experiment belongs in based on parameters
                    mask = pd.Series([True] * len(mapping_df))
                    for key, value in mapping_dict.items():
                        if key in group_keys:
                            mask &= (mapping_df[key] == value)
                    matching_df = mapping_df[mask]
                    print(param_dir)
                    print(matching_df['path'])
                    assert(len(matching_df['path'].values) == 1)
                    matching_sweep_dir = matching_df["path"].values[0]
                    exp_dir.rename(matching_sweep_dir / exp_dir.name)
                    #path to directory containing replicates of the same parameter set
                    mapping_dict["path"] = exp_dir
                    dfs.append(mapping_dict)

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
                elif key == "naive_target_fractions":
                    for i in range(len(swept_params[key])):
                        #naive target fraction (ntf)
                        mapping_dict[f"ntf{i+1}"] = swept_params["naive_target_fractions"][i]
                elif key == "f_ag":
                    for i in range(len(swept_params[key])):
                        #naive target fraction (ntf)
                        mapping_dict[f"f_ag{i+1}"] = swept_params["f_ag"][i]
                elif key != "experiment_dir":
                    mapping_dict[key] = swept_params[key]
            #path to directory containing replicates of the same parameter set
            mapping_dict["path"] = param_dir
            dfs.append(mapping_dict)

    df = pd.DataFrame(dfs)
    return df

def get_unmasked_ag(ig_conc, ag_conc, ka):
    term1 = ag_conc + ig_conc + 1/ka
    term2 = np.emath.sqrt(np.square(term1) - 4 * ag_conc * ig_conc)
    IC = (term1 - term2) / 2
    IC = np.nan_to_num(IC, 0)
    return (ag_conc - IC)

def collapse_replicates(param_dir):
    """ Record data from all replicates in one data structure, results, which we write
    to a pickle file."""

    #loop through subdirectories of each path, extract concentrations and average them
    results = {}
    #assumes all simulation directories within param_dir have the same parameters except for random seed
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
    fields = ["plasma_gc", "memory_gc", "plasma_egc", "memory_egc", "gc", "naive_entry", "memory_reentry"]
    for field in fields:
        results[field] = np.copy(dummy_array)
    concentrations = ["ic_fdc_conc", "ab_conc", "ab_ka", "ic_fdc_eff_conc", "titer"]
    for conc in concentrations:
        results[conc] = np.copy(dummy_array)
    mean_fn = functools.partial(np.mean, axis=1)
    
    for j, expdir in enumerate(replicates):
        history = utils.expand(utils.read_pickle(expdir/'history.pkl'))
        fdc_conc = history['conc']['ic_fdc_conc']
        unmasked_fdc_conc = history['conc']['ic_fdc_eff_conc']
        ab_conc = history['conc']['ab_conc']
        ka = history['conc']['ab_ka']

        for i in range(n_ep):
            #record concentration dynamics from each replicate
            results["ic_fdc_conc"][:, j, i] = fdc_conc[:, i]
            results["ab_conc"][:, j, i] = ab_conc[:, i]
            results["ab_ka"][:, j, i] = ka[:, i]
            results["ic_fdc_eff_conc"][:, j, i] = unmasked_fdc_conc[:, i]
            results["titer"][:, j, i] = (ka * ab_conc)[:, i]
            #count number of B cells above affinity 10^6
            for field in fields[:-3]:
                results[field][:, j, i] = history[field]['num_above_aff'][:, 0, i, 0]
            #count mean # GC B cells that target each epitope in each replicate
            results["gc"][:, j, i] = mean_fn(history['gc']['num_above_aff'][:, :, 0, i, 0])
            #count mean # naive B cells that enter GC per day 
            results["naive_entry"][:, j, i] = mean_fn(history['gc_entry']['total_num'][:, :, 0, i, 0])
            #count mean # memory B cells that re-enter GC per day
            results["memory_reentry"][:, j, i] = mean_fn(history['gc_entry']['total_num'][:, :, 0, i, 1])
    
    #write summary files to directory with this unique parameter set
    utils.write_pickle(results, param_dir/'conc_cells.pkl')

def plot_averages(df, sweep_dir, ep_per_ag):
    """ Read from summary file and make plots. """
    #first figure out which parameters were swept, based on number of unique values in each column
    params = list(df.columns)
    params.remove('path')
    df_excl_path = df[params]
    unique_counts = np.array(df_excl_path.nunique())
    swept_params = np.array(params)[unique_counts > 1]
    bcell_types = ["memory_gc", "plasma_gc", "plasma_egc"]
    concentrations = ["ab_conc", "ab_ka"]
    conc_titles = ['[Ab] (nM)', '$K_a$ (nM$^{-1}$)']
    n_ep = int(df['n_conserved_epitopes'].iloc[0] + df['n_variable_epitopes'].iloc[0] * df['n_ag'].iloc[0])
    stats_list = []
    print(swept_params)
    mean_fn = functools.partial(np.mean, axis=1)
    for path, mat in df.groupby('path'):
        res_file = Path(path)/'conc_cells.pkl'
        if res_file.exists():
            #for parameters that were swept, include their values in the figure title / file name
            plotname = ''
            for key in swept_params:
                plotname += f'_{key}_{mat[key].iloc[0]}'
            plottitle = ''
            nparams = len(swept_params)
            for i in range(nparams):
                plottitle += f'{swept_params[i]}={mat[swept_params[i]].iloc[0]}, '
            
            results = utils.read_pickle(res_file)
            nreplicates = results["plasma_gc"].shape[1]
            plottitle += f'replicates={nreplicates}'
            time = results["time"]
            
            """ PLOT PLASMA B CELL NUMBER, [Ab], Ka OVER TIME """
            fig = plt.figure(figsize=(9.5, 8.0))
            plot_idx = 1
            colors = epitope_colors(n_ep, ep_per_ag=ep_per_ag)
            print(len(colors))
            plt.subplot(3, 2, plot_idx)
            for i in range(n_ep):
                plt.plot(time, results["ic_fdc_eff_conc"][:, :, i], lw=.3, color=colors[i], label='_nolegend_')
                plt.plot(time, mean_fn(results["ic_fdc_eff_conc"][:, :, i]), color=darken_color(colors[i], 0.2), lw=2)
            #plt.yscale('log')
            plt.title("Unmasked [IC-FDC]")
            #plt.ylim([1e-5, 1e2])
            plt.legend([f'ep{i+1}' for i in range(n_ep)], loc="lower right")
            plot_idx += 1
            for bcell in bcell_types:
                plt.subplot(3, 2, plot_idx)
                for i in range(n_ep):
                    plt.plot(time, results[bcell][:, :, i], lw=.3, color=colors[i], label='_nolegend_')
                    plt.plot(time, mean_fn(results[bcell][:, :, i]), color=darken_color(colors[i], 0.2), lw=2)
                plt.yscale('log')
                plt.title(f'{bcell} bcells with $K_a > 10^6M$')
                if bcell == "memory_gc":
                    plt.ylim([1e0, 1e5])
                else:
                    plt.ylim([1e0, 1e4])
                #plt.legend([f'ep{i+1}' for i in range(n_ep)], loc="lower right")
                plot_idx += 1
            for j, conc in enumerate(concentrations):
                plt.subplot(3, 2, plot_idx)
                for i in range(n_ep):
                    plt.plot(time, results[conc][:, :, i], lw=.3, color=colors[i], label='_nolegend_')
                    plt.plot(time, mean_fn(results[conc][:, :, i]), color=darken_color(colors[i], 0.2), lw=2)
                plt.yscale('log')
                plt.title(conc_titles[j])
                plt.ylim([1e-4, 1e3])
                #plt.legend([f'ep{i+1}' for i in range(n_ep)], loc="lower right")
                plot_idx += 1
            fig.suptitle(plottitle)
            fig.supxlabel('time (days)')
            fig.tight_layout()
            plt.savefig(PLOT_DIR/f"{sweep_dir}_bcells_concs{plotname}_3rows.png")
            
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
    ep_per_ag = sys.argv[2]
    #resort_directories(SWEEP_DIR/sweep_dir)
    df = map_simdir_to_param(SWEEP_DIR/sweep_dir)
    print(df['path'])
    df = plot_averages(df, sweep_dir, int(ep_per_ag))
    df.to_csv((SWEEP_DIR/sweep_dir)/'sweep_map.csv', index=False)