""" Script to animate B cell affinity distributions over time."""

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
from scripts.analyze_sweep import epitope_colors

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

def save_distributions(param_dir):
    """ Record data from all replicates in one data structure, which contains the occupation
    numbers of B cells in each affinity bin as a function of time, across replicates."""
    
    aff_distr = {}
    #assumes all simulation directories within param_dir have the same parameters except for random seed
    replicates = [expdir for expdir in param_dir.iterdir() if expdir.is_dir()]
    nreplicates = len(replicates)
    if nreplicates == 0:
        return
    history = utils.expand(utils.read_pickle(replicates[0]/'history.pkl'))
    parameters = utils.read_json(replicates[0]/'parameters.json')
    n_ep = parameters["n_conserved_epitopes"] + parameters["n_ag"] * parameters["n_variable_epitopes"]
    affinity_bins = parameters["affinity_bins"]
    aff_distr['affinity_bins'] = affinity_bins
    time = np.arange(history['gc']['num_above_aff'].shape[0]) * parameters['tspan_dt']
    aff_distr['time'] = time
    #make arrays thata are time x n_replicates x n_epitopes 
    dummy_array = np.zeros((len(time), len(replicates), n_ep, len(affinity_bins))) 
    fields = ["plasma_gc", "memory_gc", "plasma_egc", "memory_egc"]
    for field in fields:
        aff_distr[field] = np.copy(dummy_array)
    
    for j, expdir in enumerate(replicates):
        history = utils.expand(utils.read_pickle(expdir/'history.pkl'))
        for i in range(n_ep):
            #count number of B cells in each affinity bin
            for field in fields:
                aff_distr[field][:, j, i, :] = history[field]['num_in_aff'][:, 0, i, :]
    
    #write summary files to directory with this unique parameter set
    utils.write_pickle(aff_distr, param_dir/'aff_distr.pkl')
    return aff_distr

def animate_Bcell_affinity_distributions(aff_distr, field, title):
    n_ep = aff_distr["plasma_gc"].shape[2]
    time = aff_distr["time"]
    colors = epitope_colors(n_ep)
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        t_idx = frame
        for i in range(n_ep):
            ax.plot(aff_distr['affinity_bins'], aff_distr[field][t_idx, :, i, :].mean(axis=0), lw=2, label=f'ep {i+1}', color=colors[i])
            ax.fill_between(aff_distr['affinity_bins'], aff_distr[field][t_idx, :, i, :].min(axis=0), aff_distr[field][t_idx, :, i, :].max(axis=0),
                            color=colors[i], alpha=0.3, label=None)
        ax.set_xlabel('Affinity')
        ax.set_ylabel('Occupation Number')
        #choose y limit
        max_val = aff_distr[field].max()
        ylim = np.round(max_val / 500) * 500
        ax.set_ylim(0, ylim)
        ax.set_title(f'{field} cells (Day {t_idx})')
        ax.legend(loc="upper left")
        fig.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(time), repeat=False)
    ani.save(PLOT_DIR/f'{title}_{field}_aff_distribution.gif', writer='imagemagick', fps=10)  

if __name__ == "__main__":
    param_dir = sys.argv[1]
    title = sys.argv[2]
    aff_distr = save_distributions(SWEEP_DIR/param_dir)
    for field in ["plasma_gc", "memory_gc", "plasma_egc", "memory_egc"]:
        animate_Bcell_affinity_distributions(aff_distr, field, title)