"""
Script to vary the number of shared epitopes between 2 antigens
and run associated affinity maturation simulations.
"""
from pathlib import Path
import subprocess
import yaml
import sys
import os
import numpy as np
from scipy.optimize import fsolve

sys.path.append(os.getcwd())
from scripts import SWEEP_DIR, INPUT_SUFFIX, clean_up_log_files
from hiv_code import utils

def geometric_sequence_sum(n, r):
    """Returns the sum of the first n terms of a geometric sequence."""
    return (1 - r**n) / (1 - r)

def naive_target_fractions(n_ep, min_precursors, max_precursors):
    """ Return fraction of naive precursors that target each of `n_ep` epitopes.
    The naive target fractions will follow a geometric sequence: a, a*r, a*r^2, .., a*r^(n-1).
    The sequence should sum to 1 and also the least immunodminant class should have at least
    min_precursors, and the most immunodominant class should have at most max_precursors.
    
    Args:
        n_ep : (int) number of precursors
        min_precursors : (int) min number of precursors targeting least immunodominant ep
        max_precursors : (int) max number of precursors targeting most immunodominant ep
    
    Returns:
        naive_precursors : (int) total # precursors across epitopes
        naive_target_fractions : list of precursor frequencies for each epitope (should sum to 1)
    
    """
    #r is the parameter that defines the geometric sequence
    r = (max_precursors / min_precursors)**(1 / (n_ep - 1))
    a = (1 - r) / (1 - r**n_ep)
    naive_precursors = min_precursors / a
    #round to nearest integer
    naive_precursors = int(np.rint(naive_precursors))
    ntfs = np.array([a * r**n for n in range(n_ep)])
    ntfs /= ntfs.sum()
    return naive_precursors, ntfs

def naive_target_fractions_fixed_total(n_ep, min_precursors, naive_precursors):
    """ Same as above, except this time, fix the total number of precursors and calculate
    r in the geometric series based on the constraint that a = min_precursors / naive_precursors. """
    a = min_precursors / naive_precursors
    #calculate r from a
    def equation(r):
        return a * geometric_sequence_sum(n_ep, r) - 1
    initial_guess = 1.1  # An initial guess for r
    r = fsolve(equation, initial_guess)[0]
    ntfs = np.array([a * r**n for n in range(n_ep)])
    ntfs /= ntfs.sum()
    return ntfs

n_ag = 2
ep_per_ag = 6
min_precursors = 200
max_precursors = 1300
sweep_name = str(sys.argv[1]) 
sweep_file = sweep_name + INPUT_SUFFIX
#DIR = SWEEP_DIR/f"{n_ag * ep_per_ag}_epitope_sweep"

#first calculate the total naive precursors based on max number of epitopes
naive_precursors, ntfs = naive_target_fractions(ep_per_ag * n_ag, min_precursors, max_precursors)
print(f"Total naive precursors: {naive_precursors}")

for i in range(ep_per_ag, ep_per_ag * n_ag + 1):
    """ Let i be the total number of unique epitopes """
    n_conserved_ep = n_ag * ep_per_ag - i
    n_variable_ep = ep_per_ag - n_conserved_ep
    E1hs = np.linspace(6.0, 7.0, n_ag * ep_per_ag)
    dom_ag = E1hs[-1::-2] #6 E1hs towards Ag 1
    subdom_ag = E1hs[-2::-2] #6 E1hs towards Ag 2
    variable_ep_E1h = np.concatenate((dom_ag[0:n_variable_ep], subdom_ag[0:n_variable_ep]))
    conserved_ep_E1h = dom_ag[(len(dom_ag) - n_conserved_ep):]
    E1h = np.concatenate((variable_ep_E1h, conserved_ep_E1h))
    target_fractions = naive_target_fractions_fixed_total(i, min_precursors, naive_precursors)
    #this is what I should have done...
    dom_ag = target_fractions[-1::-2]
    subdom_ag = target_fractions[-2::-2]
    variable_ep_ntf = np.concatenate((dom_ag[0:n_variable_ep], subdom_ag[0:n_variable_ep]))
    conserved_ep_ntf = dom_ag[(len(dom_ag) - n_conserved_ep):]
    target_fractions = np.concatenate((variable_ep_ntf, conserved_ep_ntf))
    #target_fractions = target_fractions[::-1] <- THIS IS WHAT I RAN
    #parameters to update
    params_update = {"n_conserved_epitopes" : n_conserved_ep,
                        "n_variable_epitopes" : n_variable_ep,
                        "E1hs" : E1h, 
                        "naive_target_fractions" : target_fractions,
                        "n_naive_precursors" : naive_precursors}
    #make a directory and a yaml file with these parameters substituted
    path = SWEEP_DIR/f"{i}_epitope_sweep"
    path.mkdir(exist_ok=True)
    with open(SWEEP_DIR/sweep_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data.update(params_update)
    utils.write_yaml(yaml_data, path/f"{i}_epitope_sweep.yaml")
    #submit job
    result = subprocess.run(['python', "scripts/submit_job.py", f"{i}_epitope_sweep"], text=True)



