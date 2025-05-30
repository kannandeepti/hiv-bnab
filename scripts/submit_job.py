"""
Script to submit an array job on SLURM reading from a parameter sweep instruction yaml file.
Also submits an analysis job that runs once the sweep has completed.

TODO: create a default run_config file with the slurm defaults 
TODO: create a default parameters file with ALL the base parameters
"""

import numpy as np
import collections.abc
from pathlib import Path
from itertools import product
import subprocess
import sys
import os
sys.path.append(os.getcwd())

from hiv_code import utils
from scripts import SWEEP_DIR, INPUT_SUFFIX, clean_up_log_files

def get_values_from_sweep_config(param_sweep_config):
    """ Get the values to sweep from the sweep config specifications.
    
    allowed keys are 'list' -> directly specify values to sweept
                     'min' -> minimum value (inclusive)
                     'max' -> maximum value (inclusive)
                     'num' -> how many values to sweep, including min & max
                     'log' -> whether to space the values on a log scale

    TODO : type checking? or will parameters data class take care of this??
    
    """
    if 'list' in param_sweep_config:
        return np.array(param_sweep_config['list'])
    else:
        if 'min' not in param_sweep_config or 'max' not in param_sweep_config:
            raise KeyError('Both min and max must be specified for a parameter sweep')   
        min_value = param_sweep_config['min']
        max_value = param_sweep_config['max']
        if param_sweep_config.get('num', None) is None:
            #assume step size of 1 (integer spacing)
            return np.arange(min_value, max_value + 1)
        else:
            if not param_sweep_config.get('log', False):
                return np.linspace(min_value, max_value, param_sweep_config['num'])
            else:
                return np.logspace(min_value, max_value, param_sweep_config['num'])

def process_sweep_file(sweep_file):
    """ Parse file that specifies parameters to sweep.
    
    Returns:
        run_config : dict
            configuration for slurm job
        keys : list of str
            parameter names to sweep
        values : list of lists
            parameter values to sweep
    """
    config = utils.read_yaml(Path(sweep_file))
    #parameters that are not swept, but still specified in sweep yaml for clarity to hold fixed
    sweep_param_keys = []
    sweep_param_values = []
    run_config = {}
    num_sim_repeats = 1
    
    for param_name in config:
        if param_name == "run_params":
            run_config = config[param_name]
        elif param_name == "num_sim_repeats":
            #number of simulation replicates -> change random seed for each one
            #random_seeds = np.arange(0, int(config[param_name]))
            #sweep_param_keys.append('seed')
            #sweep_param_values.append(random_seeds)
            num_sim_repeats = config[param_name]
        
        elif not isinstance(config[param_name], collections.abc.Mapping):
            #parameter that has a single value
            sweep_param_keys.append(param_name)
            sweep_param_values.append([config[param_name]])
        
        else:
            param_config = config[param_name]
            sweep_param_keys.append(param_name)
            # if the parameter does not have 'sweep', it's an array of values which each could be swept (i.e. E1hs)
            if 'sweep' not in param_config:
                values_to_sweep = []
                for key in param_config:
                    if not isinstance(param_config[key], collections.abc.Mapping):
                        #no sweep
                        # TODO: make this recursive in case there are parameters that are more than 1D arrays
                        # so far not relevant, but could be if matrices become inputs to Parameters
                        values_to_sweep.append([param_config[key]])
                    else:
                        values_to_sweep.append(get_values_from_sweep_config(param_config[key]))
                sweep_param_values.append(list(product(*values_to_sweep)))   
            # the parameter is a value that is swept
            else:
                sweep_param_values.append(get_values_from_sweep_config(param_config))
    
    run_config["sweep_name"] = sweep_file.stem
    if "num_jobs" not in run_config:
        params_to_sweep = list(product(*sweep_param_values))
        num_sims = len(params_to_sweep) * num_sim_repeats
        if "ndata" in run_config:
            run_config["num_jobs"] = int(np.ceil(num_sims / run_config["ndata"]))
        elif "max_jobs" in run_config:
            #submit 1 job per parameter set OR max jobs, whichever is smaller
            run_config["num_jobs"] = min(num_sims, run_config["max_jobs"])
        else:
            run_config["num_jobs"] = num_sims
    
    return run_config, sweep_param_keys, sweep_param_values, num_sim_repeats

def write_submission_file(slurm_file, run_config):
    """ Write a slurm submission script to `slurm_file` using specifications in run_config.
    
    Args:
        slurm_file (str or Path) : full path to slurm submission script
        run_config (dict or Collection) : dictionary for slurm specifications
            allowed keys are 'partition', 'cpus_per_task', 'num_jobs', 'sweep_name'
    
    TODO : make this more general to include more slurm specifications.
    TODO : LLMapReduce for sweep analysis
    
    """
    with open(slurm_file, 'w+') as f:
        f.write('#!/bin/bash \n')
        f.write(f'#SBATCH -p {run_config["partition"]} \n')
        f.write(f'#SBATCH -c {run_config["cpus_per_task"]} \n')
        f.write(f'#SBATCH -o log_files/{run_config["sweep_name"]}.log-%A-%a \n')
        f.write(f'#SBATCH --array=0-{int(run_config["num_jobs"])} \n')

        f.write(f'source /etc/profile \n')
        f.write(f'module load anaconda/2023b \n')

        f.write(f'echo "My task ID: " $SLURM_ARRAY_TASK_ID \n')
        f.write(f'echo "Number of tasks: " $SLURM_ARRAY_TASK_COUNT \n')

        f.write(f'source activate GCdynamics \n')
        f.write(f'export PATH=/home/gridsan/dkannan/.conda/envs/GCdynamics/bin:$PATH \n')
        f.write(f'echo $PATH \n')
        f.write(f'python scripts/run_simulation.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT {run_config["sweep_name"]} \n')
        f.write(f'conda deactivate \n')

def write_analysis_file(slurm_file, run_config):
    """ Write a slurm submission script to `slurm_file` using specifications in run_config.
    
    Args:
        slurm_file (str or Path) : full path to slurm submission script
        run_config (dict or Collection) : dictionary for slurm specifications
            allowed keys are 'partition', 'cpus_per_task', 'num_jobs', 'sweep_name'
    
    TODO : make this more general to include more slurm specifications.
    TODO : LLMapReduce for sweep analysis
    
    """
    if "ep_per_ag" not in run_config:
        run_config["ep_per_ag"] = 3

    with open(slurm_file, 'w+') as f:
        f.write('#!/bin/bash \n')
        f.write(f'#SBATCH -p {run_config["partition"]} \n')
        f.write(f'#SBATCH -c {run_config["cpus_per_analysis"]} \n')
        f.write(f'#SBATCH -o log_files/{run_config["sweep_name"]}.analysis.log-%A-%a \n')

        f.write(f'source /etc/profile \n')
        f.write(f'module load anaconda/2023b \n')

        f.write(f'source activate GCdynamics \n')
        f.write(f'export PATH=/home/gridsan/dkannan/.conda/envs/GCdynamics/bin:$PATH \n')
        f.write(f'echo $PATH \n')
        f.write(f'python scripts/analyze_sweep.py {run_config["sweep_name"]} {run_config["ep_per_ag"]}\n')
        f.write(f'conda deactivate \n')

def write_input_json_files(sweep_dir, sweep_param_keys, sweep_param_values, num_sim_repeats):
    """ Generate a combinatorial sweep and write 1 input file per parameter set.
    
    Args:
        sweep_dir (str or Path) : full path to directory containing input files
        sweep_param_keys (list of str) : list of keys in parameter dict that will be written
        sweep_param_values (list of lists) : one list per key containing values to sweep
    
    """
    # parameters to sweep
    params_to_sweep = list(product(*sweep_param_values))
    for i, param_set in enumerate(params_to_sweep):
        param_dir = sweep_dir/f"sweep_{i}"
        param_dir.mkdir(exist_ok=True) #will overwrite this directory if it exists
        param_dict = dict(zip(sweep_param_keys, param_set))
        param_dict["experiment_dir"] = str(param_dir)
        for j in range(num_sim_repeats):
            param_dict["seed"] = j
            #write parameters that were changed beyond the defaults to a specific json
            utils.write_json(param_dict, param_dir/f'sweep_{i}_{j}.json')

def submit_array_job():
    sweep_name = str(sys.argv[1]) 
    sweep_file = sweep_name + INPUT_SUFFIX
    sweep_dir = SWEEP_DIR/sweep_name

    try:
        run_config, sweep_param_keys, sweep_param_values, num_sim_repeats = process_sweep_file(sweep_dir/sweep_file)
    except FileExistsError as e:
        print(f"Sweep file in {str(sweep_dir/sweep_file)} does not exist.")
        raise e

    write_input_json_files(sweep_dir, sweep_param_keys, sweep_param_values, num_sim_repeats)
    write_submission_file(sweep_dir/"run_sweep.sbatch", run_config) #array job
    write_analysis_file(sweep_dir/"run_analysis.sbatch", run_config) #single job
    result = subprocess.run(['sbatch', str(sweep_dir/"run_sweep.sbatch")], capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout
        jobid = output.split()[-1]
        subprocess.run(['sbatch', f'--depend=afterany:{jobid}', str(sweep_dir/"run_analysis.sbatch")])
    #clean up log files here:
    clean_up_log_files(sweep_name + ".analysis")

if __name__ == "__main__":
    submit_array_job()


            



        
        

