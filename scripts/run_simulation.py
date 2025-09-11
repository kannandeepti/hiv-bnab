"""
Script to run simulations for a SLURM array job

This script is called by the submit_job.py script to run simulations for a given sweep

    python run_simulation.py <task_id> <num_tasks> <sweep_name>

<task_id> is the task ID of the SLURM array job
<num_tasks> is the total number of tasks in the SLURM array job
<sweep_name> is the name of the sweep directory

This script will run the simulations for the given task ID. It will also clean up log files
for jobs that completed successfully without errors.
"""

import numpy as np
import importlib.util
from pathlib import Path
from itertools import product
import os
import sys
import time

sys.path.append(os.getcwd())
import hiv_code
from hiv_code import utils
from hiv_code.simulation import Simulation
from scripts import SWEEP_DIR, LOG_DIR, clean_up_log_files


def enumerate_completed_tasks(sweep_name):
    """Determine which simulations have already been run in this sweep."""
    sweep_dir = SWEEP_DIR / sweep_name
    sweeps_ran = {}  # dictionary mapping param_dir to list of seed replicates ran
    for param_dir in sweep_dir.iterdir():
        if param_dir.is_dir():
            replicates_ran = []
            for exp_dir in param_dir.iterdir():
                if exp_dir.is_dir() and (exp_dir / "history.pkl").is_file():
                    params = utils.read_json(exp_dir / "parameters.json")
                    replicates_ran.append(params["seed"])
            sweeps_ran[param_dir] = replicates_ran
    return sweeps_ran


def prune_completed_tasks(sweep_name, params_per_task):
    """If job gets interrupted or cancelled, determine which simulations
    still need to be run. First, generate a list of all simulations that
    have already been run within this sweep. Then remove any simulations
    that have already been run from params_per_task"""
    sweeps_ran = enumerate_completed_tasks(sweep_name)
    for input_file in params_per_task[:]:
        if input_file.parent in sweeps_ran.keys():
            replicates_ran = sweeps_ran[input_file.parent]
            seed_num = int(str(input_file.stem).split("_")[-1])
            if seed_num in replicates_ran:
                # remove this input file from params_per_task
                params_per_task.remove(input_file)
    return params_per_task


def batch_tasks(sweep_name):
    """
    Looks inside sweep_dir for input files that end in *.json ->
    number of files corresponds to number of unique parameter sets.
    Divides total number of sweeps by number of jobs, such that each job
    runs some sweeps in serial.
    """
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    sweep_dir = SWEEP_DIR / sweep_name

    # count parameters to sweep from input files in directory
    # TODO : search recursively through subdirectories for json files
    input_param_files = list(sweep_dir.glob("*/*.json"))
    # batch to process with this task
    params_per_task = input_param_files[my_task_id : len(input_param_files) : num_tasks]
    print(params_per_task)
    params_per_task = prune_completed_tasks(sweep_name, params_per_task)
    print(params_per_task)
    tic = time.time()
    sims_ran = 0
    for input_file in params_per_task:
        # parallel run idx is a number that is appended to the end of the experiment directory  name
        # choose this number to match the `seed` paramter (which replicate)
        seed_num = str(input_file.stem).split("_")[-1]
        sim = Simulation(input_file, parallel_run_idx=int(seed_num))
        sim.run()
        sims_ran += 1

    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s")


if __name__ == "__main__":
    sweep_name = str(sys.argv[3])
    batch_tasks(sweep_name)
    clean_up_log_files(sweep_name)
