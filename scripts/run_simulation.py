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

SWEEP_DIR = Path('/home/gridsan/dkannan/git-remotes/gc_dynamics/sweeps')

def batch_tasks():
    """
    Looks inside sweep_dir for input files that end in *.json -> 
    number of files corresponds to number of unique parameter sets.
    Divides total number of sweeps by number of jobs, such that each job
    runs some sweeps in serial.
    """

    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    sweep_name = str(sys.argv[3]) 
    sweep_dir = SWEEP_DIR/sweep_name

    # count parameters to sweep from input files in directory
    input_param_files = list(sweep_dir.glob('*.json'))
    # batch to process with this task
    params_per_task = input_param_files[my_task_id: len(input_param_files): num_tasks]
    print(params_per_task)
    tic = time.time()
    sims_ran = 0
    for input_file in params_per_task:
        sim = Simulation(input_file, parallel_run_idx=my_task_id)
        sim.run()
        sims_ran += 1

    toc = time.time()
    nsecs = toc - tic
    nhours = int(np.floor(nsecs // 3600))
    nmins = int((nsecs % 3600) // 60)
    nsecs = int(nsecs % 60)
    print(f"Ran {sims_ran} simulations in {nhours}h {nmins}m {nsecs}s")

if __name__ == "__main__": 
    batch_tasks()