import numpy as np
import importlib.util
from pathlib import Path
from itertools import product
import os
import sys
import time
import subprocess
import re
sys.path.append(os.getcwd())
import hiv_code
from hiv_code import utils
from hiv_code.simulation import Simulation

SWEEP_DIR = Path('/home/gridsan/dkannan/git-remotes/gc_dynamics/sweeps')
LOG_DIR = Path('/home/gridsan/dkannan/git-remotes/gc_dynamics/log_files')

def log_file_has_errors(log_file_path):
    """ Check if a log file contains error messages. """
    error_keywords = ['Error', 'Exception', 'Traceback', 'Failed']
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if any(keyword in line for keyword in error_keywords):
                return True
    return False

def clean_up_log_files(sweep_name):
    """ Clean up log files for jobs that completed successfully without errors. """
    log_files_path = Path('log_files')
    job_pattern = re.compile(rf"{sweep_name}\.log-(\d+)-(\d+)")

    # Check log files for errors and delete if no errors found
    for log_file in log_files_path.glob(f"{sweep_name}.log-*"):
        match = job_pattern.match(log_file.name)
        if match:
            job_id, array_id = match.groups()
            result = subprocess.run(['sacct', '--format=JobID,State', '-n', '-P', '-j', job_id], capture_output=True, text=True)
            if result.returncode == 0:
                job_id, state = result.stdout.splitlines()[0].split('|')
                if 'COMPLETED' in state and not log_file_has_errors(log_file):
                    log_file.unlink()  # Remove the file
                else:
                    print(f"Errors found in log file: {log_file}, preserving it.")
    
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
    sweep_dir = SWEEP_DIR/sweep_name

    # count parameters to sweep from input files in directory
    # TODO : search recursively through subdirectories for json files
    input_param_files = list(sweep_dir.glob('*/*.json'))
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
    sweep_name = str(sys.argv[3]) 
    batch_tasks(sweep_name)
    clean_up_log_files(sweep_name)