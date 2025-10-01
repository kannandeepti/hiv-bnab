"""
Script to submit an array job on SLURM reading from a parameter sweep instruction yaml file.
Also submits an analysis job that runs once the simulations in the sweep have completed.

To run this script, use the following command:

    python scripts/submit_job.py --config_file <config_file> --sweep_name <sweep_name>

Replace `<sweep_name>` with the name of the sweep directory, i.e. 3_epitope_sweep_C_C0
and ensure that <sweep_name> exists in the `sweep_dir` specified in the scripts config file.
This script will process the parameter sweep instructions in `sweep_dir`/<sweep_name>/<sweep_name>.yaml and create
subdirectories `sweep_dir`/<sweep_name>/sweep_<i> where i is the index of each unique
parameter set. Within each sweep_<i> directory, the script will write a json file for each
simulation replicate sweep_<i>_<j>.json where j is the replicate index.

The script will then submit an array job to run simulations using those json files as
input. Once the simulations have completed, the script will submit a single job to run
the analysis.

The script will also clean up log files for jobs that completed successfully without errors.
"""

import numpy as np
import collections.abc
from pathlib import Path
from itertools import product
import subprocess
import sys
import os
from tap import tapify

from hiv_bnab import utils
from scripts import INPUT_SUFFIX, load_config, clean_up_log_files


def get_values_from_sweep_config(param_sweep_config: dict):
    """Get the values to sweep from the sweep config specifications.

    allowed keys are 'list' -> directly specify values to sweep
                     'min' -> minimum value (inclusive)
                     'max' -> maximum value (inclusive)
                     'num' -> how many values to sweep, including min & max
                     'log' -> whether to space the values on a log scale

    Args:
        param_sweep_config (dict) : dictionary containing sweep specifications

    Returns:
        list of values to sweep
    """
    if "list" in param_sweep_config:
        return np.array(param_sweep_config["list"])
    else:
        if "min" not in param_sweep_config or "max" not in param_sweep_config:
            raise KeyError("Both min and max must be specified for a parameter sweep")
        min_value = param_sweep_config["min"]
        max_value = param_sweep_config["max"]
        if param_sweep_config.get("num", None) is None:
            # assume step size of 1 (integer spacing)
            return np.arange(min_value, max_value + 1)
        else:
            if not param_sweep_config.get("log", False):
                return np.linspace(min_value, max_value, param_sweep_config["num"])
            else:
                return np.logspace(min_value, max_value, param_sweep_config["num"])


def process_sweep_file(sweep_file: Path | str):
    """Parse file that specifies parameters to sweep.

    Args:
        sweep_file (str or Path) : full path to sweep file

    Returns:
        run_config : dict
            configuration for slurm job
        keys : list of str
            parameter names to sweep
        values : list of lists
            parameter values to sweep
    """
    config = utils.read_yaml(Path(sweep_file))
    # parameters that are not swept, but still specified in sweep yaml for clarity to hold fixed
    sweep_param_keys = []
    sweep_param_values = []
    run_config = {}
    num_sim_repeats = 1

    for param_name in config:
        if param_name == "run_params":
            run_config = config[param_name]
        elif param_name == "num_sim_repeats":
            # number of simulation replicates -> change random seed for each one
            # random_seeds = np.arange(0, int(config[param_name]))
            # sweep_param_keys.append('seed')
            # sweep_param_values.append(random_seeds)
            num_sim_repeats = config[param_name]

        elif not isinstance(config[param_name], collections.abc.Mapping):
            # parameter that has a single value
            sweep_param_keys.append(param_name)
            sweep_param_values.append([config[param_name]])

        else:
            param_config = config[param_name]
            sweep_param_keys.append(param_name)
            # if the parameter does not have 'sweep', it's an array of values which each could be swept (i.e. E1hs)
            if "sweep" not in param_config:
                values_to_sweep = []
                for key in param_config:
                    if not isinstance(param_config[key], collections.abc.Mapping):
                        # no sweep
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
            # submit 1 job per parameter set OR max jobs, whichever is smaller
            run_config["num_jobs"] = min(num_sims, run_config["max_jobs"])
        else:
            run_config["num_jobs"] = num_sims

    return run_config, sweep_param_keys, sweep_param_values, num_sim_repeats


def write_submission_file(slurm_file: Path | str, run_config: dict, log_files_path: str):
    """Write a slurm submission script to `slurm_file` using specifications in run_config.

    Args:
        slurm_file (str or Path) : full path to slurm submission script
        run_config (dict or Collection) : dictionary for slurm specifications
            required keys are 'partition', 'cpus_per_task', 'num_jobs', 'sweep_name'

    TODO : make this more general to include more slurm specifications.
    Note: replace the path in export PATH with the path to the conda environment in the user's home directory.
    """
    if (
        "cpus_per_task" not in run_config
        or "num_jobs" not in run_config
        or "partition" not in run_config
        or "sweep_name" not in run_config
        or "conda_env" not in run_config
    ):
        raise ValueError("Required keys are 'partition', 'cpus_per_task', 'num_jobs', 'sweep_name'")

    with open(slurm_file, "w+") as f:
        f.write("#!/bin/bash \n")
        f.write(f'#SBATCH -p {run_config["partition"]} \n')
        f.write(f'#SBATCH -c {run_config["cpus_per_task"]} \n')
        f.write(f'#SBATCH -o log_files/{run_config["sweep_name"]}.log-%A-%a \n')
        f.write(f'#SBATCH --array=0-{int(run_config["num_jobs"])} \n')

        if "subsmission_script_prepend" in run_config:
            for line in run_config["subsmission_script_prepend"]:
                f.write(line)

        f.write(f'echo "My task ID: " $SLURM_ARRAY_TASK_ID \n')
        f.write(f'echo "Number of tasks: " $SLURM_ARRAY_TASK_COUNT \n')

        f.write("cd $SLURM_SUBMIT_DIR \n")
        f.write(f"source activate {run_config['conda_env']} \n")
        if "path_to_conda_env" in run_config:
            f.write(f"export PATH={run_config['path_to_conda_env']}/bin:$PATH \n")
        f.write(
            f'python scripts/run_simulation.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT {run_config["sweep_dir"]}/{run_config["sweep_name"]} {log_files_path}\n'
        )
        f.write(f"conda deactivate \n")


def write_analysis_file(slurm_file: Path | str, run_config: dict, config_file: str):
    """Write a slurm submission script to `slurm_file` using specifications in run_config.

    Args:
        slurm_file (str or Path) : full path to slurm submission script
        run_config (dict or Collection) : dictionary for slurm specifications
            required keys are 'partition', 'cpus_per_analysis', 'sweep_name'

    TODO : make this more general to include more slurm specifications.
    Note: replace the path in export PATH with the path to the conda environment in the user's home directory.

    """
    if (
        "partition" not in run_config
        or "cpus_per_analysis" not in run_config
        or "sweep_name" not in run_config
        or "conda_env" not in run_config
    ):
        raise ValueError("Required keys are 'partition', 'cpus_per_analysis', 'sweep_name', 'conda_env'")

    if "ep_per_ag" not in run_config:
        run_config["ep_per_ag"] = 3

    with open(slurm_file, "w+") as f:
        f.write("#!/bin/bash \n")
        f.write(f'#SBATCH -p {run_config["partition"]} \n')
        f.write(f'#SBATCH -c {run_config["cpus_per_analysis"]} \n')
        f.write(f'#SBATCH -o log_files/{run_config["sweep_name"]}.analysis.log-%A-%a \n')

        if "subsmission_script_prepend" in run_config:
            for line in run_config["subsmission_script_prepend"]:
                f.write(line)
        f.write("cd $SLURM_SUBMIT_DIR \n")
        f.write(f"source activate {run_config['conda_env']} \n")
        if "path_to_conda_env" in run_config:
            f.write(f"export PATH={run_config['path_to_conda_env']}/bin:$PATH \n")
        if "resort_directories" in run_config:
            f.write(
                f'python scripts/analyze_sweep.py --config_file {config_file} --sweep_name {run_config["sweep_name"]} --ep_per_ag {run_config["ep_per_ag"]} --resort_directories {run_config["resort_directories"]}\n'
            )
        else:
            f.write(
                f'python scripts/analyze_sweep.py --config_file {config_file} --sweep_name {run_config["sweep_name"]} --ep_per_ag {run_config["ep_per_ag"]}\n'
            )
        f.write(f"conda deactivate \n")


def write_input_json_files(sweep_dir, sweep_param_keys, sweep_param_values, num_sim_repeats):
    """Generate a combinatorial sweep and write 1 input file per parameter set.

    Args:
        sweep_dir (str or Path) : full path to directory containing input files
        sweep_param_keys (list of str) : list of keys in parameter dict that will be written
        sweep_param_values (list of lists) : one list per key containing values to sweep

    """
    # parameters to sweep
    params_to_sweep = list(product(*sweep_param_values))
    for i, param_set in enumerate(params_to_sweep):
        param_dir = sweep_dir / f"sweep_{i}"
        param_dir.mkdir(exist_ok=True)  # will overwrite this directory if it exists
        param_dict = dict(zip(sweep_param_keys, param_set))
        param_dict["experiment_dir"] = str(param_dir)
        for j in range(num_sim_repeats):
            param_dict["seed"] = j
            # write parameters that were changed beyond the defaults to a specific json
            utils.write_json(param_dict, param_dir / f"sweep_{i}_{j}.json")


def submit_array_job(config_file: str, sweep_name: str):
    config = load_config(config_file)
    SWEEP_DIR = Path(config["sweep_dir"])
    sweep_file = sweep_name + INPUT_SUFFIX
    sweep_dir = SWEEP_DIR / sweep_name

    try:
        run_config, sweep_param_keys, sweep_param_values, num_sim_repeats = process_sweep_file(sweep_dir / sweep_file)
    except FileExistsError as e:
        print(f"Sweep file in {str(sweep_dir/sweep_file)} does not exist.")
        raise e
    # read slurm config yaml file and merge with run_config
    slurm_config = load_config(config_file)
    run_config = {**slurm_config, **run_config}
    write_input_json_files(sweep_dir, sweep_param_keys, sweep_param_values, num_sim_repeats)
    write_submission_file(sweep_dir / "run_sweep.sbatch", run_config, config["log_dir"])  # array job
    write_analysis_file(sweep_dir / "run_analysis.sbatch", run_config, config_file)  # single job
    result = subprocess.run(["sbatch", str(sweep_dir / "run_sweep.sbatch")], capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout
        jobid = output.split()[-1]
        subprocess.run(
            [
                "sbatch",
                f"--depend=afterany:{jobid}",
                str(sweep_dir / "run_analysis.sbatch"),
            ]
        )
    # clean up log files here:
    clean_up_log_files(config["log_dir"], sweep_name + ".analysis")


if __name__ == "__main__":
    tapify(submit_array_job)
