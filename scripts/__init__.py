from pathlib import Path
import re
import subprocess

SWEEP_DIR = Path('/home/gridsan/dkannan/git-remotes/hiv-bnab/seeding_10_death_0.4_egc_ntfh_2000')
PLOT_DIR = Path('/home/gridsan/dkannan/git-remotes/hiv-bnab/plots')
LOG_DIR = Path('/home/gridsan/dkannan/git-remotes/hiv-bnab/log_files')
INPUT_SUFFIX = ".yaml" 

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