from pathlib import Path
import re
import subprocess
import yaml

INPUT_SUFFIX = ".yaml"


def load_config(config_file: Path | str):
    """Load configuration from config.yaml file."""
    config_file = Path(config_file)
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")


def log_file_has_errors(log_file_path: Path | str) -> bool:
    """Check if a log file contains error messages."""
    error_keywords = ["Error", "Exception", "Traceback", "Failed"]
    log_file_path = Path(log_file_path)
    if not log_file_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    with open(log_file_path, "r") as log_file:
        for line in log_file:
            if any(keyword in line for keyword in error_keywords):
                return True
    return False


def clean_up_log_files(log_files_path: str, sweep_name: str):
    """Clean up log files for jobs that completed successfully without errors."""
    job_pattern = re.compile(rf"{sweep_name}\.log-(\d+)-(\d+)")

    # Check log files for errors and delete if no errors found
    for log_file in Path(log_files_path).glob(f"{sweep_name}.log-*"):
        match = job_pattern.match(log_file.name)
        if match:
            job_id, array_id = match.groups()
            result = subprocess.run(
                ["sacct", "--format=JobID,State", "-n", "-P", "-j", job_id], capture_output=True, text=True
            )
            if result.returncode == 0:
                job_id, state = result.stdout.splitlines()[0].split("|")
                if "COMPLETED" in state and not log_file_has_errors(log_file):
                    log_file.unlink()  # Remove the file
