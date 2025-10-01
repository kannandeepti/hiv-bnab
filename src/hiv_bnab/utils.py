import copy
import functools
import json
import yaml
import pickle
import time
from typing import Callable, Any, Type
from pathlib import PosixPath

import numpy as np
import scipy


def total_time(cls: Type) -> float:
    """Return _total_time of the class."""
    return getattr(cls, "_total_time", 0)


def reset_total_time(cls: Type) -> None:
    """Reset _total_time of the class to 0."""
    cls._total_time = 0


def timing_decorator(method: Callable[..., Any]):
    """Decorator for getting total time spent in a function."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self.__class__, "_total_time"):
            self.__class__._total_time = 0  # Initialize total time at the class level
        start_time = time.perf_counter()  # Record the start time
        result = method(self, *args, **kwargs)  # Call the original method
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        self.__class__._total_time += (
            elapsed_time  # Update the total time at the class level
        )
        return result

    return wrapper


def compress(dictionary: dict, nonzero_threshold: float = 0.5) -> dict:
    """Turn arrays in a dictionary into sparse matrices.

    Only turns array into sparse matrix if the fraction of
    nonzero elements is less than 0.5. Scipy cannot turn
    3D matrices into sparse matrices, so they are
    flattened and the shape is stored in the dictionary.

    Args:
        dictionary: Original dictionary of data.

    Returns:
        newdict: Dictionary containing data as sparse matrices.
    """
    newdict = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            if np.nonzero(value)[0].size / value.size < nonzero_threshold:
                newdict[key + "shape"] = value.shape
                newdict[key] = scipy.sparse.csr_matrix(value.flatten())
        elif isinstance(value, dict):
            newdict[key] = compress(value)
    return newdict


def expand(dictionary: dict) -> dict:
    """Expand a compressed dictionary.

    Flattened arrays are reshaped using the stored shape.

    Args:
        dictionary: Dictionary containing data as sparse matrices.

    Returns:
        newdict: Original dictionary of data.
    """
    newdict = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, scipy.sparse.csr_matrix):
            newdict[key] = np.reshape(value.toarray(), dictionary[key + "shape"])
        elif isinstance(value, dict):
            newdict[key] = expand(value)
    return newdict


def convert_numpy_objects(obj):
    """Convert numpy objects to Python-native types so that they
    are JSON serializable
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, PosixPath):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_objects(i) for i in obj]
    else:
        return obj


def write_pickle(data: Any, file: str) -> None:
    with open(file, "wb") as f:
        pickle.dump(data, f)


def read_pickle(file: str) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


def write_json(data: dict, file: str) -> None:
    data_converted = convert_numpy_objects(data)
    with open(file, "w") as f:
        json.dump(data_converted, f)


def write_yaml(data: dict, file: str) -> None:
    data_converted = convert_numpy_objects(data)
    with open(file, "w") as f:
        return yaml.dump(data_converted, f)


def read_json(file: str) -> dict:
    with open(file, "r") as f:
        return json.load(f)


def read_yaml(file) -> dict:
    with open(file, "r") as f:
        return yaml.safe_load(f)


def get_sample(arr1: np.ndarray, p: float) -> np.ndarray:
    """Get a sample from arr1 with fraction/prob p.

    Args:
        arr1: Array to sample from.
        p: Probability of selecting element in arr1.

    Returns:
        output_idx: Samples from arr1.
    """
    output_idx = np.random.choice(
        arr1, size=np.random.binomial(len(arr1), p=p), replace=False
    )
    return output_idx


def get_other_idx(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Get values in arr1 that are not in arr2.

    Args:
        arr1: Array to search over.
        arr2: Array with elements that should not be in arr1.

    Returns:
        Array of elements in arr1 and not in arr2.
    """
    return arr1[~np.isin(arr1, arr2)]


def get_death_rate_from_half_life(half_life: float, dt: float) -> float:
    """Given a bcell half-life, calculate the corresponding death rate.

    Args:
        half_life: Half-life of the bcell population.
        dt: Timestep.

    Returns:
        Death rate of the bcell population.
    """
    return 1 / dt * (1 - (2 ** (-dt / half_life)))


def fsolve_mult(f: Callable[..., Any], guess: float = 1.1) -> float:
    """
    Scipy fsolve doesn't always work. Try fsolve with many
    different initial guesses.

    Args:
        f: Function to solve.
        guess: Initial guess.

    Returns:
        r: Solution.
    """
    max_tries = 2000
    r = scipy.optimize.fsolve(f, guess)
    n_tries = 0
    while f(r) > 0.05:
        guess += 0.2
        r = scipy.optimize.fsolve(f, guess)
        n_tries += 1
        if guess > 10:
            guess = -10
        if n_tries > max_tries:
            raise ValueError(f"fsolve could not solve in {max_tries} attempts.")
    return r
