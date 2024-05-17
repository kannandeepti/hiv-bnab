import json
import pickle
from enum import Enum
from typing import Callable, Any

import numpy as np
import scipy



class DerivedCells(Enum):
    """Values used to tag bcells as where they are derived from."""
    UNSET = 0
    GC = 1
    EGC = 2


def write_pickle(data: Any, file: str) -> None:
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(file: str) -> Any:
    with open(file, 'rb') as f:
        return pickle.load(f)


def write_json(data: dict, file: str) -> None:
    with open(file, 'w') as f:
        json.dump(data, f)
    

def read_json(file: str) -> dict:
    with open(file, 'r') as f:
        return json.load(f)


def get_sample(arr1: np.ndarray, p: float) -> np.ndarray:
    """Get a sample from arr1 with fraction/prob p."""
    output_idx = np.random.choice(
        arr1, 
        size=np.random.binomial(len(arr1), p=p), 
        replace=False
    )
    return output_idx


def get_other_idx(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Get values in arr1 that are not in arr2."""
    return np.array([i for i in arr1 if i not in arr2], dtype=int)


def get_death_rate_from_half_life(half_life: float, dt: float) -> float:
    """Given a bcell half-life, calculate the corresponding death rate."""
    return 1 / dt * (1 - (2 ** (-dt / half_life)))


def fsolve_mult(f: Callable[..., Any], guess: float=1.1):
    """
    Scipy fsolve doesn't always work. Try fsolve with many
    different initial guesses.
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
            raise ValueError(f'fsolve could not solve in {max_tries} attempts.')
    return r