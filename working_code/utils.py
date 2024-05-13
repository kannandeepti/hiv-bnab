import json
from enum import Enum
from typing import Callable, Any
import pickle
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
    with open(file, 'r') as f:
        json.dump(data, f)
    

def read_json(file: str) -> dict:
    with open(file, 'r') as f:
        return json.load(f)


def get_sample(arr1: np.ndarray, p: float) -> np.ndarray:
    output_idx = np.random.choice(
        arr1, 
        size=np.random.binomial(len(arr1), p=p), 
        replace=False
    )
    return output_idx


def get_other_idx(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.array([i for i in arr1 if i not in arr2])


def get_death_rate_from_half_life(half_life: float, dt: float) -> float:
    return 1 / dt * (1 - (2 ** (-dt / half_life)))


def any(x):
    """
    Equivalent of any function in Matlab.
    """
    if (isinstance(x, int) or isinstance(x, float)) and (x != 0):
        return 1
    elif (isinstance(x, int) or isinstance(x, float)) and (x == 0):
        return 0
    if len(x.shape) > 1:
        first_ind = np.where(np.array(x.shape) > 1)[0][0]
        y = np.zeros(shape = x.shape[first_ind + 1:])
        inds = np.nonzero(x)[first_ind + 1:]
        if len(inds) > 0:
            y[np.nonzero(x)[first_ind + 1:]] = 1
        return y
    else:
        if len(np.nonzero(x)[0]) > 0:
            return 1
        else:
            return 0


def reshape_(x: np.adarray, row: bool=False) -> np.ndarray:
    """
    If vector, then reshape to a 1D column matrix.
    """
    lshape = len(x.shape)
    if lshape == 0:
        raise ValueError('Empty vector')
    elif lshape == 1 and row is False:
        return np.reshape(x, (-1, 1))
    elif lshape == 1 and row is True:
        return np.reshape(x, (1, -1))
    return x


def matlab_percentile(in_data: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """
    Calculate percentiles in the way IDL and Matlab do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    in_data: numpy.ndarray
        input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values

    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """

    data = np.sort(in_data)
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def percentile(arr: np.ndarray, percentile: float, axis: int=1) -> np.ndarray:
    """
    Do matlab_percentile for a matrix.
    """
    if len(arr.shape) == 2:
        if axis == 0:
            return np.array([matlab_percentile(row, percentile) for row in arr])
        elif axis == 1:
            return np.array([matlab_percentile(row, percentile) for row in arr.T])
        else:
            raise ValueError('prctile does not do matrices more than 2D')
    elif len(arr.shape) == 1 and arr.shape[0] != 0:
        return matlab_percentile(arr, percentile)
    elif len(arr.shape) == 1 and arr.shape[0] == 0:
        return np.array([np.nan])



def fsolve_mult(f: Callable[..., Any], guess: float=1.1):
    """
    Scipy fsolve doesn't always work. Try fsolve with many
    different initial guesses.
    """
    max_tries = 2000
    r = scipy.optimize.fsolve(f, guess)
    num_tries = 0
    while f(r) > 0.05:
        guess += 0.2
        r = scipy.optimize.fsolve(f, guess)
        num_tries += 1
        if guess > 10:
            guess = -10
        if num_tries > max_tries:
            raise ValueError(f'fsolve could not solve in {max_tries} attempts.')
    return r


def cellsNumAff(cellsarr, M, param):
    """
    Obtain the summary of number and affinities of B cells
    Outpus:
      numbyaff: 1x3x4 array; Dim1,2,3 - GC, Epitope, 
                # of B cells with affinities greater than 6, 7, 8, 9
      affprct: 1x3x4 aray; Dim1,2,3 - GC, Epitope,
                100, 90, 75, 50 percentiles affinities
    Inputs:
      cellsarr: 2D array of B cells, each column representing a B cell.
                Can be GC, memory, or plasma cells
      M: Number of GC/EGC
      param: parameter struct
    """
    
    thresholds = np.array([6, 7, 8, 9])
    percentile = np.array([100, 90, 75, 50])
    aff = cellsarr[M * 2 + 0: M * 3]
    aff_reshape = reshape_(aff, row = True)
    target = cellsarr[M * 1 + 0: M * 2]
    target_reshape = reshape_(target, row = True)
    
    numbyaff = np.zeros(shape = (M, param['n_ep'], 4))
    affprct = np.zeros(shape = (M, param['n_ep'], 4))
    
    for i in range(4):
        for ep in range(param['n_ep']):
            temp = reshape_((aff * (target == ep + 1)) > thresholds[i], row = True)
            numbyaff[:, ep, i] = np.sum(temp, axis=1)
            for k in range(M):
                temp = percentile((aff_reshape[k, target_reshape[k, :] == ep + 1]).T,
                               percentile[i]).T
                affprct[k, ep, i] = temp
                
    return np.squeeze(numbyaff), np.squeeze(affprct)