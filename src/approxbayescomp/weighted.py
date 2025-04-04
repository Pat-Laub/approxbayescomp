# -*- coding: utf-8 -*-
"""
Library to compute weighted quantiles, including the weighted median, of
numpy arrays.

@author: Patrick Laub and Pierre-O Goffard

Original version from:
    https://github.com/nudomarinero/wquantiles/blob/master/weighted.py
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from numba import float64, int64, njit  # type: ignore
from numpy.random import default_rng
from scipy.stats import gaussian_kde  # type: ignore


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quantile > 1.0) or (quantile < 0.0):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def quantile(data, weights, quantile):
    """
    Weighted quantile of an array with respect to the last axis.

    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last
        axis of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile : float
        The output value.
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    if nd == 0:
        TypeError("data must have at least one dimension")
    elif nd == 1:
        return quantile_1D(data, weights, quantile)
    elif nd > 1:
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1D, -1, imr, weights, quantile)
        return result.reshape(n[:-1])


def median(data, weights):
    """
    Weighted median of an array with respect to the last axis.

    Alias for `quantile(data, weights, 0.5)`.
    """
    return quantile(data, weights, 0.5)


def iqr(data, weights):
    return quantile(data, weights, 0.75) - quantile(data, weights, 0.25)


# Extracted from filterpy library
# https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
# NOTE: It crashes if weights doesn't add to one.
# @njit(int64[:](float64[:]), nogil=True)
@njit(nogil=True)
def systematic_resample(weights):
    """Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (rnd.random() + np.arange(N)) / N

    indexes = np.zeros(N, np.int64)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def resample(rng, weights, repeats=10):
    weights = np.asarray(weights)
    allIndices = []
    for rep in range(repeats):
        allIndices.append(systematic_resample(weights))

    indices = np.concatenate(allIndices)
    return indices


################################################################
# function to resample the ABC posterior and fit a gaussian kde.
# ###############################################################
def resample_and_kde(data, weights, cut=3, clip=(-np.inf, np.inf), seed=1, repeats=10):
    # Resample the data
    rng = default_rng(seed)
    weights = np.asarray(weights)
    dataResampled = data[resample(rng, weights, repeats=repeats)]

    # Choose support for KDE
    neff = 1 / sum(weights**2)
    scott = neff ** (-1.0 / 5)
    cov = np.cov(data, bias=False, aweights=weights)
    bw = scott * np.sqrt(cov)

    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])

    K = gaussian_kde(data.T, weights=weights)
    xs = np.linspace(support_min, support_max, 200)
    ys = K(xs)

    return dataResampled, xs, ys
