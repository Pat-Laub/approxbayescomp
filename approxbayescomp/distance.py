# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve  # type: ignore
from numba import float64, njit  # type: ignore
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore


@njit(float64(float64[:], float64[:]), nogil=True)
def l1(x, y):
    return np.linalg.norm(x - y, 1)


@njit(float64(float64[:], float64[:]), nogil=True)
def l2(x, y):
    return np.linalg.norm(x - y, 2)


@njit(float64[:](float64[:]), nogil=True)
def sorted(x):
    return np.sort(x)


@njit(float64(float64[:], float64[:]), nogil=True)
def l1_scaled(x, y):
    return np.linalg.norm(x - y, 1) / len(x)


# Computing the 1-dimension Wasserstein distance is
# equivalent to sorting the data, then applying
# a scaled L^1 distance.
wasserstein = (sorted, l1_scaled)


def wasserstein2D(x, y):
    """
    Compute the Wasserstein using the Hungarian algorithm.

    Parameters
    ----------
    x, y : n*2 arrays, where n is the sample size


    Returns
    -------
    scalar
    The Wasserstein distance between the 2D samples
    """
    n = x.shape[0]
    d = cdist(x, y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / n


def wrap_ss_curve_matching(gamma):
    def ss_curve_matching(xData):
        t = np.arange(len(xData))
        Y = np.vstack([gamma * t, xData]).T
        return Y

    return ss_curve_matching


def wass_2Ddist_approx(ssData, ssFake):
    """
    Compute the Wasserstein via a projection onto the hilbert space filling curve.

    Parameters
    ----------
    ssData, ssFake : n*2 arrays, where n is the sample size


    Returns
    -------
    scalar
    The Wasserstein distance between the 2D samples
    """

    maxData = np.max([np.max(ssData), np.max(ssFake)])
    k = int(np.ceil(np.log(maxData + 1) / np.log(2)))
    hilbert_curve = HilbertCurve(k, 2)
    permut = np.argsort(hilbert_curve.distances_from_points(ssData.astype(int)))
    permutFake = np.argsort(hilbert_curve.distances_from_points(ssFake.astype(int)))

    diff = ssData[permut, :] - ssFake[permutFake, :]
    sqrtSumSqrs = np.sqrt(np.sum(diff**2, axis=1))
    dist = np.mean(sqrtSumSqrs)

    return dist
