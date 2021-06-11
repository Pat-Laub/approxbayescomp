import numpy as np
from numba import njit
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from hilbertcurve.hilbertcurve import HilbertCurve

@njit(nogil=True)
def wass_sumstats(x):
    return np.sort(x)


@njit(nogil=True)
def wass_dist(sortedData, sortedFake, p=1.0):
    n = len(sortedData)
    return np.linalg.norm(sortedData - sortedFake, p) / n


@njit(nogil=True)
def identity(x):
    return x 


def wrap_ss_curve_matching_approx(scale_x, scale_t):
    def ss_curve_matching_approx(xData):
        ssData = (
            (xData * scale_x).astype(np.int64),
            (np.arange(1, len(xData) + 1, 1) * scale_t),
        )
        return np.vstack(ssData).T

    return ss_curve_matching_approx


def wass_2Ddist_ss_approx(xData):
    ssData = xData.astype(np.int64)
    return ssData


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
    sqrtSumSqrs = np.sqrt(np.sum(diff ** 2, axis=1))
    dist = np.mean(sqrtSumSqrs)

    return dist


def wrap_ss_curve_matching(gamma):
    def ss_curve_matching(xData):
        t = np.arange(len(xData))
        Y = np.vstack([gamma * t, xData]).T
        return Y

    return ss_curve_matching


def wass_2Ddist_ss(xData):
    return xData


def wass_2Ddist(ssData, ssFake):
    """
    Compute the Wasserstein using the Hungarian algorithm.

    Parameters
    ----------
    ssData, ssFake : n*2 arrays, where n is the sample size


    Returns
    -------
    scalar
    The Wasserstein distance between the 2D samples
    """
    n = ssData.shape[0]
    d = cdist(ssData, ssFake)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / n
