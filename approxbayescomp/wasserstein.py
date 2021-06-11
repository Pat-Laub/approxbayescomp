import numpy as np
from numba import njit
from hilbertcurve import HilbertCurve


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


def wrap_ss_curve_matching(scale_x, scale_t):
    def ss_curve_matching(xData):
        ssData = (
            (xData * scale_x).astype(np.int64),
            (np.arange(1, len(xData) + 1, 1) * scale_t),
        )
        return np.vstack(ssData).T

    return ss_curve_matching


def wass_2Ddist_ss(xData):
    ssData = xData.astype(np.int64)
    return ssData


def wass_2Ddist(ssData, ssFake):
    """
    Compute the Wasserstein via a projection onto the hilbert space filling curve.

    Parameters
    ----------
    bivariateData, bivariateFaken : n*2 arrays, where n is the sample size


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
