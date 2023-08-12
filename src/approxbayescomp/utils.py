import collections
from numba import float64, int64, njit, void  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore

import numpy as np
import numpy.random as rnd

from .weighted import systematic_resample


@njit(void(int64), nogil=True)
def numba_seed(seed):
    rnd.seed(seed)


def kde(data: np.ndarray, weights: np.ndarray, bw: float = np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)


SimpleKDE = collections.namedtuple("SimpleKDE", ["dataset", "weights", "d", "n", "inv_cov", "L", "log_det"])


@njit(float64(float64[:], float64[:], float64[:], float64), nogil=True)
def uniform_pdf(theta, lower, upper, normConst):
    for i in range(len(theta)):
        if theta[i] <= lower[i] or theta[i] >= upper[i]:
            return 0
    return normConst


@njit(
    # float64(
    #     float64[:], int64, int64, float64[:, :], float64[:], float64[:, :], float64
    # ),
    nogil=True
)
def gaussian_kde_logpdf(x, d, n, dataset, weights, inv_cov, log_det):
    """
    Evaluate the log of the estimated pdf on a provided set of points.
    """
    x = x.flatten()

    energy = np.empty(n, np.float64)

    # loop over data
    for i in range(n):
        diff = (dataset[:, i].flatten() - x).reshape((d, 1))
        tdiff = np.dot(inv_cov, diff)
        energy[i] = np.sum(diff * tdiff)
    log_to_sum = 2.0 * np.log(weights) - log_det - energy
    return np.log(np.sum(np.exp(0.5 * log_to_sum)))


def index_generator(rg, weights):
    N = len(weights)
    inds = range(N)
    uniform = len(set(weights)) == 1

    while True:
        # Generate a sample of length N from weights
        # distribution using systematic resampling.
        if not uniform:
            inds = systematic_resample(weights)

        # As the previous sample is sorted, randomly choose
        # to start somewhere in the middle of the sequence.
        start = rg.choice(N)

        for i in range(N):
            yield inds[(start + i) % N]
