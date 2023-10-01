import collections
import numpy as np
from scipy.stats import gaussian_kde  # type: ignore
from numba import float64, int64, njit, void  # type: ignore


def kde(data: np.ndarray, weights: np.ndarray, bw: float = np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)


SimpleKDE = collections.namedtuple("SimpleKDE", ["dataset", "weights", "d", "n", "inv_cov", "L", "log_det"])


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
