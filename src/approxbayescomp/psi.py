import collections
from numba import float64, int64, njit, void  # type: ignore
from numba.core.types import unicode_type  # type: ignore
import numpy as np

Psi = collections.namedtuple("Psi", ["name", "param"], defaults=["sum", 0.0])


def compute_psi(freqs: np.ndarray, sevs: np.ndarray, psi):
    return _compute_psi(freqs, sevs, psi.name, psi.param)


@njit(float64[:](int64[:], float64[:], unicode_type, float64), nogil=True)
def _compute_psi(freqs, sevs, psi_name, psi_param):
    xs = -np.ones(len(freqs))
    i = 0

    if psi_name == "sum":
        for r, n in enumerate(freqs):
            xs[r] = np.sum(sevs[i : i + n])
            i += n

    elif psi_name == "max":
        for r, n in enumerate(freqs):
            if n > 0:
                xs[r] = np.max(sevs[i : i + n])
            else:
                xs[r] = 0
            i += n

    elif psi_name == "GSL":
        for r, n in enumerate(freqs):
            xs[r] = np.maximum(np.sum(sevs[i : i + n]) - psi_param, 0)
            i += n

    elif psi_name == "ISL":
        for r, n in enumerate(freqs):
            xs[r] = np.sum(np.maximum(sevs[i : i + n] - psi_param, 0))
            i += n

    elif psi_name == "severities":
        return sevs

    else:
        raise Exception("Psi function unsupported")

    return xs
