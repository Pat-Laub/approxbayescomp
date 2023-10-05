import numpy as np


def dtw_distance(s1, s2, penalty=None, only_ub=False):
    # If only the upper bound is requested, compute and return the Euclidean distance
    if only_ub:
        return np.linalg.norm(np.array(s1) - np.array(s2))

    n = len(s1)
    m = len(s2)
    dtw = np.zeros((n + 1, m + 1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0

    # The penalty is added when there's an insertion or deletion
    if penalty is None:
        penalty = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j] + penalty, dtw[i, j - 1] + penalty, dtw[i - 1, j - 1])

    return np.sqrt(dtw[n, m])
