import numpy as np


def wass_sumstats(x):
    return np.sort(x)


def wass_dist(sortedData, sortedFake, t=0, p=1.0):
    n = len(sortedData)
    return np.linalg.norm(sortedData - sortedFake, p) / n
