import numpy as np


def wass_adap_sumstats(x):
    return (np.sum(x == 0), np.sort(x))


def wass_adap_dist(ssData, ssFake, t=0, p=1.0):
    numZerosData, sortedData = ssData
    numZerosFake, sortedFake = ssFake
    zerosDiff = np.abs(numZerosData - numZerosFake)
    n = len(sortedData)
    if t < 4:
        return (np.exp(t * zerosDiff) + np.linalg.norm(sortedData - sortedFake, p)) / n
    else:
        if zerosDiff > 0:
            return np.inf

        return np.linalg.norm(sortedData - sortedFake, p) / n
