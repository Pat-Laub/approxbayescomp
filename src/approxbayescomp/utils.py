import collections
from typing import Any, Callable, Generator, Iterable, Optional
from numba import float64, int64, njit, void  # type: ignore

import numpy as np
import numpy.random as rnd

from .population import Population
from .weighted import systematic_resample


@njit(void(int64), nogil=True)
def numba_seed(seed):
    rnd.seed(seed)


@njit(float64(float64[:], float64[:], float64[:], float64), nogil=True)
def uniform_pdf(theta, lower, upper, normConst):
    for i in range(len(theta)):
        if theta[i] <= lower[i] or theta[i] >= upper[i]:
            return 0
    return normConst


def index_generator(rg: rnd.Generator, weights: np.ndarray) -> Generator[int, None, None]:
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


def make_iterable(x: Any) -> Iterable[Any]:
    """Ensure that the variable can be looped over.

    Args:
        x: The variable to check.

    Returns:
        The variable if it has a length, otherwise a tuple containing the variable.
    """
    if not isinstance(x, collections.abc.Iterable):
        x = (x,)
    return x


def validate_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=float).squeeze()


def validate_model_prior(modelPrior: Optional[np.ndarray], M: int) -> np.ndarray:
    if not modelPrior:
        modelPrior = np.ones(M) / M
    return modelPrior


def validate_distance(
    sumstats, distance
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], float]]:
    if isinstance(distance, collections.abc.Sequence):
        sumstats = distance[0]
        distance = distance[1]

    if sumstats is None:
        return (lambda x: x, distance)

    return sumstats, distance


def print_header(popSize: int, T: int, numSumStats: int, numProcs: int):
    potentialPlural = "processes" if numProcs > 1 else "process"
    print(
        f"Starting ABC-SMC with population size of {popSize} and sample size "
        + f"of {T} (~> {numSumStats}) on {numProcs} {potentialPlural}."
    )


def print_update(
    t: int, eps: float, elapsed: float, numSims: int, totalSimulationCost: int, fit: Population, nextFit: Population
):
    """
    After each sequential Monte Carlo iteration, print out a summary
    of the just-sampled population, and of the subpopulation which was
    prepared for the next round.
    """
    update = f"Finished SMC iteration {t}, " if t > 0 else "Finished sampling from prior, "
    update += f"eps = {eps:.2f}, "
    elapsedMins = np.round(elapsed / 60, 1)
    update += f"time = {np.round(elapsed)}s / {elapsedMins}m, "
    update += f"popSize = {fit.size()} -> {nextFit.size()}, "
    if fit.M > 1:
        update += f"ESS = {fit.ess()} -> {nextFit.ess()}, "
    else:
        update += f"ESS = {fit.ess()[0]} -> {nextFit.ess()[0]}, "
    update += f"# sims = {numSims}, total # sims = {totalSimulationCost}"
    if fit.M > 1:
        update += f"\n\tmodel populations = {fit.model_sizes()}, "
        update += f"model weights = {np.round(fit.model_weights(), 2)}"
    print(update)
