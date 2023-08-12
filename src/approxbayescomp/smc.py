# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
from __future__ import annotations

import collections
import inspect
import warnings
from time import time
from typing import Callable, Dict, Optional, Tuple, List, Generator, cast

import joblib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.random as rnd
from numba import float64, int64, njit, void  # type: ignore
from numba.core.errors import NumbaPerformanceWarning  # type: ignore
from numpy.random import SeedSequence, default_rng  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from .distance import wasserstein
from .plot import plot_posteriors
from .population import Population
from .prior import Prior
from .simulate import sample_discrete_dist, sample_multivariate_normal
from .utils import numba_seed, index_generator, gaussian_kde_logpdf, make_iterable

# Suppress a numba.PerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Create a type alias for the model function
Model = Callable[[np.ndarray], np.ndarray] | Callable[[rnd.Generator, np.ndarray], np.ndarray]


def _sample_one_first_iteration(
    seed, modelPrior, models: Tuple[Model], priors, sumstats, distance, ssData, simulatorUsesOldNumpyRNG
):
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    # On the first iteration of SMC we sample from the prior
    # and accept everthing, so the code is a bit simpler.
    m = sample_discrete_dist(modelPrior)
    model = models[m]
    prior = priors[m]

    theta = prior.sample(rg)

    if simulatorUsesOldNumpyRNG:
        xFake = model(theta)  # type: ignore
    else:
        xFake = model(rg, theta)  # type: ignore

    if sumstats is not None:
        dist = distance(ssData, sumstats(xFake))
    else:
        dist = distance(ssData, xFake)

    return m, theta, 1.0, dist, 1


def sample_particles(
    seed,
    simulationBudget,
    stopTaskAfterNParticles: Optional[int],
    modelPrior,
    models: Tuple[Model],
    priors: Tuple[Prior],
    kdes,
    sumstats,
    distance,
    eps,
    matchZeros,
    numZerosData,
    ssData,
    systematic,
    simulatorUsesOldNumpyRNG,
):
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    if systematic:
        modelGen = index_generator(rg, modelPrior)
        thetaGens: Dict[int, Generator[int, None, None]] = {}

    acceptedParticles: List[Tuple[int, Tuple, float, float]] = []
    numAttempts = 0

    while numAttempts < simulationBudget:
        numAttempts += 1

        if not systematic:
            m = sample_discrete_dist(modelPrior)
        else:
            m = next(modelGen)

        model = models[m]
        prior = priors[m]
        K = kdes[m]
        if K is None:
            continue

        if not systematic:
            i = sample_discrete_dist(K.weights)
        else:
            if m not in thetaGens.keys():
                thetaGens[m] = index_generator(rg, K.weights)
            i = next(thetaGens[m])

        mu = K.dataset[:, i].flatten()

        theta = sample_multivariate_normal(rg, mu, K.L)

        priorVal = prior.pdf(theta)
        if priorVal <= 0:
            continue

        if simulatorUsesOldNumpyRNG:
            xFake = model(theta)  # type: ignore
        else:
            xFake = model(rg, theta)  # type: ignore

        if matchZeros and not np.all(np.sum(xFake == 0, axis=0) == numZerosData):
            continue

        if sumstats is not None:
            ssFake = sumstats(xFake)
        else:
            ssFake = xFake

        if "max_dist" in inspect.signature(distance).parameters:
            dist = distance(ssData, ssFake, max_dist=eps)
        else:
            dist = distance(ssData, ssFake)

        if dist < eps:
            thetaLogWeight = np.log(priorVal) - gaussian_kde_logpdf(
                theta, K.d, K.n, K.dataset, K.weights, K.inv_cov, K.log_det
            )
            weight = np.exp(thetaLogWeight)

            if weight > 0:
                acceptedParticles.append((m, theta, weight, dist))

                if stopTaskAfterNParticles is not None and len(acceptedParticles) >= stopTaskAfterNParticles:
                    break

    return acceptedParticles, numAttempts


# Sample a population of particles
def sample_population(
    sg,
    t,
    parallel,
    modelPrior,
    models: Tuple[Model],
    priors: Tuple[Prior],
    prevFit,
    sumstats,
    distance,
    eps,
    popSize,
    matchZeros,
    numZerosData,
    ssData,
    recycling,
    systematic,
    prevNumSims,
    strictPopulationSize,
    simulatorUsesOldNumpyRNG,
):
    samples = []
    ms = []
    weights = []
    dists = []
    numSims = 0

    if t == 0:
        sample_first_iteration = joblib.delayed(_sample_one_first_iteration)
        seeds = (s.generate_state(1)[0] for s in sg.spawn(popSize))
        results = parallel(
            sample_first_iteration(
                seed, modelPrior, models, priors, sumstats, distance, ssData, simulatorUsesOldNumpyRNG
            )
            for seed in seeds
        )

        numSims = popSize
        for i in range(popSize):
            m, theta, weight, dist, _ = results[i]
            ms.append(m)
            samples.append(theta)
            weights.append(weight)
            dists.append(dist)

    else:
        sample = joblib.delayed(sample_particles)

        kdes = prevFit.fit_kdes()

        if strictPopulationSize:
            # If we are only going to simulate exactly n particles,
            # then we create n batches which each simulate one particle
            # and they just keep going until they get that one particle.
            numParallelTasks = popSize
            simulationBudget = np.inf
            stopTaskAfterNParticles = 1
        else:
            # Start by guessing how many simulations will be required
            # to generate at least n particles at this epsilon threshold.
            estNumSimsRequired = prevNumSims

            # Split the simulation burden evenly between each of the
            # CPU processes. Tell each core to give us as many particles
            # as it can find when given a fixed simulation budget.
            numParallelTasks = parallel.n_jobs
            simulationBudget = int(np.ceil(estNumSimsRequired / numParallelTasks))
            stopTaskAfterNParticles = None

        # bar = tqdm(total=n, position=0, leave=False)

        numParticles = 0
        while numParticles < popSize:
            seeds = (s.generate_state(1)[0] for s in sg.spawn(numParallelTasks))
            results = parallel(
                sample(
                    seed,
                    simulationBudget,
                    stopTaskAfterNParticles,
                    modelPrior,
                    models,
                    priors,
                    kdes,
                    sumstats,
                    distance,
                    eps,
                    matchZeros,
                    numZerosData,
                    ssData,
                    systematic,
                    simulatorUsesOldNumpyRNG,
                )
                for seed in seeds
            )

            for particles, simsUsed in results:
                numSims += simsUsed
                for particle in particles:
                    m, theta, weight, dist = particle
                    ms.append(m)
                    samples.append(theta)
                    weights.append(weight)
                    dists.append(dist)

                    numParticles += 1
                    # bar.update(1)

            # If we collect less than n particles, then we can reduce
            # the simulation budget for the next time around
            numParticlesLeft = popSize - numParticles
            if numParticlesLeft > 0 and not strictPopulationSize:
                estNumSimsRequired = 1.1 * numParticlesLeft * (numSims / max(numParticles, 1))
                simulationBudget = int(np.ceil(estNumSimsRequired / numParallelTasks))

        # bar.close()

    fit = Population(ms, weights, samples, dists, len(models))

    # Combine the previous generation with this one.
    if recycling and prevFit is not None:
        fit = fit.combine(prevFit)

    return fit, numSims


def validate_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=float).squeeze()


def validate_model_prior(modelPrior, M) -> np.ndarray:
    if not modelPrior:
        modelPrior = np.ones(M) / M
    return modelPrior


def validate_distance(
    sumstats, distance
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], float]]:
    if isinstance(distance, collections.abc.Sequence):
        sumstats = distance[0]
        distance = distance[1]

    return sumstats, distance


def get_summary_stats(obs, sumstats) -> np.ndarray:
    if sumstats is not None:
        ssData = sumstats(obs)
    else:
        ssData = obs
    return ssData


def take_best_n_particles(fit: Population, n: int) -> Tuple[Population, float]:
    """
    Create a subpopulation of particles by selecting the best n particles.
    A particle's quality is assessed by its distance value.
    """
    sortInds = np.argsort(fit.dists)
    return fit.subpopulation(sortInds[:n]), fit.dists[sortInds[n - 1]]


def reduce_population_size(fit: Population, targetESS: float, epsMin: float) -> Tuple[Population, float]:
    """
    Create a subpopulation of particles by discarding the worst particles until the
    ESS drops to a target value. A particle's quality is assessed by its distance value.
    """
    fit = fit.clone()
    totalESS = fit.total_ess()
    eps = np.max(fit.dists)

    while totalESS > targetESS:
        fit.drop_worst_particle()
        eps = np.max(fit.dists)
        totalESS = fit.total_ess()

        if eps < epsMin:
            # Don't bother aiming for an even better threshold
            # if the user is satisfied with epsMin.
            eps = epsMin
            break

    return fit, eps


def prepare_next_population(
    onFinalIteration: bool, popSize: int, epsMin: float, fit: Population
) -> Tuple[Population, float]:
    """
    After sampling a round in the sequential Monte Carlo algorithm, we
    discard particles in order to create a smaller population which represent
    a better fit to the data. The original population is unaltered.
    """
    if onFinalIteration or np.sort(fit.dists)[popSize - 1] < epsMin:
        # Take the best popSize particles to be the final population.
        nextFit, eps = take_best_n_particles(fit, popSize)
    else:
        # Otherwise, throw away enough particles until the ESS
        # drops to popSize/2.
        nextFit, eps = reduce_population_size(fit, popSize / 2, epsMin)

    # Also, if not finished SMC iterations, throw away models
    # which only have a couple of samples as these will just crash
    # the KDE function.
    if not onFinalIteration:
        nextFit.drop_small_models()

    return nextFit, eps


def print_header(popSize, T, numSumStats, numProcs):
    potentialPlural = "processes" if numProcs > 1 else "process"
    print(
        f"Starting ABC-SMC with population size of {popSize} and sample size "
        + f"of {T} (~> {numSumStats}) on {numProcs} {potentialPlural}."
    )


def print_update(t, eps, elapsed, numSims, totalSimulationCost, fit, nextFit):
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


def smc(
    numIters: int,
    popSize: int,
    obs: np.ndarray,
    models: Tuple[Model] | Model,
    priors: Tuple[Prior] | Prior,
    distance=wasserstein,
    sumstats=None,
    modelPrior=None,
    numProcs=1,
    epsMin=0,
    seed=None,
    verbose=False,
    matchZeros=False,
    recycling=True,
    systematic=False,
    strictPopulationSize=False,
    simulatorUsesOldNumpyRNG=False,
    showProgressBar=False,
    plotProgress=False,
    plotProgressRefLines=None,
):
    if numProcs == 1:
        strictPopulationSize = True

    models = cast(Tuple[Model], make_iterable(models))
    priors = cast(Tuple[Prior], make_iterable(priors))

    obs = cast(np.ndarray, validate_obs(obs))
    modelPrior = cast(np.ndarray, validate_model_prior(modelPrior, len(models)))
    sumstats, distance = validate_distance(sumstats, distance)  # type: ignore

    numZerosData = np.sum(obs == 0, axis=0)
    ssData = get_summary_stats(obs, sumstats)

    sg = SeedSequence(seed)

    if verbose:
        print_header(popSize, len(obs), len(ssData), numProcs)

    totalSimulationCost = 0
    eps = np.inf

    # To keep the linter happy, declare some variables as None temporarily
    numSims = None
    prevFit = None

    with joblib.Parallel(n_jobs=numProcs) as parallel:
        for t in range(0, numIters + 1):
            if showProgressBar and t == 1:
                bar = tqdm(total=numIters, position=0, leave=False)

            startTime = time()

            try:
                fit, numSims = sample_population(
                    sg,
                    t,
                    parallel,
                    modelPrior,
                    models,
                    priors,
                    prevFit,
                    sumstats,
                    distance,
                    eps,
                    popSize,
                    matchZeros,
                    numZerosData,
                    ssData,
                    recycling,
                    systematic,
                    numSims,
                    strictPopulationSize,
                    simulatorUsesOldNumpyRNG,
                )
            except KeyboardInterrupt:
                if t == 0:
                    print("A running approxbayescomp.smc(..) call was cancelled.")
                    raise
                else:
                    print(
                        "A running approxbayescomp.smc(..) call was cancelled, the previous population has been returned."
                    )
                    fit = prevFit
                    break

            elapsed = time() - startTime
            totalSimulationCost += numSims

            nextFit, eps = prepare_next_population(t == numIters, popSize, epsMin, fit)

            if verbose:
                print_update(t, eps, elapsed, numSims, totalSimulationCost, fit, nextFit)

            fit = nextFit
            prevFit = nextFit

            if plotProgress and len(models) == 1:
                plot_posteriors(fit, priors[0], refLines=plotProgressRefLines)
                plt.show()

            if eps < epsMin:
                if verbose:
                    print("Stopping now due to exceeding epsilon target.")
                break

            if showProgressBar and t > 0:
                bar.update(1)

    if showProgressBar:
        bar.close()

    return fit
