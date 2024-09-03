# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from time import time
from typing import Callable, Optional, Union, Generator, cast

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
from .population import Population, take_best_n_particles, reduce_population_size
from .prior import Prior
from .simulate import sample_discrete_dist, sample_multivariate_normal
from .kde import gaussian_kde_logpdf
from .utils import (
    index_generator,
    make_iterable,
    numba_seed,
    print_header,
    print_update,
    validate_model_prior,
    validate_obs,
    validate_distance,
)

# Suppress a numba.PerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Create a type alias for the simulator method
Simulator = Union[Callable[[np.ndarray], np.ndarray], Callable[[rnd.Generator, np.ndarray], np.ndarray]]


@dataclass
class SamplingConfig:
    sumstats: Callable[[np.ndarray], np.ndarray]
    distance: Callable[[np.ndarray, np.ndarray], float]
    matchZeros: bool
    numZerosData: int
    ssData: np.ndarray
    systematic: bool
    recycling: bool
    strictPopulationSize: bool


class Model:
    def __init__(self, simulator: Simulator, prior: Prior, simulatorUsesOldNumpyRNG: bool = False):
        self.simulator = simulator
        self.prior = prior
        self.simulatorUsesOldNumpyRNG = simulatorUsesOldNumpyRNG

    def __call__(self, theta, rg=None):
        if self.simulatorUsesOldNumpyRNG:
            return self.simulator(theta)
        return self.simulator(rg, theta)


def sample_one_first_iteration(
    seed: int, modelPrior: np.ndarray, models: list[Model], opts: SamplingConfig
) -> tuple[int, np.ndarray, float, float, int]:
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    # On the first iteration of SMC we sample from the prior
    # and accept everthing, so the code is a bit simpler.
    m = sample_discrete_dist(modelPrior)
    model = models[m]
    theta = model.prior.sample(rg)
    xFake = model(theta, rg)
    dist = opts.distance(opts.ssData, opts.sumstats(xFake))

    return m, theta, 1.0, dist, 1


def sample_particles(
    seed: int,
    simulationBudget: Union[int, float],
    stopTaskAfterNParticles: Optional[int],
    modelPrior: np.ndarray,
    models: list[Model],
    kdes,
    eps: float,
    opts: SamplingConfig,
) -> tuple[list[tuple[int, tuple, float, float]], int]:
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    if opts.systematic:
        modelGen = index_generator(rg, modelPrior)
        thetaGens: dict[int, Generator[int, None, None]] = {}

    acceptedParticles: list[tuple[int, tuple, float, float]] = []
    numAttempts = 0

    while numAttempts < simulationBudget:
        numAttempts += 1

        if not opts.systematic:
            m = sample_discrete_dist(modelPrior)
        else:
            m = next(modelGen)

        model = models[m]
        K = kdes[m]
        if K is None:
            continue

        if not opts.systematic:
            i = sample_discrete_dist(K.weights)
        else:
            if m not in thetaGens.keys():
                thetaGens[m] = index_generator(rg, K.weights)
            i = next(thetaGens[m])

        mu = K.dataset[:, i].flatten()

        theta = sample_multivariate_normal(rg, mu, K.L)

        priorVal = model.prior.pdf(theta)
        if priorVal <= 0:
            continue

        xFake = model(theta, rg)

        if opts.matchZeros and not np.all(np.sum(xFake == 0, axis=0) == opts.numZerosData):
            continue

        if "max_dist" in inspect.signature(opts.distance).parameters:
            dist = opts.distance(opts.ssData, opts.sumstats(xFake), max_dist=eps)  # type: ignore
        else:
            dist = opts.distance(opts.ssData, opts.sumstats(xFake))

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
def sample_first_population(
    sg, parallel, modelPrior: np.ndarray, models: list[Model], popSize: int, opts: SamplingConfig
) -> tuple[Population, int]:
    samples = []
    ms = []
    weights = []
    dists = []
    numSims = 0

    sample_first_iteration = joblib.delayed(sample_one_first_iteration)
    seeds = (s.generate_state(1)[0] for s in sg.spawn(popSize))
    results = parallel(sample_first_iteration(seed, modelPrior, models, opts) for seed in seeds)

    numSims = popSize
    for i in range(popSize):
        m, theta, weight, dist, _ = results[i]
        ms.append(m)
        samples.append(theta)
        weights.append(weight)
        dists.append(dist)

    fit = Population(ms, weights, samples, dists, len(models))

    return fit, numSims


# Sample a population of particles
def sample_population(
    sg,
    parallel,
    modelPrior: np.ndarray,
    models: list[Model],
    prevFit: Population,
    eps: float,
    popSize: int,
    prevNumSims: int,
    opts: SamplingConfig,
):
    samples = []
    ms = []
    weights = []
    dists = []
    numSims = 0

    sample = joblib.delayed(sample_particles)

    kdes = prevFit.fit_kdes()

    if opts.strictPopulationSize:
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
            sample(seed, simulationBudget, stopTaskAfterNParticles, modelPrior, models, kdes, eps, opts)
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
        if numParticlesLeft > 0 and not opts.strictPopulationSize:
            estNumSimsRequired = int(np.ceil(1.1 * numParticlesLeft * (numSims / max(numParticles, 1))))
            simulationBudget = int(np.ceil(estNumSimsRequired / numParallelTasks))

    # bar.close()

    fit = Population(ms, weights, samples, dists, len(models))

    # Combine the previous generation with this one.
    if opts.recycling and prevFit is not None:
        fit = fit.combine(prevFit)

    return fit, numSims


def prepare_next_population(
    onFinalIteration: bool, popSize: int, epsMin: float, fit: Population
) -> tuple[Population, float]:
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


def smc(
    numIters: int,
    popSize: int,
    obs: np.ndarray,
    simulators: Union[list[Simulator], Simulator],
    priors: Union[tuple[Prior], Prior],
    distance=wasserstein,
    sumstats=None,
    modelPrior: Optional[np.ndarray] = None,
    numProcs: int = 1,
    epsMin: float = 0,
    minEpsImprovement: float = 1e-4,
    seed: Optional[int] = None,
    verbose: bool = False,
    matchZeros: bool = False,
    recycling: bool = True,
    systematic: bool = False,
    strictPopulationSize: bool = False,
    simulatorUsesOldNumpyRNG: bool = False,
    showProgressBar: bool = False,
    plotProgress: bool = False,
    plotProgressRefLines: Optional[tuple[float]] = None,
):
    if numProcs == 1:
        strictPopulationSize = True

    models = [
        Model(simulator, prior, simulatorUsesOldNumpyRNG)
        for simulator, prior in zip(make_iterable(simulators), make_iterable(priors))
    ]

    obs = cast(np.ndarray, validate_obs(obs))
    modelPrior = cast(np.ndarray, validate_model_prior(modelPrior, len(models)))
    sumstats, distance = validate_distance(sumstats, distance)

    numZerosData = np.sum(obs == 0, axis=0)
    ssData = sumstats(obs)

    sg = SeedSequence(seed)

    if verbose:
        print_header(popSize, len(obs), len(ssData), numProcs)

    totalSimulationCost = 0
    eps = np.inf
    prevEps = np.inf

    # To keep the linter happy, declare some variables as None temporarily
    numSims = cast(int, None)
    prevFit = cast(Population, None)

    opts = SamplingConfig(
        sumstats, distance, matchZeros, numZerosData, ssData, systematic, recycling, strictPopulationSize
    )

    with joblib.Parallel(n_jobs=numProcs) as parallel:
        for t in range(0, numIters + 1):
            if showProgressBar and t == 1:
                bar = tqdm(total=numIters, position=0, leave=False)

            startTime = time()

            try:
                if t == 0:
                    fit, numSims = sample_first_population(sg, parallel, modelPrior, models, popSize, opts)
                else:
                    fit, numSims = sample_population(
                        sg, parallel, modelPrior, models, prevFit, eps, popSize, numSims, opts
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

            # Check for relative improvement in epsilon
            if t > 0:
                relativeImprovement = (prevEps - eps) / prevEps if prevEps > 0 else 0
                if relativeImprovement < minEpsImprovement:
                    if verbose:
                        print(
                            f"Stopping now due to marginal relative improvement in epsilon: {relativeImprovement:.6f}"
                        )
                    break

            prevEps = eps

            fit = nextFit
            prevFit = nextFit

            if plotProgress and len(models) == 1:
                plot_posteriors(fit, models[0].prior, refLines=plotProgressRefLines)
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
