# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
from __future__ import annotations

import collections
import inspect
import warnings
from time import time
from typing import Optional, Tuple

import joblib  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from numba import float64, int64, njit, void  # type: ignore
from numba.core.errors import NumbaPerformanceWarning  # type: ignore
from numba.core.types import unicode_type  # type: ignore
from numpy.random import SeedSequence, default_rng  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from .distance import wasserstein
from .plot import plot_posteriors
from .simulate import (
    sample_discrete_dist,
    sample_multivariate_normal,
    simulate_claim_data,
)
from .weighted import systematic_resample

# Suppress a numba.PerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@njit(void(int64), nogil=True)
def numba_seed(seed):
    rnd.seed(seed)


Psi = collections.namedtuple("Psi", ["name", "param"], defaults=["sum", 0.0])

Model = collections.namedtuple(
    "Model",
    ["freq", "sev", "psi", "obsFreqs"],
    defaults=["ones", None, None, None],
)


def kde(data: np.ndarray, weights: np.ndarray, bw: float = np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)


SimpleKDE = collections.namedtuple(
    "SimpleKDE", ["dataset", "weights", "d", "n", "inv_cov", "L", "log_det"]
)


class Population(object):
    """
    A Population object stores a collection of particles in the
    SMC procedure. Each particle is a potential theta parameter which
    could explain the observed data. Each theta has a corresponding
    weight, belongs to a specific model (as we may be fitting multiple
    competing models simultaneously), and has an observed distance of
    its fake data to the observed data.
    """

    def __init__(self, models, weights, samples, dists, M) -> None:
        self.models = np.array(models)
        self.weights = np.array(weights)
        self.weights /= np.sum(weights)
        if type(samples) == list:
            self.samples = np.vstack(samples)
        else:
            self.samples = np.array(samples)
        self.dists = np.array(dists)
        self.M = M

    def size(self) -> int:
        return len(self.models)

    def model_sizes(self) -> Tuple[int, ...]:
        return tuple(np.sum(self.models == m) for m in range(self.M))

    def model_weights(self) -> Tuple[float, ...]:
        return tuple(np.sum(self.weights[self.models == m]) for m in range(self.M))

    def clone(self) -> Population:
        """
        Create a deep copy of this population object.
        """
        return Population(
            self.models.copy(),
            self.weights.copy(),
            self.samples.copy(),
            self.dists.copy(),
            self.M,
        )

    def subpopulation(self, keep) -> Population:
        """
        Create a subpopulation of particles from this population where we keep
        only the particles at the locations of True in the supplied boolean vector.
        """
        return Population(
            self.models[keep],
            self.weights[keep],
            self.samples[keep, :],
            self.dists[keep],
            self.M,
        )

    def combine(self, other) -> Population:
        """
        Combine this population with another to create one larger population.
        """
        ms = np.concatenate([self.models, other.models])
        samples = np.concatenate([self.samples, other.samples], axis=0)
        dists = np.concatenate([self.dists, other.dists])

        # Some care needs to be taken to adjust the weights when combining.
        # See Appendix B.1 of Leah South's PhD thesis.
        # https://eprints.qut.edu.au/132155/1/Leah_South_Thesis.pdf
        popWeights = np.array((self.total_ess(), other.total_ess()), dtype=np.float64)
        popWeights /= sum(popWeights)
        weights = np.concatenate(
            [popWeights[0] * self.weights, popWeights[1] * other.weights]
        )

        return Population(ms, weights, samples, dists, self.M)

    def drop_worst_particle(self) -> None:
        """
        Throw away the particle in this population which has the largest
        distance to the observed data.
        """
        dropIndex = np.argmax(self.dists)
        self.models = np.delete(self.models, dropIndex, 0)
        self.weights = np.delete(self.weights, dropIndex, 0)
        self.samples = np.delete(self.samples, dropIndex, 0)
        self.dists = np.delete(self.dists, dropIndex, 0)
        self.weights /= np.sum(self.weights)

    def drop_small_models(self) -> None:
        """
        Throw away the particles which correspond to models which
        have an extremely small population size.
        """
        modelPopulations = self.model_sizes()
        modelWeights = self.model_weights()

        for m in range(self.M):
            if modelPopulations[m] < 5 or modelWeights[m] == 0:
                keep = self.models != m
                self.models = self.models[keep]
                self.weights = self.weights[keep]
                self.weights /= np.sum(self.weights)
                self.samples = self.samples[keep, :]
                self.dists = self.dists[keep]

    def ess(self) -> Tuple[int, ...]:
        """
        Calculate the effective sample size (ESS) for each model in this population.
        """
        essPerModel = []
        for modelNum in set(self.models):
            weightsForThisModel = self.weights[self.models == modelNum]
            weightsForThisModel /= np.sum(weightsForThisModel)
            essPerModel.append(int(np.round(1 / np.sum(weightsForThisModel ** 2))))

        return tuple(essPerModel)

    def total_ess(self) -> int:
        """
        Calculate the total effective sample size (ESS) of this population.
        That is, add together the ESS for each model under consideration.
        """
        return sum(self.ess())

    def fit_kdes(self) -> Tuple[Optional[SimpleKDE], ...]:
        """
        Fit a kernel density estimator (KDE) to each model's subpopulation
        in this population. Return all the KDEs in tuple. If there isn't
        enough data for some model to fit a KDE, that entry will be None.
        """
        kdes = []

        for m in range(self.M):
            samplesForThisModel = self.samples[self.models == m, :]
            weightsForThisModel = self.weights[self.models == m]

            K = None
            if samplesForThisModel.shape[0] >= 5:
                try:
                    K = kde(samplesForThisModel, weightsForThisModel)
                    L = np.linalg.cholesky(K.covariance)  # type: ignore
                    log_det = 2 * np.log(np.diag(L)).sum()
                    K = SimpleKDE(K.dataset, K.weights, K.d, K.n, K.inv_cov, L, log_det)
                except np.linalg.LinAlgError:  # type: ignore
                    pass

            kdes.append(K)

        return tuple(kdes)


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
    nogil=True,
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


def _sample_one_first_iteration(
    seed,
    modelPrior,
    models,
    priors,
    sumstats,
    distance,
    ssData,
    T,
    simulatorUsesOldNumpyRNG,
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

    if type(model) == Model:
        claimsFake = simulate_claim_data(rg, T, model.freq, model.sev, theta)
        if type(model.freq) == str and model.freq.startswith("bivariate"):
            xFake1 = _compute_psi(
                claimsFake[0][0], claimsFake[0][1], model.psi.name, model.psi.param
            )
            xFake2 = _compute_psi(
                claimsFake[1][0], claimsFake[1][1], model.psi.name, model.psi.param
            )

            xFake = np.vstack([xFake1, xFake2]).T
        else:
            xFake = _compute_psi(
                claimsFake[0], claimsFake[1], model.psi.name, model.psi.param
            )
    else:
        if simulatorUsesOldNumpyRNG:
            xFake = model(theta)
        else:
            xFake = model(rg, theta)

    if sumstats is not None:
        dist = distance(ssData, sumstats(xFake))
    else:
        dist = distance(ssData, xFake)

    return m, theta, 1.0, dist, 1


def sample_particles(
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
    T,
    systematic,
    simulatorUsesOldNumpyRNG,
):
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    if systematic:
        modelGen = index_generator(rg, modelPrior)
        thetaGens = {}

    acceptedParticles = []
    numAttempts = 0

    while (
        len(acceptedParticles) < stopTaskAfterNParticles
        and numAttempts < simulationBudget
    ):
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

        if type(model) == Model:
            claimsFake = simulate_claim_data(rg, T, model.freq, model.sev, theta)
            if type(model.freq) == str and model.freq.startswith("bivariate"):
                xFake1 = _compute_psi(
                    claimsFake[0][0], claimsFake[0][1], model.psi.name, model.psi.param
                )
                xFake2 = _compute_psi(
                    claimsFake[1][0], claimsFake[1][1], model.psi.name, model.psi.param
                )
                xFake = np.vstack([xFake1, xFake2]).T
            else:
                xFake = _compute_psi(
                    claimsFake[0], claimsFake[1], model.psi.name, model.psi.param
                )
        else:
            if simulatorUsesOldNumpyRNG:
                xFake = model(theta)
            else:
                xFake = model(rg, theta)

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

    return acceptedParticles, numAttempts


# Sample a population of particles
def sample_population(
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
    T,
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
                seed,
                modelPrior,
                models,
                priors,
                sumstats,
                distance,
                ssData,
                T,
                simulatorUsesOldNumpyRNG,
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
            stopTaskAfterNParticles = np.inf

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
                    T,
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
                estNumSimsRequired = (
                    1.1 * numParticlesLeft * (numSims / max(numParticles, 1))
                )
                simulationBudget = int(np.ceil(estNumSimsRequired / numParallelTasks))

        # bar.close()

    fit = Population(ms, weights, samples, dists, len(models))

    # Combine the previous generation with this one.
    if recycling and prevFit is not None:
        fit = fit.combine(prevFit)

    return fit, numSims


def smc_setup(
    obs,
    modelPrior,
    models,
    priors,
    sumstats,
    distance,
):
    obs = np.asarray(obs, dtype=float).squeeze()
    T = obs.shape[0]

    if type(models) == Model or callable(models):
        modelPrior = np.array([1.0])
        models = [models]
        priors = [priors]

    M = len(models)

    if not modelPrior:
        modelPrior = np.ones(M) / M

    numZerosData = np.sum(obs == 0, axis=0)

    if isinstance(distance, collections.abc.Sequence):
        sumstats = distance[0]
        distance = distance[1]

    if sumstats is not None:
        ssData = sumstats(obs)
    else:
        ssData = obs

    if not np.isscalar(ssData) and len(ssData) > 1:
        numSumStats = len(np.array(ssData).flatten())
    else:
        numSumStats = 1

    newModels = []
    for model in models:
        if type(model) == Model:
            if model.psi:
                newPsi = model.psi
            else:
                newPsi = Psi("severities")
            newModel = Model(model.freq, model.sev, newPsi)
        else:
            newModel = model
        newModels.append(newModel)

    newModels = tuple(newModels)

    return (
        obs,
        T,
        modelPrior,
        priors,
        newModels,
        numSumStats,
        numZerosData,
        sumstats,
        distance,
        ssData,
    )


def take_best_n_particles(fit: Population, n: int) -> Tuple[Population, float]:
    """
    Create a subpopulation of particles by selecting the best n particles.
    A particle's quality is assessed by its distance value.
    """
    eps = np.sort(fit.dists)[n - 1]
    keep = fit.dists <= eps
    return fit.subpopulation(keep), eps


def reduce_population_size(
    fit: Population, targetESS: float, epsMin: float
) -> Tuple[Population, float]:
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
    update = (
        f"Finished SMC iteration {t}, " if t > 0 else "Finished sampling from prior, "
    )
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
    numIters,
    popSize,
    obs,
    models,
    priors,
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
    simulatorUsesOldNumpyRNG=True,
    showProgressBar=False,
    plotProgress=False,
    plotProgressRefLines=None,
):
    if numProcs == 1:
        strictPopulationSize = True

    (
        obs,
        T,
        modelPrior,
        priors,
        models,
        numSumStats,
        numZerosData,
        sumstats,
        distance,
        ssData,
    ) = smc_setup(
        obs,
        modelPrior,
        models,
        priors,
        sumstats,
        distance,
    )

    sg = SeedSequence(seed)

    if verbose:
        print_header(popSize, T, numSumStats, numProcs)

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
                    T,
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
                print_update(
                    t, eps, elapsed, numSims, totalSimulationCost, fit, nextFit
                )

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
