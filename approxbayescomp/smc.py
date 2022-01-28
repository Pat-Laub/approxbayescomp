# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
import inspect
from collections import namedtuple
from time import time

import joblib  # type: ignore
import numpy as np
import numpy.random as rnd
from numba import njit  # type: ignore
from numpy.random import SeedSequence, default_rng  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from tqdm.auto import tqdm

from .simulate import (
    sample_discrete_dist,
    sample_multivariate_normal,
    sample_uniform_dist,
    simulate_claim_data,
)
from .wasserstein import wass_dist, wass_sumstats
from .weighted import systematic_resample

try:
    PANDAS_INSTALLED = False
    import pandas

    PANDAS_INSTALLED = True
except ModuleNotFoundError:
    pass


@njit(nogil=True)
def numba_seed(seed: int):
    rnd.seed(seed)


Psi = namedtuple("Psi", ["name", "param"], defaults=["sum", 0.0])

Model = namedtuple(
    "Model",
    ["freq", "sev", "psi", "prior", "obsFreqs"],
    defaults=["ones", None, None, None, None],
)

SimulationModel = namedtuple("SimulationModel", ["simulator", "prior"])

Fit = namedtuple("Fit", ["models", "weights", "samples", "dists"])

# Currently it's difficult to get numba to compile a whole class, and in particular
# it can't handle the Prior classes. So, e.g. the 'SimpleIndependentUniformPrior' pulls
# out the key details from the IndependentUniformPrior class & turns it into a boring
# 'bag of data' (named tuple) which numba can handle/compile.
SimpleIndependentUniformPrior = namedtuple(
    "SimpleIndependentUniformPrior", ["lower", "upper", "width", "normConst"]
)
SimpleKDE = namedtuple(
    "SimpleKDE", ["dataset", "weights", "d", "n", "inv_cov", "L", "log_det"]
)


def kde(data: np.ndarray, weights: np.ndarray, bw: float = np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)


def compute_psi(freqs: np.ndarray, sevs: np.ndarray, psi):
    return _compute_psi(freqs, sevs, psi.name, psi.param)


@njit(nogil=True)
def _compute_psi(freqs, sevs, psi_name, psi_param):
    xs = -np.ones(len(freqs))
    i = 0

    if psi_name == "sum":
        for r, n in enumerate(freqs):
            xs[r] = np.sum(sevs[i : i + n])
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


@njit(nogil=True)
def uniform_pdf(theta, lower, upper, normConst):
    for i in range(len(theta)):
        if theta[i] <= lower[i] or theta[i] >= upper[i]:
            return 0
    return normConst


@njit(nogil=True)
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
    seed, models, modelPrior, sumstats, distance, ssData, T, simulatorUsesOldNumpyRNG
):
    rg = default_rng(seed)
    rnd.seed(seed)
    numba_seed(seed)

    # On the first iteration of SMC we sample from the prior
    # and accept everthing, so the code is a bit simpler.
    m = sample_discrete_dist(modelPrior)
    model = models[m]
    theta = sample_uniform_dist(model.prior.lower, model.prior.width)

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
            xFake = model.simulator(theta)
        else:
            xFake = model.simulator(rg, theta)

    dist = distance(ssData, sumstats(xFake))

    return m, theta, 1.0, dist, 1


@njit(nogil=True)
def num_zeros_match(numZerosData, xFake):
    if numZerosData >= 0:
        numZerosFake = 0
        for xFake_i in xFake:
            if xFake_i == 0:
                numZerosFake += 1

        return numZerosData == numZerosFake
    return True


def sample_particles(
    seed,
    simulationBudget,
    stopTaskAfterNParticles,
    models,
    modelPrior,
    kdes,
    sumstats,
    distance,
    eps,
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

        priorVal = uniform_pdf(
            theta, model.prior.lower, model.prior.upper, model.prior.normConst
        )
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
                xFake = model.simulator(theta)
            else:
                xFake = model.simulator(rg, theta)

        if not num_zeros_match(numZerosData, xFake):
            continue

        if "max_dist" in inspect.signature(distance).parameters:
            dist = distance(ssData, sumstats(xFake), max_dist=eps)
        else:
            dist = distance(ssData, sumstats(xFake))

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
    models,
    modelPrior,
    kdes,
    sumstats,
    distance,
    eps,
    n,
    numZerosData,
    ssData,
    T,
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
        seeds = (s.generate_state(1)[0] for s in sg.spawn(n))
        results = parallel(
            sample_first_iteration(
                seed,
                models,
                modelPrior,
                sumstats,
                distance,
                ssData,
                T,
                simulatorUsesOldNumpyRNG,
            )
            for seed in seeds
        )

        numSims = n
        for i in range(n):
            m, theta, weight, dist, _ = results[i]
            ms.append(m)
            samples.append(theta)
            weights.append(weight)
            dists.append(dist)

    else:
        sample = joblib.delayed(sample_particles)

        if strictPopulationSize:
            # If we are only going to simulate exactly n particles,
            # then we create n batches which each simulate one particle
            # and they just keep going until they get that one particle.
            numParallelTasks = n
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
        while numParticles < n:
            seeds = (s.generate_state(1)[0] for s in sg.spawn(numParallelTasks))
            results = parallel(
                sample(
                    seed,
                    simulationBudget,
                    stopTaskAfterNParticles,
                    models,
                    modelPrior,
                    kdes,
                    sumstats,
                    distance,
                    eps,
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
            numParticlesLeft = n - numParticles
            if numParticlesLeft > 0 and not strictPopulationSize:
                estNumSimsRequired = (
                    1.1 * numParticlesLeft * (numSims / max(numParticles, 1))
                )
                simulationBudget = int(np.ceil(estNumSimsRequired / numParallelTasks))

        # bar.close()

    ms = np.array(ms)
    weights = np.array(weights)
    weights /= np.sum(weights)
    samples = np.vstack(samples)
    dists = np.array(dists)

    return Fit(ms, weights, samples, dists), numSims


def group_samples_by_model(ms, samples, M):
    d = {m: [] for m in range(M)}

    for m, sample in zip(ms, samples):
        d[m].append(sample)

    for m in range(M):
        if len(d[m]) > 0:
            d[m] = np.vstack(d[m])
        else:
            d[m] = np.array([])

    return d


def fit_all_kdes(ms, samples, weights, M):
    simpleKDEs = []

    samplesGrouped = group_samples_by_model(ms, samples, M)

    for m in range(M):
        samples_m = samplesGrouped[m]

        K = None
        if samples_m.shape[0] >= 5:
            try:
                K = kde(samples_m, weights[ms == m])
                L = np.linalg.cholesky(K.covariance)
                log_det = 2 * np.log(np.diag(L)).sum()
                K = SimpleKDE(K.dataset, K.weights, K.d, K.n, K.inv_cov, L, log_det)
            except np.linalg.LinAlgError:
                pass

        simpleKDEs.append(K)

    return tuple(simpleKDEs)


def calculate_ess(M, ms, weights):
    # Calculate effective sample size for each model
    if M == 1:
        ESS = 1 / np.sum(weights ** 2)
    else:
        ESS = []
        for m in range(M):
            if (ms == m).sum() > 0:
                ESS.append(
                    np.sum(weights[ms == m]) ** 2 / np.sum(weights[ms == m] ** 2)
                )
            else:
                ESS.append(0)

    return np.round(ESS).astype(int)


def smc_setup(
    obs,
    models,
    sumstats,
    modelPrior,
    numProcs,
    matchZeros,
):

    if type(models) == Model or type(models) == SimulationModel:
        models = [models]
        modelPrior = np.array([1.0])

    M = len(models)

    if not modelPrior:
        modelPrior = np.ones(M) / M

    if (
        type(models[0]) == Model
        and type(models[0].freq) == str
        and models[0].freq.startswith("bivariate")
    ):
        T = obs.shape[0]
        numZerosData = (
            (np.sum(obs[:, 0] == 0), np.sum(obs[:, 1] == 0)) if matchZeros else (-1, -1)
        )
    else:
        T = len(obs)
        numZerosData = np.sum(obs == 0) if matchZeros else -1

        if PANDAS_INSTALLED and type(obs) == pandas.core.series.Series:
            obs = obs.to_numpy()

    ssData = sumstats(obs)
    if not np.isscalar(ssData) and len(ssData) > 1:
        numSumStats = len(np.array(ssData).flatten())
    else:
        numSumStats = 1

    newPriors = [
        SimpleIndependentUniformPrior(
            model.prior.lower,
            model.prior.upper,
            model.prior.widths,
            model.prior.normConst,
        )
        for model in models
    ]

    newModels = []
    for model, newPrior in zip(models, newPriors):
        if type(model) == Model:
            if model.psi:
                newPsi = model.psi
            else:
                newPsi = Psi("severities")
            newModel = Model(model.freq, model.sev, newPsi, newPrior)
        else:
            newModel = SimulationModel(model.simulator, newPrior)
        newModels.append(newModel)

    models = tuple(newModels)

    return T, M, models, modelPrior, numProcs, numSumStats, numZerosData, ssData


def take_best_n_particles(fit, n):
    """
    Create a subpopulation of particles by selecting the best n particles.
    A particle's quality is assessed by its distance value.
    """
    ms, weights, samples, dists = fit.models, fit.weights, fit.samples, fit.dists
    eps = np.sort(dists)[n - 1]
    bestParticles = dists <= eps
    ms = ms[bestParticles]
    weights = weights[bestParticles]
    weights /= np.sum(weights)
    samples = samples[bestParticles, :]
    dists = dists[bestParticles]
    return Fit(ms, weights, samples, dists), eps


def reduce_population_size(fit, targetESS, epsMin, M):
    """
    Create a subpopulation of particles by discarding the worst particles until the
    ESS drops to a target value. A particle's quality is assessed by its distance value.
    """
    ms, weights, samples, dists = fit.models, fit.weights, fit.samples, fit.dists
    ESS = calculate_ess(M, ms, weights)
    eps = np.max(dists)

    # Go through each particle, starting at the furthest ones from the observed data.
    for ind in reversed(np.argsort(dists)):
        if np.sum(ESS) <= targetESS:
            break

        eps = dists[ind]
        weights[ind] = 0
        weights /= np.sum(weights)
        ESS = calculate_ess(M, ms, weights)

        if eps < epsMin:
            # Don't bother aiming for an even better threshold
            # if the user is satisfied with epsMin.
            eps = epsMin
            break

    ms = ms[weights > 0]
    samples = samples[weights > 0, :]
    dists = dists[weights > 0]
    weights = weights[weights > 0]
    weights /= np.sum(weights)

    return Fit(ms, weights, samples, dists), eps


def combine_populations(fit1, fit2):
    ms = np.concatenate([fit1.models, fit2.models])
    weights = np.concatenate([fit1.weights, fit2.weights])
    weights /= np.sum(weights)
    samples = np.concatenate([fit1.samples, fit2.samples], axis=0)
    dists = np.concatenate([fit1.dists, fit2.dists])
    return Fit(ms, weights, samples, dists)


def prepare_next_population(onFinalIteration, popSize, epsMin, M, fit):
    """
    After sampling a round in the sequential Monte Carlo algorithm, we
    discard particles in order to create a smaller population which represent
    a better fit to the data.
    """
    if np.sort(fit.dists)[popSize - 1] < epsMin or onFinalIteration:
        # Take the best popSize particles to be the final population.
        fit, eps = take_best_n_particles(fit, popSize)
    else:
        # Otherwise, throw away enough particles until the ESS
        # drops to popSize/2.
        fit, eps = reduce_population_size(fit, popSize / 2, epsMin, M)

    # Also, if not finished SMC iterations, throw away models
    # which only have a couple of samples as these will just crash
    # the KDE function.
    ms, weights, samples, dists = fit.models, fit.weights, fit.samples, fit.dists
    modelPopulations = [np.sum(ms == m) for m in range(M)]
    modelWeights = np.array([np.sum(weights[ms == m]) for m in range(M)])
    modelWeights /= np.sum(modelWeights)

    if not onFinalIteration:
        for m in range(M):
            if modelPopulations[m] < 5 or modelWeights[m] == 0:
                samples = samples[ms != m, :]
                dists = dists[ms != m]
                weights = weights[ms != m]
                weights /= np.sum(weights)
                ms = ms[ms != m]

    return Fit(ms, weights, samples, dists), eps


def print_header(popSize, T, numSumStats, numProcs):
    potentialPlural = "processes" if numProcs > 1 else "process"
    print(
        f"Starting ABC-SMC with population size of {popSize} and sample size "
        + f"of {T} (~> {numSumStats}) on {numProcs} {potentialPlural}."
    )


def print_update(
    t,
    eps,
    elapsed,
    popSizeBefore,
    ESSBefore,
    numSims,
    totalSimulationCost,
    M,
    ms,
    weights,
):
    """
    After each sequential Monte Carlo iteration, print out a summary
    of the just-sampled population, and of the subpopulation which was
    prepared for the next round.
    """
    popSizeAfter = len(ms)
    ESSAfter = calculate_ess(M, ms, weights)
    modelPopulations = [np.sum(ms == m) for m in range(M)]
    modelWeights = [np.sum(weights[ms == m]) for m in range(M)]
    modelWeights = np.round(np.array(modelWeights) / np.sum(modelWeights), 2)

    update = (
        f"Finished SMC iteration {t}, " if t > 0 else "Finished sampling from prior, "
    )
    update += f"eps = {eps:.2f}, "
    elapsedMins = np.round(elapsed / 60, 1)
    update += f"time = {np.round(elapsed)}s / {elapsedMins}m, "
    update += f"popSize = {popSizeBefore} -> {popSizeAfter}, "
    update += f"ESS = {ESSBefore} -> {ESSAfter}, "
    update += f"# sims = {numSims}, total # sims = {totalSimulationCost}"
    if M > 1:
        update += f"\n\tmodel populations = {modelPopulations}, "
        update += f"model weights = {modelWeights}"
    print(update)


def smc(
    numIters,
    popSize,
    obs,
    models,
    sumstats=wass_sumstats,
    distance=wass_dist,
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
):
    obs = np.asarray(obs)
    if numProcs == 1:
        strictPopulationSize = True

    T, M, models, modelPrior, numProcs, numSumStats, numZerosData, ssData = smc_setup(
        obs,
        models,
        sumstats,
        modelPrior,
        numProcs,
        matchZeros,
    )

    sg = SeedSequence(seed)

    if verbose:
        print_header(popSize, T, numSumStats, numProcs)

    totalSimulationCost = 0
    interrupted = False
    eps = np.inf
    kdes = None
    showProgressBar = not verbose

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
                    models,
                    modelPrior,
                    kdes,
                    sumstats,
                    distance,
                    eps,
                    popSize,
                    numZerosData,
                    ssData,
                    T,
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
                    interrupted = True

            elapsed = time() - startTime
            totalSimulationCost += numSims

            # Combine the previous generation with this one.
            if recycling and t > 0:
                fit = combine_populations(prevFit, fit)

            # Store the original population size and effective sample size
            # before we start throwing away particles.
            if verbose:
                popSizeBefore = len(fit.models)
                ESSBefore = calculate_ess(M, fit.models, fit.weights)

            fit, eps = prepare_next_population(
                t == numIters or interrupted, popSize, epsMin, M, fit
            )

            if verbose:
                print_update(
                    t,
                    eps,
                    elapsed,
                    popSizeBefore,
                    ESSBefore,
                    numSims,
                    totalSimulationCost,
                    M,
                    fit.models,
                    fit.weights,
                )

            if interrupted:
                break

            if eps < epsMin:
                if verbose:
                    print("Stopping now due to exceeding epsilon target.")
                break

            if showProgressBar and t > 0:
                bar.update(1)

            # Store this generation to be recycled in the next one.
            if recycling:
                prevFit = fit

            if t < numIters:
                kdes = fit_all_kdes(fit.models, fit.samples, fit.weights, M)

    if showProgressBar:
        bar.close()

    return fit
