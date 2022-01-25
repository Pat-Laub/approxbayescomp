# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
from collections import namedtuple
from time import time

import joblib  # type: ignore
import numpy as np
import numpy.random as rnd
from numba import njit  # type: ignore
from numpy.random import SeedSequence, default_rng  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from tqdm.auto import tqdm

from .simulate import sample_discrete_dist, sim_multivariate_normal, simulate_claim_data
from .wasserstein import wass_dist, wass_sumstats
from .weighted import systematic_resample

try:
    PANDAS_INSTALLED = False
    import pandas

    PANDAS_INSTALLED = True
except ModuleNotFoundError:
    pass


@njit()
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
def uniform_sampler(lower, width):
    d = len(lower)
    theta = np.empty(d, np.float64)
    for i in range(d):
        theta[i] = lower[i] + width[i] * rnd.random()
    return theta


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
    seed, models, modelPrior, sumstats, distance, ssData, T
):
    rg = default_rng(seed)
    numba_seed(seed)

    # On the first iteration of SMC we sample from the prior
    # and accept everthing, so the code is a bit simpler.
    m = sample_discrete_dist(modelPrior)
    model = models[m]
    theta = uniform_sampler(model.prior.lower, model.prior.width)

    if type(model) == Model:
        claimsFake = simulate_claim_data(
            rg, T, model.freq, model.sev, theta
        )  # , obsFreqs
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
):
    rg = default_rng(seed)
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

        theta = sim_multivariate_normal(rg, mu, K.L)

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
                if not num_zeros_match(numZerosData[0], xFake1):
                    continue

                xFake2 = _compute_psi(
                    claimsFake[1][0], claimsFake[1][1], model.psi.name, model.psi.param
                )
                if not num_zeros_match(numZerosData[1], xFake2):
                    continue

                xFake = np.vstack([xFake1, xFake2]).T
            else:
                xFake = _compute_psi(
                    claimsFake[0], claimsFake[1], model.psi.name, model.psi.param
                )

                if not num_zeros_match(numZerosData, xFake):
                    continue

        else:
            xFake = model.simulator(rg, theta)
            if not num_zeros_match(numZerosData, xFake):
                continue

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
            estNumSimsRequired = prevNumSims / 10
            # estNumSimsRequired = int(np.ceil(2.5 * prevNumSims))

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
    dists = np.array(dists)

    weights /= np.sum(weights)
    if len(models) == 1:
        samples = np.array(samples)

    return ms, weights, samples, dists, numSims


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


def prepare_next_population(
    numIters,
    popSize,
    recycling,
    verbose,
    saveIters,
    M,
    t,
    eps,
    ms,
    weights,
    samples,
    dists,
    numSims,
    elapsed,
):

    popSize0 = len(ms)
    ESS0 = calculate_ess(M, ms, weights)

    # If we're not finished with the SMC iterations, throw
    # away enough particles to reduce the ESS to a target level.
    # This is done by decreasing epsilon for the next iteration.
    if t < numIters:
        ESS = calculate_ess(M, ms, weights)

        # Go through each particle, starting at the furthest
        # ones from the observed data.
        for ind in reversed(np.argsort(dists)):
            if np.sum(ESS) <= popSize / 2:
                break

            eps = dists[ind]

            weights /= 1 - weights[ind]
            weights[ind] = 0

            ESS = calculate_ess(M, ms, weights)

        ms = ms[weights > 0]
        samples = samples[weights > 0, :]
        dists = dists[weights > 0]
        weights = weights[weights > 0]
        weights /= np.sum(weights)
    else:
        if recycling:
            while len(ms) > popSize:
                argmax = np.argmax(dists)
                ms = np.delete(ms, argmax)
                weights = np.delete(weights, argmax)
                samples = np.delete(samples, argmax, axis=0)
                dists = np.delete(dists, argmax)

            weights /= np.sum(weights)

    # Also, if not finished SMC iterations, throw away models
    # which only have a couple of samples. These will just crash
    # the KDE function.
    modelPopulations = [np.sum(ms == m) for m in range(M)]
    modelWeights = np.array([np.sum(weights[ms == m]) for m in range(M)])
    modelWeights /= np.sum(modelWeights)
    if t < numIters:
        for m in range(M):
            if modelPopulations[m] < 5 or modelWeights[m] == 0:
                samples = samples[ms != m, :]
                dists = dists[ms != m]
                weights = weights[ms != m]
                weights /= np.sum(weights)
                ms = ms[ms != m]

        kdes = fit_all_kdes(ms, samples, weights, M)
    else:
        # As we're finishing up, we won't need another set of KDEs
        kdes = None

    modelPopulations = [np.sum(ms == m) for m in range(M)]
    modelWeights = [np.sum(weights[ms == m]) for m in range(M)]
    modelWeights = np.round(np.array(modelWeights) / np.sum(modelWeights), 2)

    if verbose:
        ESS = calculate_ess(M, ms, weights)
        update = (
            f"Finished SMC iteration {t}, "
            if t > 0
            else "Finished sampling from prior, "
        )
        update += f"eps = {eps:.2f}, "
        elapsedMins = np.round(elapsed / 60, 1)
        update += f"time = {np.round(elapsed)}s / {elapsedMins}m, "
        update += f"popSize = {popSize0} -> {len(ms)}, "
        update += f"ESS = {ESS0} -> {ESS}, numSims = {numSims}"
        if M > 1:
            update += f"\n\tmodel populations = {modelPopulations}, "
            update += f"model weights = {modelWeights}"
        print(update)
    else:
        ESS = None

    if saveIters:
        np.savetxt(f"smc-samples-{t:02}.txt", samples)
        np.savetxt(f"smc-weights-{t:02}.txt", weights)
        np.savetxt(f"smc-dists-{t:02}.txt", dists)

    return eps, kdes, ms, samples, dists, weights, ESS


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
    saveIters=False,
    seed=None,
    verbose=False,
    matchZeros=False,
    recycling=True,
    systematic=False,
    strictPopulationSize=False,
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
        potentialPlural = "processes" if numProcs > 1 else "process"
        print(
            f"Starting ABC-SMC with population size of {popSize} and sample size "
            + f"of {T} (~> {numSumStats}) on {numProcs} {potentialPlural}."
        )

    totalSimulationCost = 0
    eps = np.inf
    kdes = None

    # To keep the linter happy, declare some variables as None temporarily
    numSims = None
    ms = weights = samples = dists = None
    if recycling:
        oldMs = oldWeights = oldSamples = oldDists = None

    showProgressBar = not verbose

    with joblib.Parallel(n_jobs=numProcs) as parallel:
        for t in range(0, numIters + 1):
            if eps <= epsMin:
                if verbose:
                    print("Stopping now due to exceeding epsilon target.")
                break

            if showProgressBar and t == 1:
                bar = tqdm(total=numIters, position=0, leave=False)

            startTime = time()

            try:
                newMs, newWeights, newSamples, newDists, numSims = sample_population(
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
                )
            except KeyboardInterrupt:
                if t == 0:
                    print("A running approxbayescomp.smc(..) call was cancelled.")
                    raise
                else:
                    print(
                        "A running approxbayescomp.smc(..) call was cancelled, the previous population has been returned."
                    )
                    if showProgressBar:
                        bar.close()
                    return Fit(ms, weights, samples, dists)

            elapsed = time() - startTime

            newSamples = np.vstack(newSamples)

            # Combine the previous generation with this one.
            if recycling and t > 0:
                ms = np.concatenate([oldMs, newMs])
                weights = np.concatenate([oldWeights, newWeights])
                weights /= np.sum(weights)
                samples = np.concatenate([oldSamples, newSamples], axis=0)
                dists = np.concatenate([oldDists, newDists])
            else:
                ms = newMs
                weights = newWeights
                samples = newSamples
                dists = newDists

            # Store this generation to be recycled in the next one.
            if recycling:
                oldMs = newMs
                oldWeights = newWeights
                oldSamples = newSamples
                oldDists = newDists

            totalSimulationCost += numSims

            eps, kdes, ms, samples, dists, weights, ESS = prepare_next_population(
                numIters,
                popSize,
                recycling,
                verbose,
                saveIters,
                M,
                t,
                eps,
                ms,
                weights,
                samples,
                dists,
                numSims,
                elapsed,
            )

            if showProgressBar and t > 0:
                bar.update(1)

    if showProgressBar:
        bar.close()

    if verbose:
        update = f"Final population dists <= {dists.max():.2f}, ESS = {ESS}, total sims={totalSimulationCost}"
        if M > 1:
            modelPopulations = [np.sum(ms == m) for m in range(M)]
            modelWeights = [np.sum(weights[ms == m]) for m in range(M)]
            modelWeights = np.round(np.array(modelWeights) / np.sum(modelWeights), 2)
            update += f"\n\tmodel populations = {modelPopulations}, "
            update += f"model weights = {modelWeights}"

        print(update)

    return Fit(ms, weights, samples, dists)
