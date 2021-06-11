# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
import joblib
import numpy as np
import numpy.random as rnd
import pandas
import psutil
from numba import njit

@njit()
def numba_seed(seed):
    rnd.seed(seed)

from numpy.random import SeedSequence, default_rng
from time import time
from fastprogress.fastprogress import master_bar, progress_bar
from scipy.stats import gaussian_kde

from collections import namedtuple

Psi = namedtuple("Psi", ["name", "param"], defaults=["sum", 0.0])

Model = namedtuple(
    "Model", ["freq", "sev", "psi", "prior", "obsFreqs"], defaults=["ones", None, None, None, None]
)
Fit = namedtuple("Fit", ["models", "weights", "samples", "dists"])

# Currently it's difficult to get numba to compile a whole class, and in particular
# it can't handle the Prior classes. So, e.g. the 'SimpleIndependentUniformPrior' pulls out
# the key details from the IndependentUniformPrior class & turns it into a boring
# 'bag of data' (named tuple) which numba can handle/compile.
SimpleIndependentUniformPrior = namedtuple(
    "Prior", ["lower", "upper", "width", "normConst"]
)
SimpleKDE = namedtuple(
    "KDE", ["dataset", "weights", "d", "n", "inv_cov", "L", "log_det"]
)

from .simulate import simulate_claim_data, sample_discrete_dist, sim_multivariate_normal
from .weighted import quantile, systematic_resample
from .plot import _plot_results
from .wasserstein import wass_sumstats, wass_dist


def kde(data, weights, bw=np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)

def compute_psi(freqs, sevs, psi):
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

    claimsFake = simulate_claim_data(rg, T, model.freq, model.sev, theta) #, obsFreqs
    if type(model.freq) == str and model.freq.startswith("bivariate"):
        xFake1 = _compute_psi(claimsFake[0][0], claimsFake[0][1], model.psi.name, model.psi.param)
        xFake2 = _compute_psi(claimsFake[1][0], claimsFake[1][1], model.psi.name, model.psi.param)
        
        xFake = np.vstack([xFake1, xFake2]).T
    else:
        xFake = _compute_psi(claimsFake[0], claimsFake[1], model.psi.name, model.psi.param)

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

def _sample_one(
    seed, models, modelPrior, kdes, sumstats, distance, eps, numZerosData, ssData, T, systematic=False
):
    rg = default_rng(seed)
    numba_seed(seed)

    if systematic:
        modelGen = index_generator(rg, modelPrior)
        thetaGens = {}

    attempt = 0
    m = 0
    while True:
        attempt += 1

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

        claimsFake = simulate_claim_data(rg, T, model.freq, model.sev, theta)
        if type(model.freq) == str and model.freq.startswith("bivariate"):
            xFake1 = _compute_psi(claimsFake[0][0], claimsFake[0][1], model.psi.name, model.psi.param)
            if not num_zeros_match(numZerosData[0], xFake1):
                continue

            xFake2 = _compute_psi(claimsFake[1][0], claimsFake[1][1], model.psi.name, model.psi.param)
            if not num_zeros_match(numZerosData[1], xFake2):
                continue

            xFake = np.vstack([xFake1, xFake2]).T
        else:
            xFake = _compute_psi(claimsFake[0], claimsFake[1], model.psi.name, model.psi.param)

            if not num_zeros_match(numZerosData, xFake):
                continue

        dist = distance(ssData, sumstats(xFake))

        if dist < eps:

            thetaLogWeight = np.log(priorVal) - gaussian_kde_logpdf(
                theta, K.d, K.n, K.dataset, K.weights, K.inv_cov, K.log_det
            )
            weight = np.exp(thetaLogWeight)

            if weight > 0:
                break
            else:
                # In the super-unlikely event that the particle's weight
                # is so small that it is rounded down to 0, then throw
                # this away and keep sampling.
                continue

    return m, theta, weight, dist, attempt


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
    numProcs,
    mb,
    systematic=False,
):
    samples = []
    ms = np.zeros(n) * np.NaN
    weights = np.zeros(n) * np.NaN
    dists = np.zeros(n) * np.NaN
    numSims = 0

    seeds = (s.generate_state(1)[0] for s in sg.spawn(n))

    sample_first_iteration = joblib.delayed(_sample_one_first_iteration)
    sample = joblib.delayed(_sample_one)
    if t == 0:
        results = parallel(
            sample_first_iteration(
                seed, models, modelPrior, sumstats, distance, ssData, T,
            )
            for seed in progress_bar(seeds, parent=mb, total=n)
        )

    else:
        results = parallel(
            sample(
                seed,
                models,
                modelPrior,
                kdes,
                sumstats,
                distance,
                eps,
                numZerosData,
                ssData,
                T,
                systematic
            )
            for seed in progress_bar(seeds, parent=mb, total=n)
        )

    for i in range(n):
        m, theta, weight, dist, attempts = results[i]
        ms[i] = m
        samples.append(theta)
        weights[i] = weight
        dists[i] = dist
        numSims += attempts

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
                    np.sum(weights[ms == m]) ** 2
                    / np.sum(weights[ms == m] ** 2)
                )
            else:
                ESS.append(0)

    return np.round(ESS).astype(np.int)


def smc(
    numIters,
    popSize,
    obs,
    models,
    sumstats=wass_sumstats,
    distance=wass_dist,
    modelPrior=None,
    testName="",
    numProcs=None,
    quant=0.5,
    epsMin=0,
    saveIters=False,
    plotResults=False,
    thetaTrue=None,
    seed=1,
    timeout=30,
    verbose=False,
    matchZeros=True,
    systematic=False,
    recycling=True
):

    if type(models) == Model:
        models = [models]
        modelPrior = np.array([1.0])

    M = len(models)

    if not modelPrior:
        modelPrior = np.ones(M) / M

    if type(models[0].freq) == str and models[0].freq.startswith("bivariate"):
        T = obs.shape[0]
        numZerosData = (np.sum(obs[:,0] == 0), np.sum(obs[:,1] == 0)) if matchZeros else (-1, -1)
    else:
        T = len(obs)
        numZerosData = np.sum(obs == 0) if matchZeros else -1

        if type(obs) == pandas.core.series.Series:
            obs = obs.to_numpy()

    ssData = sumstats(obs)
    if not np.isscalar(ssData) and len(ssData) > 1:
        numSumStats = len(np.array(ssData).flatten())
    else:
        numSumStats = 1

    if not numProcs:
        numProcs = psutil.cpu_count(logical=False)

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
        if model.psi:
            newPsi = model.psi
        else:
            newPsi = Psi("severities")

        newModel = Model(model.freq, model.sev, newPsi, newPrior)
        newModels.append(newModel)
    
    models = tuple(newModels)

    sg = SeedSequence(seed)

    mb = master_bar(range(0, numIters + 1))

    if verbose:
        mb.write(
            f"Starting ABC-SMC with population size of {popSize} and sample size "
            + f"of {T} (~> {numSumStats}) on {numProcs} processes."
        )

    eps = np.inf
    kdes = None

    with joblib.Parallel(
        n_jobs=numProcs #, timeout=timeout
    ) as parallel:
        for t in mb:
            if eps <= epsMin:
                if verbose:
                    print("Stopping now due to exceeding epsilon target.")
                break

            startTime = time()

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
                numProcs,
                mb,
            )

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
                    
                    weights /= (1 - weights[ind])
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

            modelPopulations = [np.sum(ms == m) for m in range(M)]
            modelWeights = [np.sum(weights[ms == m]) for m in range(M)]
            modelWeights = np.round(np.array(modelWeights) / np.sum(modelWeights), 2)

            if verbose:
                ESS = calculate_ess(M, ms, weights)
                update = f"Finished iteration {t}, eps = {eps:.2f}, time = {np.round(elapsed)}s / {np.round(elapsed / 60, 1)}m, ESS = {ESS0} -> {ESS}, numSims = {numSims}"
                if M > 1:
                    update += f"\n\tmodel populations = {modelPopulations}, model weights = {modelWeights}"
                mb.write(update)

            if saveIters:
                np.savetxt(f"smc-samples-{t:02}.txt", samples)
                np.savetxt(f"smc-weights-{t:02}.txt", weights)
                np.savetxt(f"smc-dists-{t:02}.txt", dists)

            if plotResults:
                filename = f"{testName}SMC-{t:02}.pdf" if saveIters else ""
                _plot_results(
                    samples,
                    weights,
                    model.prior,
                    thetaTrue=thetaTrue,
                    filename=filename,
                )

    if verbose:
        update = f"Final population dists <= {dists.max():.2f}, ESS = {ESS}"
        if M > 1:
            modelPopulations = [np.sum(ms == m) for m in range(M)]
            modelWeights = [np.sum(weights[ms == m]) for m in range(M)]
            modelWeights = np.round(np.array(modelWeights) / np.sum(modelWeights), 2)
            update += f"\n\tmodel populations = {modelPopulations}, model weights = {modelWeights}"

        print(update)

    return Fit(ms, weights, samples, dists)
