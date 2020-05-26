# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
import joblib
import numpy as np
from numba import jit
from numpy.random import Generator, PCG64, SeedSequence
from time import time
from fastprogress.fastprogress import master_bar, progress_bar
from scipy.stats import gaussian_kde

from collections import namedtuple

Psi = namedtuple("Psi", ["name", "param"], defaults=["sum", 0.0])
Model = namedtuple(
    "Model", ["freq", "sev", "psi", "prior"], defaults=[None, None, None, None]
)
Fit = namedtuple("Fit", ["models", "weights", "samples", "dists"])

from .simulate import simulate_claim_data
from .weighted import quantile
from .plot import _plot_results
from .wasserstein_adaptive import wass_adap_sumstats, wass_adap_dist


def kde(data, weights, bw=np.sqrt(2)):
    return gaussian_kde(data.T, weights=weights, bw_method=bw)


@jit(nopython=True)
def compute_psi(freqs, sevs, psi):
    xs = -np.ones(len(freqs))
    i = 0

    if psi.name == "sum":
        for r, n in enumerate(freqs):
            xs[r] = np.sum(sevs[i : i + n])
            i += n

    elif psi.name == "GSL":
        for r, n in enumerate(freqs):
            xs[r] = np.maximum(np.sum(sevs[i : i + n]) - psi.param, 0)
            i += n

    elif psi.name == "ISL":
        for r, n in enumerate(freqs):
            xs[r] = np.sum(np.maximum(sevs[i : i + n] - psi.param, 0))
            i += n

    else:
        raise Exception("Psi function unsupported")

    return xs


def sample_one(
    rg, t, models, modelPrior, modelDist, thetaDists, sumstats, distance, eps, ssData, T
):
    M = len(models)

    if t == 0:
        # On the first iteration of SMC we sample from the prior
        # and accept everthing, so the code is a bit simpler.
        m = rg.choice(M, p=modelPrior)
        model = models[m]
        theta = model.prior.sample(rg)
        freqs, sevs = simulate_claim_data(rg, T, model.freq, model.sev, theta)
        if model.psi:
            xFake = compute_psi(freqs, sevs, model.psi)
        else:
            xFake = sevs
        ssFake = sumstats(xFake)
        dist = distance(ssData, ssFake, t)
        return m, theta, 1.0, dist, 1

    #     m = rg.choice(M, p=modelDist)
    #     model = models[m]
    #     thetaDist = thetaDists[m]

    #     if thetaDist is None:
    #         raise RuntimeError()

    attempt = 0
    while True:
        attempt += 1

        m = rg.choice(M, p=modelDist)
        model = models[m]
        thetaDist = thetaDists[m]

        if thetaDist is None:
            raise RuntimeError()

        theta = thetaDist.resample(size=1, seed=rg).reshape(-1)

        priorVal = model.prior.pdf(theta)
        if priorVal <= 0:
            continue

        freqs, sevs = simulate_claim_data(rg, T, model.freq, model.sev, theta)
        if model.psi:
            xFake = compute_psi(freqs, sevs, model.psi)
        else:
            xFake = sevs

        ssFake = sumstats(xFake)
        dist = distance(ssData, ssFake, t)
        if dist < eps:
            break

    thetaLogWeight = np.log(priorVal) - np.log(thetaDist(theta))
    modelLogWeight = np.log(modelPrior[m]) - np.log(modelDist[m])
    weight = np.exp(thetaLogWeight + modelLogWeight)

    return m, theta, weight, dist, attempt


# Sample a population of particles
def sample_population(
    sg,
    t,
    parallel,
    models,
    modelPrior,
    modelDist,
    thetaDists,
    sumstats,
    distance,
    eps,
    n,
    ssData,
    T,
    numProcs,
    mb,
):
    samples = []
    ms = np.zeros(n) * np.NaN
    weights = np.zeros(n) * np.NaN
    dists = np.zeros(n) * np.NaN
    numSims = 0

    rgs = (Generator(PCG64(s)) for s in sg.spawn(n))

    sample = joblib.delayed(sample_one)

    results = parallel(
        sample(
            rg,
            t,
            models,
            modelPrior,
            modelDist,
            thetaDists,
            sumstats,
            distance,
            eps,
            ssData,
            T,
        )
        for rg in progress_bar(rgs, parent=mb, total=n)
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
        d[m] = np.array(d[m])

    return d


def fit_all_kdes(ms, samples, weights, M):
    thetaDists = []

    samplesGrouped = group_samples_by_model(ms, samples, M)

    for m in range(M):
        samples_m = samplesGrouped[m]
        K = None
        if len(samples_m) >= 10:
            try:
                K = kde(samples_m, weights[ms == m])
            except np.linalg.LinAlgError:
                pass

        thetaDists.append(K)

    return thetaDists


def smc(
    numIters,
    popSize,
    obs,
    models,
    sumstats=wass_adap_sumstats,
    distance=wass_adap_dist,
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
):

    if type(models) == Model:
        models = [models]
        modelPrior = [1.0]

    M = len(models)

    if not modelPrior:
        modelPrior = np.ones(M) / M

    T = len(obs)
    ssData = sumstats(obs)
    numSumStats = sum(getattr(part, "__len__", lambda: 1)() for part in sumstats(obs))

    if not numProcs:
        numProcs = max(joblib.cpu_count() // 2, 1)

    sg = SeedSequence(seed)

    simulationCost = 0
    mb = master_bar(range(0, numIters + 1))

    if verbose:
        mb.write(
            f"Starting ABC-SMC with population size of {popSize} and sample size "
            + f"of {T} (~> {numSumStats}) on {numProcs} processes."
        )

    eps = np.inf
    numsims = 0
    modelDist = thetaDists = timeout_t = None

    for t in mb:
        if eps <= epsMin:
            if verbose:
                print("Stopping now due to exceeding epsilon target.")
            break

        with joblib.Parallel(n_jobs=numProcs, timeout=timeout_t) as parallel:
            try:
                ms, weights, samples, dists, numSims = sample_population(
                    sg,
                    t,
                    parallel,
                    models,
                    modelPrior,
                    modelDist,
                    thetaDists,
                    sumstats,
                    distance,
                    eps,
                    popSize,
                    ssData,
                    T,
                    numProcs,
                    mb,
                )
            except Exception:
                # If crashing on the first iteration then we really want to know about it.
                # For later iterations, the code probably just timed out.
                if t > 0:
                    if verbose:
                        print("Stopping now due to timeout condition.")
                        print(
                            "Discarding this population and returning the "
                            + "previous one."
                        )
                    return Fit(ms, weights, samples, dists)
                else:
                    raise

        eps = quantile(dists, weights, quant)

        thetaDists = fit_all_kdes(ms, samples, weights, M)

        # Ignore if < 10 particles exist from a model.
        counts = [np.sum(ms == m) for m in range(M)]
        modelWeights = np.array(
            [np.sum(weights[ms == m]) if counts[m] > 10 else 0 for m in range(M)]
        )
        modelDist = modelWeights / np.sum(modelWeights)

        timeout_t = timeout

        if verbose:
            # Calculate effective sample size for each model
            if M == 1:
                ESS = np.round(1 / np.sum(weights ** 2), 2)
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
                ESS = np.round(ESS)

            update = f"Finished iteration {t}, eps = {eps:.2f}, ESS = {ESS}, numSims = {numSims}"
            update += (
                f", models = {[np.mean(ms == m) for m in range(M)]}" if M > 1 else ""
            )
            mb.write(update)

        if saveIters:
            np.savetxt(f"smc-samples-{t:02}.txt", samples)
            np.savetxt(f"smc-weights-{t:02}.txt", weights)
            np.savetxt(f"smc-dists-{t:02}.txt", dists)

        if plotResults:
            filename = f"{testName}SMC-{t:02}.pdf" if saveIters else ""
            _plot_results(
                samples, weights, model.prior, thetaTrue=thetaTrue, filename=filename
            )

    if verbose:
        update = f"Final population dists <= {dists.max():.2f}, ESS = {ESS}"
        update += f", models = {[np.mean(ms == m) for m in range(M)]}" if M > 1 else ""
        print(update)

    return Fit(ms, weights, samples, dists)


# =============================================================================
# def ABC_AR(n, x_data, prior, distance, T, freq, sev, type_function,
#             par_function, name="", numProcs=1, quant=0.5, thetaTrue = None):
#     return ABC_SMC(2, n, x_data, prior, distance, T, freq, sev, type_function,
#             par_function, name, numProcs, quant,
#             False, thetaTrue)
# =============================================================================
