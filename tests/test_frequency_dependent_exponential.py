import approxbayescomp as abc
import numpy as np
import numpy.random as rnd
from numba import njit

numIters = 5
numItersData = 10
popSize = 100

# Frequency-Loss Model
λ = 4
β = 2
δ = 0.2
θ_True = λ, β, δ

sev = "frequency dependent exponential"
freq = "poisson"
psi = abc.Psi("sum")  # Aggregation process

# Simulate some data to fit
T = 50

rg = rnd.default_rng(123)
freqs, sevs = abc.simulate_claim_data(rg, T, freq, sev, θ_True)
xData = abc.compute_psi(freqs, sevs, psi)

# Specify model to fit
params = ("λ", "β", "δ")
prior = abc.IndependentUniformPrior([(0, 10), (0, 20), (-1, 1)], params)
model = abc.Model("poisson", "frequency dependent exponential", psi, prior)
epsMin = 6


def test_simulation_size():
    assert len(xData) == T


def test_full_model():
    fit = abc.smc(numIters, popSize, xData, model, epsMin=epsMin, verbose=True, seed=1)
    assert np.max(fit.dists) < epsMin


def test_partially_observed_model():
    # Try fitting the same model but with the frequencies observed
    params = ("β", "δ")
    prior = abc.IndependentUniformPrior([(0, 20), (-1, 1)], params)
    model = abc.Model(freqs, "frequency dependent exponential", psi, prior)

    epsMin = 3
    fit = abc.smc(
        numItersData, popSize, xData, model, epsMin=epsMin, verbose=True, seed=1
    )
    assert np.max(fit.dists) < epsMin


def simulate_poisson_exponential_sums_new_rng(rg, theta, T):
    lam = theta[0]
    thetaSev = theta[1:]
    freqs = rg.poisson(lam, size=T)

    scale, cor = thetaSev

    aggClaims = np.empty(T, np.float64)
    for t in range(T):
        R = freqs[t]
        claims = scale * np.exp(cor * R) * rg.exponential(size=R)
        aggClaims[t] = np.sum(claims)

    return aggClaims


@njit()
def simulate_poisson_exponential_sums_old_rng(theta, T):
    lam = theta[0]
    thetaSev = theta[1:]
    freqs = rnd.poisson(lam, size=T)

    scale, cor = thetaSev

    aggClaims = np.empty(T, np.float64)
    for t in range(T):
        claims = np.empty(freqs[t], np.float64)
        for i in range(freqs[t]):
            claims[i] = scale * np.exp(cor * freqs[t]) * rnd.exponential()
        aggClaims[t] = np.sum(claims)

    return aggClaims


def test_simulator_with_new_rng():
    model = abc.SimulationModel(
        lambda rg, theta: simulate_poisson_exponential_sums_new_rng(
            rg, theta, len(xData)
        ),
        prior,
    )

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        simulatorUsesOldNumpyRNG=False,
    )
    assert np.max(fit.dists) < epsMin


def test_simulator_with_old_rng():
    model = abc.SimulationModel(
        lambda theta: simulate_poisson_exponential_sums_old_rng(theta, len(xData)),
        prior,
    )
    fit = abc.smc(numIters, popSize, xData, model, epsMin=epsMin, verbose=True, seed=1)
    assert np.max(fit.dists) < epsMin


def test_multiple_processes():
    numProcs = 4

    # Check that both strictPopulationSize=True and False work
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        strictPopulationSize=True,
    )
    assert np.max(fit.dists) < epsMin

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        strictPopulationSize=False,
    )
    assert np.max(fit.dists) < epsMin


if __name__ == "__main__":
    test_full_model()
    test_simulator_with_new_rng()
    test_simulator_with_old_rng()
    test_partially_observed_model()
    test_multiple_processes()
