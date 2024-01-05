import approxbayescomp as abc
import numpy as np
import numpy.random as rnd
import pandas as pd
import scipy.stats as st
from dtw import dtw_distance
from numba import float64, int64, njit


def simulate_dependent_poisson_exponential_sums_new_rng(rg, theta, T):
    lam = theta[0]
    thetaSev = theta[1:]
    freqs = rg.poisson(lam, size=T)

    scale, cor = thetaSev

    aggClaims = np.empty(T, np.float64)
    for t in range(T):
        claims = scale * np.exp(cor * freqs[t]) * rg.exponential(size=freqs[t])
        aggClaims[t] = np.sum(claims)

    return aggClaims


@njit(float64[:](float64[:], int64), nogil=True)
def simulate_dependent_poisson_exponential_sums_old_rng(theta, T):
    lam = theta[0]
    thetaSev = theta[1:]
    freqs = rnd.poisson(lam, size=T)  # TODO: Why does the size argument JIT here?
    scale, cor = thetaSev

    aggClaims = np.zeros(T)
    for t in range(T):
        for i in range(freqs[t]):
            aggClaims[t] += scale * np.exp(cor * freqs[t]) * rnd.exponential()

    return aggClaims


numIters = 5
numItersData = 10
popSize = 100

# Frequency-Loss Model
λ = 4
β = 2
δ = 0.2
trueTheta = λ, β, δ

sev = "frequency dependent exponential"
freq = "poisson"
psi = abc.Psi("sum")  # Aggregation process

# Simulate some data to fit
T = 50

rg = rnd.default_rng(123)
freqs, sevs = abc.simulate_claim_data(rg, T, freq, sev, trueTheta)
xData = abc.compute_psi(freqs, sevs, psi)

print(f"Number of zeros in the data: {np.sum(xData == 0)}")

# Specify model to fit
params = ("λ", "β", "δ")
prior = abc.IndependentUniformPrior([(0, 10), (0, 20), (-1, 1)], params)
model = abc.Model("poisson", "frequency dependent exponential", psi)
epsMin = 6


def check_fit(fit, popSize, epsMin, numTheta):
    assert fit.models.shape == (popSize,)
    assert fit.weights.shape == (popSize,)
    assert fit.samples.shape == (popSize, numTheta)
    assert fit.dists.shape == (popSize,)
    assert np.all(fit.models == 0)
    assert np.abs(np.sum(fit.weights) - 1) < 1e-5
    assert np.max(fit.dists) < epsMin


def test_simulation_size():
    assert len(xData) == T


def test_partially_observed_model():
    # Try fitting the same model but with the frequencies observed
    print("\ntest_partially_observed_model()\n")
    params = ("β", "δ")
    prior = abc.IndependentUniformPrior([(0, 20), (-1, 1)], params)
    model = abc.Model(freqs, "frequency dependent exponential", psi)

    epsMin = 3
    fit = abc.smc(
        numItersData, popSize, xData, model, prior, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_full_model():
    print("\ntest_full_model()\n")
    fit = abc.smc(numIters, popSize, xData, model, prior, verbose=True, seed=1)
    check_fit(fit, popSize, epsMin, prior.dim)


def test_eps_min():
    # Check that SMC will stop early after reaching the epsilon target.
    print("\ntest_eps_min()\n")
    fit = abc.smc(
        numIters, popSize, xData, model, prior, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_match_zeros():
    # Check that matchZeros=True is working.
    print("\ntest_match_zeros()\n")
    fit = abc.smc(
        numIters, popSize, xData, model, prior, matchZeros=True, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_systematic_sampling():
    # Check that the systematic sampling option is working.
    print("\ntest_systematic_sampling()\n")
    fit = abc.smc(
        numIters, popSize, xData, model, prior, systematic=True, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_simulator_with_new_rng():
    print("\ntest_simulator_with_new_rng()\n")

    def model(rg, theta):
        return simulate_dependent_poisson_exponential_sums_new_rng(
            rg, theta, len(xData)
        )

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        simulatorUsesOldNumpyRNG=False,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_simulator_with_old_rng():
    print("\ntest_simulator_with_old_rng()\n")

    def model(theta):
        return simulate_dependent_poisson_exponential_sums_old_rng(theta, len(xData))

    fit = abc.smc(
        numIters, popSize, xData, model, prior, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_multiple_processes():
    print("\ntest_multiple_processes()\n")
    numProcs = 4

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_strict_population_size():
    # Check that strictPopulationSize=True works
    print("\ntest_strict_population_size()\n")
    numProcs = 4
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        strictPopulationSize=True,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_list_input():
    print("\ntest_list_input()\n")
    fit = abc.smc(numIters, popSize, list(xData), model, prior, verbose=True, seed=1)
    check_fit(fit, popSize, epsMin, prior.dim)


def test_pandas_input():
    print("\ntest_pandas_input()\n")
    df = pd.DataFrame({"x": xData})

    fit = abc.smc(
        numIters, popSize, df["x"], model, prior, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_dynamic_time_warping():
    print("\ntest_dynamic_time_warping()\n")

    def model(theta):
        return simulate_dependent_poisson_exponential_sums_old_rng(theta, len(xData))

    epsMin = 150
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        distance=dtw_distance,
        epsMin=epsMin,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_nonuniform_prior():
    print("\ntest_nonuniform_prior()\n")

    locs = (0, 0, -1)
    means = (5, 10, 1)
    marginals = [st.expon(loc=loc, scale=mean) for loc, mean in zip(locs, means)]
    prior = abc.IndependentPrior(marginals)

    fit = abc.smc(numIters, popSize, xData, model, prior, verbose=True, seed=1)
    check_fit(fit, popSize, epsMin, prior.dim)


if __name__ == "__main__":
    test_partially_observed_model()
    test_full_model()
    test_eps_min()
    test_match_zeros()
    test_systematic_sampling()
    test_simulator_with_new_rng()
    test_simulator_with_old_rng()
    test_multiple_processes()
    test_strict_population_size()
    test_list_input()
    test_pandas_input()
    test_dynamic_time_warping()
    test_nonuniform_prior()
