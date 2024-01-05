import approxbayescomp as abc
import numpy as np
import numpy.random as rnd
from dtw import dtw_distance
from numba import float64, int64, njit


def simulate_poisson_exponential_maxs_new_rng(rg, theta, T):
    lam = theta[0]
    thetaSev = theta[1:]
    freqs = rg.poisson(lam, size=T)
    scale = thetaSev
    maxClaims = np.empty(T, np.float64)
    for t in range(T):
        claims = scale * rg.exponential(size=freqs[t])
        if freqs[t] > 0:
            maxClaims[t] = np.max(claims)
        else:
            maxClaims[t] = 0
    return maxClaims


def simulate_geometric_exponential_maxs_new_rng(rg, theta, T):
    p = theta[0]
    thetaSev = theta[1:]
    freqs = rg.geometric(1 - p, size=T)
    scale = thetaSev
    maxClaims = np.empty(T, np.float64)
    for t in range(T):
        claims = scale * rg.exponential(size=freqs[t])
        if freqs[t] > 0:
            maxClaims[t] = np.max(claims)
        else:
            maxClaims[t] = 0
    return maxClaims


@njit(float64[:](float64[:], int64), nogil=True)
def simulate_poisson_exponential_maxs_old_rng(theta, T):
    lam = theta[0]
    scale = theta[1]

    maxClaims = np.zeros(T)
    for t in range(T):
        numClaims = rnd.poisson(lam)
        for i in range(numClaims):
            claim = scale * rnd.exponential()
            if claim > maxClaims[t]:
                maxClaims[t] = claim

    return maxClaims


@njit(float64[:](float64[:], int64), nogil=True)
def simulate_geometric_exponential_maxs_old_rng(theta, T):
    p = theta[0]
    scale = theta[1]

    maxClaims = np.zeros(T)
    for t in range(T):
        numClaims = rnd.geometric(1 - p)
        for i in range(numClaims):
            claim = scale * rnd.exponential()
            if claim > maxClaims[t]:
                maxClaims[t] = claim

    return maxClaims


numIters = 5
numItersData = 10
popSize = 100

# Frequency-Loss Model
λ = 4
μ = 0.2
trueTheta = λ, μ

freq = "poisson"
sev = "exponential"
psi = abc.Psi("max")

# Simulate some data to fit
T = 50

rg = rnd.default_rng(123)
freqs, sevs = abc.simulate_claim_data(rg, T, freq, sev, trueTheta)
xData = abc.compute_psi(freqs, sevs, psi)

# Specify models to fit
model1 = abc.Model("poisson", "exponential", psi)
model2 = abc.Model("geometric", "exponential", psi)

prior1 = abc.IndependentUniformPrior([(0, 10), (0, 20)], ("λ", "μ"))
prior2 = abc.IndependentUniformPrior([(0, 1), (0, 20)], ("p", "μ"))

# TODO: Allow differing numTheta for each model.

models = (model1, model2)
priors = (prior1, prior2)

epsMin = 0.5


def check_fit(fit, popSize, epsMin, numTheta=2):
    assert fit.models.shape == (popSize,)
    assert fit.weights.shape == (popSize,)
    assert fit.samples.shape == (popSize, numTheta)
    assert fit.dists.shape == (popSize,)
    assert np.all((fit.models == 0) + (fit.models == 1))
    assert np.abs(np.sum(fit.weights) - 1) < 1e-5
    assert np.max(fit.dists) < epsMin


def test_simulation_size():
    assert len(xData) == T


def test_partially_observed_model():
    # Try fitting the same model but with the frequencies observed
    print("\ntest_partially_observed_model()\n")
    model = abc.Model(freqs, "exponential", psi)
    prior = abc.IndependentUniformPrior([(0, 20)], ("μ"))

    epsMin = 0.1
    fit = abc.smc(
        numItersData, popSize, xData, model, prior, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin, 1)


def test_full_model():
    print("\ntest_full_model()\n")
    fit = abc.smc(numIters, popSize, xData, models, priors, verbose=True, seed=1)
    check_fit(fit, popSize, epsMin)


def test_eps_min():
    # Check that it will stop after reaching the epsilon target.
    print("\ntest_eps_min()\n")
    fit = abc.smc(
        numIters, popSize, xData, models, priors, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin)


def test_match_zeros():
    print("\ntest_match_zeros()\n")
    fit = abc.smc(
        numIters, popSize, xData, models, priors, matchZeros=True, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin)


def test_simulator_with_new_rng():
    print("\ntest_simulator_with_new_rng()\n")

    def model1(rg, theta):
        return simulate_poisson_exponential_maxs_new_rng(rg, theta, len(xData))

    def model2(rg, theta):
        return simulate_geometric_exponential_maxs_new_rng(rg, theta, len(xData))

    models = (model1, model2)

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        models,
        priors,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        simulatorUsesOldNumpyRNG=False,
    )
    check_fit(fit, popSize, epsMin)


def test_simulator_with_old_rng():
    print("\ntest_simulator_with_old_rng()\n")

    def model1(theta):
        return simulate_poisson_exponential_maxs_old_rng(theta, len(xData))

    def model2(theta):
        return simulate_geometric_exponential_maxs_old_rng(theta, len(xData))

    models = (model1, model2)

    fit = abc.smc(
        numIters, popSize, xData, models, priors, epsMin=epsMin, verbose=True, seed=1
    )
    check_fit(fit, popSize, epsMin)


def test_multiple_processes():
    print("\ntest_multiple_processes()\n")
    numProcs = 4

    fit = abc.smc(
        numIters,
        popSize,
        xData,
        models,
        priors,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin)


def test_strict_population_size():
    # Check that strictPopulationSize=True works
    print("\ntest_strict_population_size()\n")
    numProcs = 4
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        models,
        priors,
        numProcs=numProcs,
        epsMin=epsMin,
        verbose=True,
        seed=1,
        strictPopulationSize=True,
    )
    check_fit(fit, popSize, epsMin)


def test_dynamic_time_warping():
    print("\ntest_dynamic_time_warping()\n")

    def model1(theta):
        return simulate_poisson_exponential_maxs_old_rng(theta, len(xData))

    def model2(theta):
        return simulate_geometric_exponential_maxs_old_rng(theta, len(xData))

    models = (model1, model2)

    epsMin = 5
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        models,
        priors,
        distance=dtw_distance,
        epsMin=epsMin,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin)


if __name__ == "__main__":
    test_partially_observed_model()
    test_full_model()
    test_eps_min()
    test_match_zeros()
    test_simulator_with_new_rng()
    test_simulator_with_old_rng()
    test_multiple_processes()
    test_strict_population_size()
    test_dynamic_time_warping()
