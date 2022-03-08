import approxbayescomp as abc
import numpy as np
import numpy.random as rnd
import pandas as pd
import scipy.stats as st

numIters = 6
popSize = 100

# Frequency-Loss Model
freq = "bivariate poisson"
sev = "exponential"
σ = 0.2
w1 = 15
w2 = 5
m1 = 10
m2 = 70
trueTheta = (σ, w1, w2, m1, m2)

# Setting the time horizon
T = 50

# Simulating the claim data
rg = rnd.default_rng(1234)
claimsData = abc.simulate_claim_data(rg, T, freq, sev, trueTheta)

# Simulating the observed data
psi = abc.Psi("GSL", 25)

xData1 = abc.compute_psi(claimsData[0][0], claimsData[0][1], psi)
xData2 = abc.compute_psi(claimsData[1][0], claimsData[1][1], psi)

xData = np.vstack([xData1, xData2]).T

print(f"Number of zeros in the data: {np.sum(xData == 0)}")

# Specify model to fit
model = abc.Model(freq, sev, psi)
prior = abc.IndependentUniformPrior([(0, 2), (0, 50), (0, 50), (0, 100), (0, 100)])

epsMin = 200


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


def test_full_model():
    print("\ntest_full_model()\n")
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_eps_min():
    # Check that SMC will stop early after reaching the epsilon target.
    print("\ntest_eps_min()\n")
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        epsMin=epsMin,
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_match_zeros():
    # Check that matchZeros=True is working.
    print("\ntest_match_zeros()\n")
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        matchZeros=True,
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
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
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_strict_population_size():
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
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
        strictPopulationSize=True,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_pandas_input():
    print("\ntest_pandas_input()\n")
    df = pd.DataFrame({"x": xData[:, 0], "y": xData[:, 1]})

    fit = abc.smc(
        numIters,
        popSize,
        df,
        model,
        prior,
        epsMin=epsMin,
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


def test_nonuniform_prior():
    print("\ntest_nonuniform_prior()\n")
    means = (1, 25, 25, 50, 50)
    marginals = [st.expon(scale=mean) for mean in means]
    prior = abc.IndependentPrior(marginals)
    fit = abc.smc(
        numIters,
        popSize,
        xData,
        model,
        prior,
        distance=abc.wasserstein2D,
        verbose=True,
        seed=1,
    )
    check_fit(fit, popSize, epsMin, prior.dim)


if __name__ == "__main__":
    test_full_model()
    test_eps_min()
    test_match_zeros()
    test_multiple_processes()
    test_strict_population_size()
    test_pandas_input()
    test_nonuniform_prior()
