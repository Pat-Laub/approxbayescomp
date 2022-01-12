import numpy as np
import numpy.random as rnd

import approxbayescomp as abc

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


def test_simulation_size():
    assert len(xData) == T


def test_full_model():
    # Specify model to fit
    params = ("λ", "β", "δ")
    prior = abc.IndependentUniformPrior([(0, 10), (0, 20), (-1, 1)], params)
    model = abc.Model("poisson", "frequency dependent exponential", psi, prior)

    epsMin = 6
    fit = abc.smc(numIters, popSize, xData, model, epsMin=epsMin, verbose=True)
    assert np.max(fit.dists) < epsMin


def test_partially_observed_model():
    # Try fitting the same model but with the frequencies observed
    params = ("β", "δ")
    prior = abc.IndependentUniformPrior([(0, 20), (-1, 1)], params)
    model = abc.Model(freqs, "frequency dependent exponential", psi, prior)

    epsMin = 3
    fit = abc.smc(numItersData, popSize, xData, model, epsMin=epsMin, verbose=True)
    assert np.max(fit.dists) < epsMin
