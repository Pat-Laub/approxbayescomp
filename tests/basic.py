import numpy as np
import numpy.random as rnd

import approxbayescomp as abc

numIters = 5
numItersData = 10
popSize = 500
epsMin = 1


# Frequency-Loss Model
λ, β, δ = 4, 2, 0.2
θ_True = λ, β, δ

sev = "frequency dependent exponential"
freq = "poisson"
psi = abc.Psi("sum")  # Aggregation process

# Simulate some data to fit
sample_sizes = [50, 250]
T = sample_sizes[-1]

rg = rnd.default_rng(123)
freqs, sevs = abc.simulate_claim_data(rg, T, freq, sev, θ_True)
xData = abc.compute_psi(freqs, sevs, psi)

# Specify model to fit
params = ("λ", "β", "δ")
prior = abc.IndependentUniformPrior([(0, 10), (0, 20), (-1, 1)], params)
model = abc.Model("poisson", "frequency dependent exponential", psi, prior)

abcFits = []

for ss in sample_sizes:
    fit = abc.smc(numIters, popSize, xData[:ss], model, epsMin=epsMin, verbose=True)
    abcFits.append(fit)

params = ("β", "δ")
prior = abc.IndependentUniformPrior([(0, 20), (-1, 1)], params)

for ss in sample_sizes:
    nDataSS = freqs[:ss]
    model = abc.Model(nDataSS, "frequency dependent exponential", psi, prior)
    fit = abc.smc(numItersData, popSize, xData[:ss], model, epsMin=epsMin, verbose=True)
    abcFits.append(fit)
