import approxbayescomp as abc
import numpy as np

# Load data to fit (modify this line to load real observations!)
obsData = [1.0, 2.0, 3.0]

# Specify our prior beliefs over (lambda, mu, sigma).
prior = abc.IndependentUniformPrior([(0, 100), (-5, 5), (0, 3)])


# Write a function to simulate from the data-generating process.
def simulate_aggregate_claims(rg, T, theta):
    """
    Generate 'T' observations from the model specified by 'theta'
    using the supplied random number generator 'rg'.
    """
    lam, mu, sigma = theta
    freqs = rg.poisson(lam, size=T)
    aggClaims = np.empty(T, np.float64)
    for t in range(T):
        aggClaims[t] = np.sum(rg.lognormal(mu, sigma, size=freqs[t]))
    return aggClaims


# Fit the model to the data using ABC
model = abc.SimulationModel(simulate_aggregate_claims, prior)
numIters = 8  # The number of SMC iterations to perform
popSize = 1000  # The population size of the SMC method

fit = abc.smc(numIters, popSize, obsData, model)