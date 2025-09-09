import approxbayescomp as abc
import numpy as np
import numpy.random as rnd

# Load data to fit (modify this line to load real observations!)
obsData = [1.0, 2.0, 3.0]

# Specify our prior beliefs over (lambda, mu, sigma).
prior = abc.IndependentUniformPrior([(0, 100), (-5, 5), (0, 3)])


# Write a function to simulate from the data-generating process.
def simulate_aggregate_claims(theta, T):
    """
    Generate T observations from the model specified by theta.
    """
    lam, mu, sigma = theta
    freqs = rnd.poisson(lam, size=T)
    aggClaims = np.empty(T, np.float64)
    for t in range(T):
        aggClaims[t] = np.sum(rnd.lognormal(mu, sigma, size=freqs[t]))
    return aggClaims


# Fit the model to the data using ABC
def model(theta):
    return simulate_aggregate_claims(theta, len(obsData))


numIters = 6  # The number of SMC iterations to perform
popSize = 250  # The population size of the SMC method

fit = abc.smc(numIters, popSize, obsData, model, prior)
