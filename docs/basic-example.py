import approxbayescomp as abc
import numpy as np

# Load data to fit (modify this line to load real observations!)
# This toy example assumes that we observed X_1 = $1.0, X_2 = $2.0,
# and X_3 = $3.0 in aggregate claims over three months.
obsData = np.array([1.0, 2.0, 3.0])

# Frequency-Loss Model
freq = "poisson"
sev = "lognormal"
psi = abc.Psi("sum")  # Aggregation process

prior = abc.IndependentUniformPrior([(0, 100), (-5, 5), (0, 3)])
model = abc.Model(freq, sev, psi, prior)

# Fit the model to the data using ABC
numIters = 8  # The number of SMC iterations to perform
popSize = 1000  # The population size of the SMC method

fit = abc.smc(numIters, popSize, obsData, model)
