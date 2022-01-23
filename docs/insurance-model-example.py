import approxbayescomp as abc

# Load data to fit (modify this line to load real observations!)
obsData = [1.0, 2.0, 3.0]

# Specify our prior beliefs over (lambda, mu, sigma).
prior = abc.IndependentUniformPrior([(0, 100), (-5, 5), (0, 3)])

# Fit the model to the data using ABC
model = abc.Model("poisson", "lognormal", abc.Psi("sum"), prior)
numIters = 6  # The number of SMC iterations to perform
popSize = 250  # The population size of the SMC method

fit = abc.smc(numIters, popSize, obsData, model)
