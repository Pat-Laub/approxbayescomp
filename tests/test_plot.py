import pytest
import numpy as np
import matplotlib.pyplot as plt
import approxbayescomp as abc
from approxbayescomp.plot import plot_posteriors


def exponential_simulator(theta):
    """Simulate iid exponential random variables given the parameter theta."""
    return np.random.exponential(scale=theta, size=100)


def test_plot_posteriors_exponential():
    """Test plotting the posterior for a single parameter fitted to exponential data."""
    # Simulate observed data
    true_mean = 50
    obs_data = np.random.exponential(scale=true_mean, size=100)

    # Define prior and fit the model
    prior = abc.IndependentUniformPrior([(0, 100)])  # Uniform prior on [0, 100]
    num_iters = 2  # Reduced iterations for testing
    pop_size = 50  # Smaller population size for testing

    fit = abc.smc(num_iters, pop_size, obs_data, exponential_simulator, prior)

    # Plot the posterior
    fig, ax = plt.subplots()
    try:
        plot_posteriors(fit=fit, prior=prior, subtitles=["Exponential Mean"], figsize=(5, 2), dpi=100)
    except Exception as e:
        pytest.fail(f"plot_posteriors failed: {e}")
    finally:
        plt.close(fig)
