# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
import numpy as np
import numpy.random as rnd
import scipy.stats as st
from numba import njit


@njit()
def uniform_prior_pdf(theta, lower, upper, normConst):
    for i in range(len(theta)):
        if theta[i] <= lower[i] or theta[i] >= upper[i]:
            return 0
    return normConst


class IndependentUniformPrior(object):
    def __init__(self, bounds, names=None):
        self.dim = len(bounds)
        self.lower = np.array([bound[0] for bound in bounds], dtype=np.float64)
        self.upper = np.array([bound[1] for bound in bounds], dtype=np.float64)
        self.widths = self.upper - self.lower
        self.names = names
        self.normConst = 1.0 / np.prod(self.widths)
        self.marginals = [
            st.uniform(self.lower[i], self.widths[i]) for i in range(self.dim)
        ]

    def pdf(self, theta):
        return uniform_prior_pdf(theta, self.lower, self.upper, self.normConst)

    def sample(self, rg):
        return self.lower + self.widths * rg.uniform(size=self.dim)


class IndependentPrior(object):
    def __init__(self, marginals, names=None, types=None):
        self.marginals = marginals
        if types:
            self.types = types
        else:
            self.types = ["continuous" for _ in marginals]
        self.names = names
        self.rg = None

    def pdf(self, theta):
        list_lik_prior = []
        for i, theta_i in enumerate(theta.reshape(-1)):
            if self.types[i] == "discrete":
                list_lik_prior.append(self.marginals[i].pmf(theta_i))
            else:
                list_lik_prior.append(self.marginals[i].pdf(theta_i))
        return np.prod(list_lik_prior)

    def sample(self, size=None, seed=None):
        if seed is None:
            self.rg = np.random
        elif type(seed) == int:
            self.rg = np.random.default_rng(seed)
        else:
            self.rg = seed

        return np.array(
            [prior.rvs(random_state=self.rg) for prior in self.marginals]
        ).reshape(-1)


# Function that set up the prior distribution for the parameters depending on
# the claim frequency and severity distributions
def default_prior(freq, sev, hyper_theta_freq, hyper_theta_sev):
    # freq = claim frequency distribution, choose from ("poisson", "binomial",
    # "negative binomial")
    # sev = claim severity distribution, choose from ("weibull", "lognormal",
    # "gamma")
    # hyper_theta_freq = hyperparameters for the claim frequency
    # hyper_theta_sev = hyperparameters for the claim severity

    if freq == "binomial":
        low_n, up_n, low_p, up_p = hyper_theta_freq
        nPrior = st.randint(low_n, up_n)
        pPrior = st.uniform(low_p, up_p)
        if sev == "weibull" or sev == "gamma":
            low_k, up_k, low_beta, up_beta = hyper_theta_sev
            kPrior = st.uniform(low_k, up_k)
            betaPrior = st.uniform(low_beta, up_beta)
            theta_types = ("discrete", "continuous", "continuous", "continuous")
            theta_names = ("n", "p", "k", "β")
            return IndependentPrior(
                [nPrior, pPrior, kPrior, betaPrior], theta_types, theta_names
            )
        else:
            low_mu, up_mu, low_sig, up_sig = hyper_theta_sev
            muPrior = st.uniform(low_mu, up_mu)
            sigPrior = st.uniform(low_sig, up_sig)
            theta_types = ("discrete", "continuous", "continuous", "continuous")
            theta_names = ("n", "p", "μ", "σ")
            return IndependentPrior(
                [nPrior, pPrior, muPrior, sigPrior], theta_types, theta_names
            )
    elif freq == "negative binomial":
        low_a, up_a, low_p, up_p = hyper_theta_freq

        if sev == "weibull" or sev == "gamma":
            low_k, up_k, low_beta, up_beta = hyper_theta_sev
            theta_names = ("a", "p", "k", "β")
            return IndependentUniformPrior(
                [low_a, low_p, low_k, low_beta],
                [up_a, up_p, up_k, up_beta],
                theta_names,
            )
        else:
            low_mu, up_mu, low_sig, up_sig = hyper_theta_sev
            theta_names = ("a", "p", "μ", "σ")
            return IndependentUniformPrior(
                [low_a, low_p, low_mu, low_sig],
                [up_a, up_p, up_mu, up_sig],
                theta_names,
            )
    else:
        low_lam, up_lam = hyper_theta_freq
        if sev == "weibull" or sev == "gamma":
            low_k, up_k, low_beta, up_beta = hyper_theta_sev
            theta_names = ("λ", "k", "β")
            return IndependentUniformPrior(
                [low_lam, low_k, low_beta], [up_lam, up_k, up_beta], theta_names
            )
        else:
            low_mu, up_mu, low_sig, up_sig = hyper_theta_sev
            theta_names = ("λ", "μ", "σ")
            return IndependentUniformPrior(
                [low_lam, low_mu, low_sig], [up_lam, up_mu, up_sig], theta_names
            )

    raise RuntimeError("Unsupported prior distribution")
