# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rnd
from numba import njit


@njit(nogil=True)
def sample_discrete_dist(weights):
    u = rnd.random()
    cdf_i = 0
    maxIndex = len(weights)-1
    for i in range(maxIndex):
        cdf_i += weights[i]
        if u < cdf_i:
            return i
    return maxIndex

def sim_multivariate_normal(rg, mu, L):
    return mu + L @ rg.normal(size=len(mu))

# Unused version which handles any number of states in the chain
@njit(nogil=True)
def markov_chain(p_0, P, N):
    X = np.empty(N, dtype=np.int64)
    X[0] = sample_discrete_dist(p_0)
    
    for i in range(1, N):
        X[i] = sample_discrete_dist(P[X[i-1],:])
        
    return X

@njit(nogil=True)
def two_state_markov_chain(p_00, P_00, P_11, N):
    X = np.empty(N, np.int64)
    
    u = rnd.random()
    if u < p_00:
        X_i = 0
    else:
        X_i = 1
    
    X[0] = X_i
    
    for i in range(1, N):
        u = rnd.random()
        if X_i == 0:
            if u < P_00:
                X_i = 0
            else:
                X_i = 1
        else:
            if u < P_11:
                X_i = 1
            else:
                X_i = 0
                
        X[i] = X_i
        
    return X

def simulate_claim_sizes(rg, R, sev, theta_sev):
    # n_t is integer valued
    # sev is a claim sizes disribution to chosen in ("weibull", "lognormal",
    # "gamma")
    # theta_sev corresponds to the parameters of the claim sizes distribution
    if sev == "weibull":
        k, scale = theta_sev
        return scale * rg.weibull(k, size=R)
    elif sev == "exponential":
        scale = theta_sev
        return scale * rg.exponential(size=R)
    elif sev == "gamma":
        k, scale = theta_sev
        return scale * rg.gamma(k, size=R)
    elif sev == "lognormal":
        mu, sigma = theta_sev
        return rg.lognormal(mu, sigma, size=R)
    elif sev == "dependent lognormal":  # exponential of gaussian vector
        if R == 0:
            return np.array([])
        else:
            mu, sigma, ρ = theta_sev
            Σ = np.ones((R, R)) * ρ * sigma ** 2 + np.identity(R) * (
                sigma ** 2 - sigma ** 2 * ρ
            )

            return np.exp(rg.multivariate_normal(np.repeat(mu, R), Σ, 1).flatten())
    elif sev == "frequency dependent exponential":
        scale, cor = theta_sev
        return scale * np.exp(cor * R) * rg.exponential(size=R)
    elif sev == "weibull-pareto":
        shape, tail, threshold = theta_sev
        scale = (shape / (shape + tail)) ** (1 / shape) * threshold
        r = (
            (tail / threshold)
            * (1 - np.exp(-(shape + tail) / shape))
            / (tail / threshold + (shape / threshold) * np.exp(-(shape + tail) / shape))
        )
        par_rvs = threshold * (1 + rg.pareto(tail, size=R))
        binom_rvs = rg.binomial(1, r, size=R)
        weib_rvs = (
            -np.log(1 - (1 - np.exp(-((threshold / scale) ** shape))) * rg.uniform(size=R))
        ) ** (1 / shape) * scale
        return binom_rvs * weib_rvs + (1 - binom_rvs) * par_rvs
    else:
        raise Exception(f"Unknown severity distribution '{sev}")

def bivariate_clayton(rg, T, theta_cop):
    u = rg.uniform(size=(T, 2))
    u[:,1] = (u[:,0]**(-theta_cop) * (u[:,1]**(-theta_cop/(1+theta_cop)) - 1) + 1)**(-1/theta_cop)
    return u

def bivariate_frank(rg, T, theta_cop):
    u = rg.uniform(size=(T, 2))
    expU1 = np.exp(-theta_cop * u[:,0])
    u[:,1] = -1/(theta_cop) * np.log( 1 + (u[:,1] * (1-np.exp(-theta_cop)) ) / ( u[:,1] * (expU1-1) - expU1) )
    
    return u

from scipy import stats 

def bivariate_poisson(rg, T, theta_cop, theta_freqs):
    #u = bivariate_clayton(rg, T, theta_cop)
    u = bivariate_frank(rg, T, theta_cop)
    freqs1 = stats.poisson.isf(1 - u[:,0], theta_freqs[0]).astype(np.int64)
    freqs2 = stats.poisson.isf(1 - u[:,1], theta_freqs[1]).astype(np.int64)
    return (np.maximum(freqs1, 0), np.maximum(freqs2, 0))

def simulate_claim_data(rg, T, freq, sev, theta):
    # T = integer that corresponds to the number of time period observed
    # freq = claim frequency ditribution to be chosen in ("poisson",
    # "binomial", "negative binomial)
    # theta_freq = parameters of the claim frequency distribution
    # sev = claim sizes distribution to be chosen in ("weibuill", "lognormal",
    # "gamma")
    # theta_sev = parameters of the claim sizes distribution

    # Echantillon de nombre de sinistres
    if type(freq) == list or type(freq) == np.ndarray:
        theta_sev = theta
        freqs = freq
    elif freq is None or freq == "ones":
        theta_sev = theta
        freqs = np.ones(T).astype(np.int64)
    elif freq == "bernoulli":
        p = theta[0]
        theta_sev = theta[1:]
        freqs = rg.binomial(1, p, size=T)
    elif freq == "binomial":
        n, p = theta[0:2]
        theta_sev = theta[2:]
        freqs = rg.binomial(n, p, size=T)
    elif freq == "poisson":
        lam = theta[0]
        theta_sev = theta[1:]
        freqs = rg.poisson(lam, size=T)
    elif freq == "zi_poisson":
        a, lam = theta[0:2]
        theta_sev = theta[2:]
        freqs = rg.binomial(1, 1 - a, size=T) * rg.poisson(lam, size=T)
    elif freq == "seasonal_poisson":
        a, gamma, c = theta[0:3]
        theta_sev = theta[3:]
        t = np.arange(T)
        mus = a * (1 + (gamma / (2*np.pi*c)) * (np.cos(2 * np.pi * t * c) - np.cos(2 * np.pi * (t + 1) * c)))
        freqs = rg.poisson(mus)
    elif freq == "cyclical_poisson":
        a, b, c = theta[0:3]
        theta_sev = theta[3:]
        t = np.arange(T)
        mus = (a + b + b * (np.cos(2 * np.pi * t * c) - np.cos(2 * np.pi * (t + 1) * c)) / 2 / np.pi / c)
        freqs = rg.poisson(mus)
    elif freq == "ar_poisson":
        a, b, c = theta[0:3]
        theta_sev = theta[3:]
        lambdas, freqs = [0], [0]
        for s in range(T):
            lambdas.append(a + b * lambdas[-1] + c * freqs[-1])
            freqs.append(rg.poisson(lambdas[-1]))
        freqs = np.array(freqs)[1:]
    elif freq == "INAR":
        a, b = theta[0:2]
        theta_sev = theta[2:]
        freqs = [0]
        noise = rg.poisson(b, size=T)
        for s in range(T):
            freqs.append(rg.binomial(freqs[-1], a) + noise[s])
        freqs = np.array(freqs)[1:]
    elif freq == "geometric":
        p = theta[0]
        theta_sev = theta[1:]
        freqs = rg.geometric(1 - p, size=T) - 1
    elif freq == "negative binomial":
        a, p = theta[0:2]
        theta_sev = theta[2:]
        freqs = rg.negative_binomial(a, p, size=T)
    elif freq == "markov modulated poisson":

        p_00 = theta[0]
        P_00 = theta[1]
        P_11 = theta[2]

        lambda0 = theta[3]
        lambda1 = lambda0 + theta[4]
        theta_sev = theta[5:]

        chain = two_state_markov_chain(p_00, P_00, P_11, T)
        lambdaChain = (chain == 0)*lambda0 + (chain==1)*lambda1
        freqs = rg.poisson(lambdaChain)        

    elif freq == "bivariate copula poisson":
        theta_cop = theta[0]
        theta_freqs = theta[1:3]
        freqs1, freqs2 = bivariate_poisson(rg, T, theta_cop, theta_freqs)

        nparam = len(theta[3:]) // 2
        theta_sev1 = theta[3:3+nparam]
        theta_sev2 = theta[3+nparam:]
        # print(f"Num claims = {freqs1.sum(), freqs2.sum()}")
        sevs1 = simulate_claim_sizes(rg, freqs1.sum(), sev, theta_sev1)
        sevs2 = simulate_claim_sizes(rg, freqs2.sum(), sev, theta_sev2)
        return [(freqs1, sevs1), (freqs2, sevs2)]
    elif freq == "bivariate poisson":
        theta_freqs = theta[:3]
        Lambda = rg.lognormal(mean=0, sigma=theta_freqs[0], size=T)

        freqs1 = rg.poisson(Lambda * theta_freqs[1])
        freqs2 = rg.poisson(Lambda * theta_freqs[2])

        nparam = len(theta[3:]) // 2
        theta_sev1 = theta[3:3+nparam]
        theta_sev2 = theta[3+nparam:]
        sevs1 = simulate_claim_sizes(rg, freqs1.sum(), sev, theta_sev1)
        sevs2 = simulate_claim_sizes(rg, freqs2.sum(), sev, theta_sev2)

        return [(freqs1, sevs1), (freqs2, sevs2)]
    else:
        raise Exception(f"Unknown frequency distribution: {freq}")

    # If the total number is way too large, it'll probably be discarded
    # anyway so reduce the size.
    if freqs.sum() > 1e7:
        while freqs.sum() > 1e7:
            i = np.argmax(freqs)
            freqs[i] = 1
        
    if sev == "frequency dependent exponential":
        sevs = np.concatenate(
            [simulate_claim_sizes(rg, freq, sev, theta_sev) for freq in freqs]
        )
    else:
        sevs = simulate_claim_sizes(rg, freqs.sum(), sev, theta_sev)

    return (freqs, sevs)

# Below is the version which can be compiled by numba. However these are about 2-3 times slower
# than the new numpy generators used above. It may be possible to get the generators above to be
# compiled also, a la https://numpy.org/doc/stable/reference/random/extending.html, but it currently
# is not fast enough with the current versions.

@njit(nogil=True)
def _simulate_claim_sizes(R, sev, thetaSev):
    # n_t is integer valued
    # sev is a claim sizes distribution to chosen in ("weibull", "lognormal",
    # "gamma")
    # theta_sev corresponds to the parameters of the claim sizes distribution
    if sev == "weibull":
        k, scale = thetaSev
        claims = np.empty(R, np.float64)
        for i in range(R):
            claims[i] = scale * rnd.weibull(k)
        return claims
    elif sev == "exponential":
        scale = thetaSev[0]
        claims = np.empty(R, np.float64)
        for i in range(R):
            claims[i] = scale * rnd.exponential()
        return claims
    elif sev == "gamma":
        k, scale = thetaSev
        claims = np.empty(R, np.float64)
        for i in range(R):
            claims[i] = scale * rnd.gamma(k)
        return claims
    elif sev == "lognormal":
        mu, sigma = thetaSev
        claims = np.empty(R, np.float64)
        for i in range(R):
            claims[i] = rnd.lognormal(mu, sigma)
        return claims
    # elif sev == "dependent lognormal":  # exponential of gaussian vector
    #     if R == 0:
    #         return np.array([])
    #     else:
    #         mu, sigma, ρ = param1, param2, param3
    #         Σ = np.ones((R, R)) * ρ * sigma ** 2 + np.identity(R) * (
    #             sigma ** 2 - sigma ** 2 * ρ
    #         )

    #         return np.exp(rnd.multivariate_normal(np.repeat(mu, R), Σ, 1).flatten())
    elif sev == "frequency dependent exponential":
        scale, cor = thetaSev
        claims = np.empty(R, np.float64)
        for i in range(R):
            claims[i] = scale * np.exp(cor * R) * rnd.exponential()
        return claims

    else:
        return
        # raise Exception(f"Unknown severity distribution '{sev}")


# There's actually a bug in numba's negative_binomial implementation
# which rounds the first parameter to an integer instead of leaving it
# as a float. Until it's fixed, we use our own function to simulate these
# variables.
@njit(nogil=True)
def negative_binomial(n, p):
    # Assumes n > 0 and p in [0, 1]
    Y = rnd.gamma(n, (1.0 - p) / p)
    return rnd.poisson(Y)

@njit(nogil=True)
def _simulate_claim_data(T, freq, sev, theta, obsFreqs=None):
    # T = integer that corresponds to the number of time period observed
    # freq = claim frequency distribution to be chosen in ("poisson",
    # "binomial", "negative binomial)
    # theta_freq = parameters of the claim frequency distribution
    # sev = claim sizes distribution to be chosen in ("weibuill", "lognormal",
    # "gamma")
    # theta_sev = parameters of the claim sizes distribution

    if freq == "ones":
        freqs = np.ones(T).astype(np.int64)
        thetaSev = theta
    elif freq == "obs":
        freqs = obsFreqs
        thetaSev = theta
    elif freq == "bernoulli":
        p = theta[0]
        thetaSev = theta[1:]
        freqs = np.empty(T, np.int64)
        for t in range(T):
            freqs[t] = rnd.binomial(1, p)
    elif freq == "binomial":
        n, p = theta[0:2]
        thetaSev = theta[2:]
        freqs = np.empty(T, np.int64)
        for t in range(T):
            freqs[t] = rnd.binomial(n, p)
    elif freq == "poisson":
        lam = theta[0]
        thetaSev = theta[1:]
        freqs = np.empty(T, np.int64)
        for t in range(T):
            freqs[t] = rnd.poisson(lam)
    elif freq == "geometric":
        p = theta[0]
        thetaSev = theta[1:]
        freqs = np.empty(T, np.int64)
        for t in range(T):
            freqs[t] = rnd.geometric(1 - p) - 1
    elif freq == "negative binomial":
        a, p = theta[0:2]
        thetaSev = theta[2:]
        freqs = np.empty(T, np.int64)
        for t in range(T):
            freqs[t] = negative_binomial(a, p)
    else:
        return
        # raise Exception(f"Unknown frequency distribution: {freq}")
            
    N = np.sum(freqs)
    
    if sev == "frequency dependent exponential":
        sevs = np.empty(N, np.float64)
        i = 0
        for t in range(T):
            sevs[i:i+freqs[t]] = _simulate_claim_sizes(freqs[t], sev, thetaSev)
            i += freqs[t]
    else:
        sevs = _simulate_claim_sizes(N, sev, thetaSev)

    return freqs, sevs
