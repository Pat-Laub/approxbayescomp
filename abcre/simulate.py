import numpy as np


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

    else:
        raise Exception(f"Unknown severity distribution '{sev}")


def simulate_claim_data(rg, T, freq, sev, theta):
    # T = integer that corresponds to the number of time period observed
    # freq = claim frequency ditribution to be chosen in ("poisson",
    # "binomial", "negative binomial)
    # theta_freq = parameters of the claim frequency distribution
    # sev = claim sizes distribution to be chosen in ("weibuill", "lognormal",
    # "gamma")
    # theta_sev = parameters of the claim sizes distribution

    # Echantillon de nombre de sinistres
    if freq is None:
        theta_sev = theta
        freqs = np.ones(T).astype(np.int64)
    elif type(freq) == list or type(freq) == np.ndarray:
        theta_sev = theta
        freqs = freq
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
    elif freq == "geometric":
        p = theta[0]
        theta_sev = theta[1:]
        freqs = rg.geometric(1 - p, size=T) - 1
    elif freq == "negative binomial":
        a, p = theta[0:2]
        theta_sev = theta[2:]
        freqs = rg.negative_binomial(a, p, size=T)

    else:
        raise Exception(f"Unknown frequency distribution: {freq}")

    if sev == "frequency dependent exponential":
        sevs = np.concatenate(
            [simulate_claim_sizes(rg, freq, sev, theta_sev) for freq in freqs]
        )
    else:
        sevs = simulate_claim_sizes(rg, freqs.sum(), sev, theta_sev)

    return (freqs, sevs)
