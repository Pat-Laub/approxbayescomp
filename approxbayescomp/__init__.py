__version__ = "0.2.0"

from .plot import draw_prior, plot_posteriors, weighted_distplot
from .prior import IndependentPrior, IndependentUniformPrior
from .simulate import simulate_claim_data, simulate_claim_sizes
from .smc import Model, Population, Psi, compute_psi, smc
from .wasserstein import (
    identity,
    wass_2Ddist,
    wass_2Ddist_ss,
    wass_dist,
    wass_sumstats,
    wrap_ss_curve_matching,
)
from .weighted import median as weighted_median
from .weighted import quantile as weighted_quantile
from .weighted import resample, resample_and_kde
