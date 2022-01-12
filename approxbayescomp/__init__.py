__version__ = "0.1.0"

from .plot import _plot_results
from .prior import IndependentPrior, IndependentUniformPrior
from .simulate import simulate_claim_data, simulate_claim_sizes
from .smc import Fit, Model, Psi, SimulationModel, compute_psi, smc
from .wasserstein import (
    identity,
    wass_2Ddist,
    wass_2Ddist_ss,
    wass_dist,
    wass_sumstats,
    wrap_ss_curve_matching,
)
from .weighted import resample, resample_and_kde, weighted_distplot
