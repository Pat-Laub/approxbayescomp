# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
__version__ = "0.2.0"

from .distance import l1, l2, wasserstein, wasserstein2D, wrap_ss_curve_matching
from .population import Population
from .plot import draw_prior, plot_posteriors, weighted_distplot
from .prior import IndependentPrior, IndependentUniformPrior
from .psi import Psi, compute_psi
from .simulate import simulate_claim_data, simulate_claim_sizes
from .smc import smc
from .weighted import median as weighted_median
from .weighted import quantile as weighted_quantile
from .weighted import resample, resample_and_kde
