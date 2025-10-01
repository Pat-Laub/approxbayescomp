# API Reference

This section provides comprehensive documentation for all approxbayescomp classes, functions, and modules. The API documentation is automatically generated from the source code docstrings.

## Overview

The approxbayescomp package is organized into several key modules:

- **[SMC](smc.md)** - Sequential Monte Carlo ABC implementation
- **[Prior](prior.md)** - Prior distribution classes
- **[Distance](distance.md)** - Distance metrics for comparing observations
- **[Simulate](simulate.md)** - Data simulation functions
- **[Plotting](plot.md)** - Visualization tools for ABC results
- **[Weighted Statistics](weighted.md)** - Weighted statistical functions
- **[Utilities](utils.md)** - Helper functions and utilities

## Quick Reference

### Core Classes

| Class | Purpose | Module |
|-------|---------|--------|
| `Model` | ABC model specification | `approxbayescomp.smc` |
| `Population` | ABC population container | `approxbayescomp.smc` |
| `Psi` | Tolerance schedule class | `approxbayescomp.smc` |
| `IndependentPrior` | Independent prior distributions | `approxbayescomp.prior` |
| `IndependentUniformPrior` | Independent uniform priors | `approxbayescomp.prior` |

### Key Functions

| Function | Purpose | Module |
|----------|---------|--------|
| `smc()` | Run SMC-ABC algorithm | `approxbayescomp.smc` |
| `compute_psi()` | Compute tolerance schedule | `approxbayescomp.smc` |
| `simulate_claim_data()` | Simulate claim data | `approxbayescomp.simulate` |
| `l1()`, `l2()`, `wasserstein()` | Distance metrics | `approxbayescomp.distance` |
| `plot_posteriors()` | Plot posterior distributions | `approxbayescomp.plot` |
| `weighted_quantile()` | Weighted quantiles | `approxbayescomp.weighted` |
