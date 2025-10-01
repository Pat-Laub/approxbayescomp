# Distance

Distance metrics for comparing simulated and observed data in ABC.

## Standard Distance Metrics

### l1

L1 (Manhattan) distance between observations.

::: approxbayescomp.distance.l1
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### l2

L2 (Euclidean) distance between observations.

::: approxbayescomp.distance.l2
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Distribution Distances

### wasserstein

1D Wasserstein (Earth Mover's) distance between empirical distributions.

::: approxbayescomp.distance.wasserstein
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### wasserstein2D

2D Wasserstein distance for bivariate observations.

::: approxbayescomp.distance.wasserstein2D
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Specialized Distances

### wrap_ss_curve_matching

Wrapper for curve matching distance using summary statistics.

::: approxbayescomp.distance.wrap_ss_curve_matching
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
