# SMC

Sequential Monte Carlo ABC implementation with adaptive tolerance schedules.

## Main Function

### smc

Main SMC-ABC algorithm implementation.

::: approxbayescomp.smc.smc
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Core Classes

### Model

ABC model specification containing simulator, distance function, and other settings.

::: approxbayescomp.smc.Model
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Population

Container for ABC population with particles, weights, and distances.

::: approxbayescomp.smc.Population
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Psi

Tolerance schedule class for adaptive ABC.

::: approxbayescomp.smc.Psi
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Utilities

### compute_psi

Compute tolerance schedule from distances.

::: approxbayescomp.smc.compute_psi
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
