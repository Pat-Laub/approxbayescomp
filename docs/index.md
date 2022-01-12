# The Python package _approxbayescomp_ for Approximate Bayesian Computation 

## Package Description

This package implements an efficient _Approximate Bayesian Computation (ABC)_ algorithm  &mdash; the _sequential Monte Carlo (SMC)_ algorithm. It was designed to fit insurance loss distributions, though it can handle any general statistical problem.

## Installation

To install simply run 

```pip install approxbayescomp```

Soon, it will be possible to install using `conda`; at that point the preferred method will be to run

```conda install approxbayescomp```

## Example

For example, imagine we have an i.i.d. sample of random sums of lognormal variables where the number of summands is Poisson distributed.
The fit this data we would run:

``` python
import approxbayescomp as abc

# Load data to fit
obsData = [1.0, 2.0, 3.0]

# Frequency-Loss Model
freq = "poisson"
sev = "lognormal"
psi = abc.Psi("sum") # Aggregation process

# Fit the model to the data using ABC
prior = abc.IndependentUniformPrior([(0, 100), (-5, 5), (0, 3)])
model = abc.Model(freq, sev, psi, prior)
fit = abc.smc(numIters, popSize, obsData, model)
```

## Other Resources

This package is the result of our paper "[Approximate Bayesian Computation to fit and compare insurance loss models](https://arxiv.org/abs/2007.03833)". 
For a description of the aims and methodology of ABC check out our paper, it is written with ABC newcomers in mind.

For a brief video overview of the package, see our 7 min lightning talk at the Insurance Data Science conference:

<figure markdown>
  [![ABC Talk at Insurance Data Science conference](ids-screenshot.png){ width="500" }](https://www.youtube.com/watch?v=EtZdCWoFMBA)
  <figcaption><a href="https://www.youtube.com/watch?v=EtZdCWoFMBA">ABC Talk at Insurance Data Science conference</a></figcaption>
</figure>

For examples of this package in use, check out the Jupyter notebooks in our [online supplement repository](https://github.com/LaGauffre/ABCFitLoMo) for the paper.

## Details

The main design goal for this package was computational speed.
ABC is notoriously computationally demanding, so we spent a long time optimising the code as much as possible.
The key functions are JIT-compiled to C with `numba` (we experimented with JIT-compiling the entire SMC algorithm, but `numba`'s random variable generation is surprisingly slower than `numpy`'s implementation).
Everything that can be `numpy`-vectorised has been.
And we scale to use as many CPU cores available on a machine using `joblib`.
We also aimed to have total reproducibility, so for any given seed value the resulting ABC posterior samples will always be identical. 

Our main dependencies are joblib, numba, numpy, and scipy.
The package also uses psutil, matplotlib, fastprogress, and hilbertcurve, though in most cases these can be commented out if it were necessary.

!!! note
        Patrick has a rough start at a C++ version of this package at the [cppabc](https://github.com/Pat-Laub/cppabc) repository.
        It only handles the specific Geometric-Exponential random sums case, though if you are interested in collaborating to expand this, let him know!

## Authors

- [Patrick Laub](https://pat-laub.github.io/) (author, maintainer),
- [Pierre-Olivier Goffard](http://pierre-olivier.goffard.me/) (author).

## Citation

Pierre-Olivier Goffard, Patrick J. Laub (2021), _Approximate Bayesian Computations to fit and compare insurance loss models_, Insurance: Mathematics and Economics, 100, pp. 350-371

``` bibtex
@article{approxbayescomp,
  title={Approximate Bayesian Computations to fit and compare insurance loss models},
  author={Goffard, Pierre-Olivier and Laub, Patrick J},
  journal={Insurance: Mathematics and Economics},
  volume={100},
  pages={350--371},
  year={2021},
  publisher={Elsevier}
}
```
