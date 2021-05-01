# Approximate Bayesian Computation for insurance

This package is the result of our paper "[Approximate Bayesian Computation to fit and compare insurance loss models](https://arxiv.org/abs/2007.03833)".
It implements an efficient ABC algorithm -- the sequential Monte Carlo (SMC) algorithm -- and is targeted towards insurance problems (originally we had just considered reinsurance problems, hence the name _abcre_).
For a description of the aims and methodology of ABC check out our paper, it is written with ABC newcomers in mind.
For examples of this package in use, check out the Jupyter notebooks in our [online supplement repository](https://github.com/LaGauffre/ABCFitLoMo) for the paper.

The main design goal for this package was computational speed.
ABC is notoriously computationally demanding, so we spent a long time optimising the code as much as possible.
The key functions are JIT-compiled to C with `numba` (we experimented with JIT-compiling the entire SMC algorithm, but `numba`'s random variable generation is surprisingly slower than `numpy`'s implementation).
Everything that can be `numpy`-vectorised has been.
And we scale to use as many CPU cores available on a machine using `joblib`.
We also aimed to have total reproducibility, so for any given seed value the resulting ABC posterior samples will always be identical. 

To install the packages required for the `abcre` library using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), run:

```bash
conda install joblib matplotlib numba numpy psutil scipy
pip install fastprogress hilbertcurve
```

P.S. I have a rough start at a C++ version of this package at the [cppabc](https://github.com/Pat-Laub/cppabc) repository.
It only handles the specific Geometric-Exponential random sums case, though if you are interested in collaborating to expand this, let me know!
