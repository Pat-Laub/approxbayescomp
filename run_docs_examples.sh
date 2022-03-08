#!/bin/bash
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute docs/geometric-exponential.ipynb --inplace
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute docs/frequency-dependent-claim-sizes.ipynb --inplace
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute docs/bivariate-observations.ipynb --inplace
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute docs/seasonal-claim-arrivals.ipynb --inplace
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute model-selection.ipynb