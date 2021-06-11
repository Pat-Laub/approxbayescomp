import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="approxbayescomp",
    version="0.1.0",
    description="Approximate Bayesian Computation for actuaries",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Pat-Laub/approxbayescomp",
    author="Patrick Laub and Pierre-Olivier Goffard",
    author_email="patrick.laub@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
    ],
    packages=["approxbayescomp"],
    include_package_data=True,
    install_requires=["joblib", "numba", "numpy>=1.17", "scipy>=1.4",
        "psutil", "matplotlib", "fastprogress", "hilbertcurve>=2.0"],
)
