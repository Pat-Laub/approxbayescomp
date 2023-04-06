import approxbayescomp as abc
import unittest
import numpy as np
import scipy.stats as st


class TestIndependentUniformPrior(unittest.TestCase):
    def setUp(self):
        self.bounds = [(0, 1), (2, 3), (4, 5)]
        self.names = ['x', 'y', 'z']
        self.prior = abc.IndependentUniformPrior(self.bounds, self.names)

    def test_pdf(self):
        theta = np.array([0.5, 2.5, 4.5])
        expected_pdf = 1 / (1 * 1 * 1)
        self.assertAlmostEqual(self.prior.pdf(theta), expected_pdf)

    def test_sample(self):
        rg = np.random.default_rng(0)
        expected_sample = np.array([0.636962, 2.269787, 4.040974])
        np.testing.assert_allclose(self.prior.sample(rg), expected_sample, rtol=1e-5)


class TestIndependentPrior(unittest.TestCase):
    def setUp(self):
        marginals = [st.norm(0, 1), st.uniform(0, 1)]
        types = ["continuous", "continuous"]
        self.prior = abc.IndependentPrior(marginals, types=types)

    def test_pdf(self):
        theta = np.array([0.5, 0])
        expected_pdf = st.norm.pdf(0.5) * st.uniform.pdf(0)
        self.assertAlmostEqual(self.prior.pdf(theta), expected_pdf)

    def test_sample(self):
        rg = np.random.default_rng(0)
        expected_sample = np.array([0.12573 , 0.269787])
        np.testing.assert_allclose(self.prior.sample(rg), expected_sample, rtol=1e-5)


class TestScaleLocIndependentPrior(unittest.TestCase):
    def setUp(self):
        marginals = [st.expon(scale=10, loc=1), st.gamma(a=2, scale=2, loc=1)]
        types = ["continuous", "continuous"]
        self.prior = abc.IndependentPrior(marginals, types=types)

    def test_pdf(self):
        theta = np.array([2.0, 3.0])
        expected_pdf = st.expon(scale=10, loc=1).pdf(2.0) * st.gamma(a=2, scale=2, loc=1).pdf(3.0)
        self.assertAlmostEqual(self.prior.pdf(theta), expected_pdf)

    def test_sample(self):
        rg = np.random.default_rng(0)
        expected_sample = np.array([7.799319, 4.003742])
        np.testing.assert_allclose(self.prior.sample(rg), expected_sample, rtol=1e-5)
