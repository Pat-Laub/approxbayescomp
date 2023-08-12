import numpy as np
from typing import Optional, Tuple

from .utils import SimpleKDE, kde


class Population:
    """
    A Population object stores a collection of particles in the
    SMC procedure. Each particle is a potential theta parameter which
    could explain the observed data. Each theta has a corresponding
    weight, belongs to a specific model (as we may be fitting multiple
    competing models simultaneously), and has an observed distance of
    its fake data to the observed data.
    """

    def __init__(self, models, weights, samples, dists, M) -> None:
        self.models = np.array(models)
        self.weights = np.array(weights)
        self.weights /= np.sum(weights)
        if isinstance(samples, list):
            self.samples = np.vstack(samples)
        else:
            self.samples = np.array(samples)
        self.dists = np.array(dists)
        self.M = M

    def size(self) -> int:
        return len(self.models)

    def model_sizes(self) -> Tuple[int, ...]:
        return tuple(np.sum(self.models == m) for m in range(self.M))

    def model_weights(self) -> Tuple[float, ...]:
        return tuple(np.sum(self.weights[self.models == m]) for m in range(self.M))

    def drop_worst_particle(self) -> None:
        """
        Throw away the particle in this population which has the largest
        distance to the observed data.
        """
        dropIndex = np.argmax(self.dists)
        self.models = np.delete(self.models, dropIndex, 0)
        self.weights = np.delete(self.weights, dropIndex, 0)
        self.samples = np.delete(self.samples, dropIndex, 0)
        self.dists = np.delete(self.dists, dropIndex, 0)
        self.weights /= np.sum(self.weights)

    def drop_small_models(self) -> None:
        """
        Throw away the particles which correspond to models which
        have an extremely small population size.
        """
        modelPopulations = self.model_sizes()
        modelWeights = self.model_weights()

        for m in range(self.M):
            if modelPopulations[m] < 5 or modelWeights[m] == 0:
                keep = self.models != m
                self.models = self.models[keep]
                self.weights = self.weights[keep]
                self.weights /= np.sum(self.weights)
                self.samples = self.samples[keep, :]
                self.dists = self.dists[keep]

    def ess(self) -> Tuple[int, ...]:
        """
        Calculate the effective sample size (ESS) for each model in this population.
        """
        essPerModel = []
        for modelNum in set(self.models):
            weightsForThisModel = self.weights[self.models == modelNum]
            weightsForThisModel /= np.sum(weightsForThisModel)
            essPerModel.append(int(np.round(1 / np.sum(weightsForThisModel**2))))

        return tuple(essPerModel)

    def total_ess(self) -> int:
        """
        Calculate the total effective sample size (ESS) of this population.
        That is, add together the ESS for each model under consideration.
        """
        return sum(self.ess())

    def fit_kdes(self) -> Tuple[Optional[SimpleKDE], ...]:
        """
        Fit a kernel density estimator (KDE) to each model's subpopulation
        in this population. Return all the KDEs in tuple. If there isn't
        enough data for some model to fit a KDE, that entry will be None.
        """
        kdes = []

        for m in range(self.M):
            samplesForThisModel = self.samples[self.models == m, :]
            weightsForThisModel = self.weights[self.models == m]

            K = None
            if samplesForThisModel.shape[0] >= 5:
                try:
                    K = kde(samplesForThisModel, weightsForThisModel)
                    L = np.linalg.cholesky(K.covariance)  # type: ignore
                    log_det = 2 * np.log(np.diag(L)).sum()
                    K = SimpleKDE(K.dataset, K.weights, K.d, K.n, K.inv_cov, L, log_det)
                except np.linalg.LinAlgError:  # type: ignore
                    pass

            kdes.append(K)

        return tuple(kdes)

    def clone(self) -> "Population":
        """
        Create a deep copy of this population object.
        """
        return Population(self.models.copy(), self.weights.copy(), self.samples.copy(), self.dists.copy(), self.M)

    def subpopulation(self, keep) -> "Population":
        """
        Create a subpopulation of particles from this population where we keep
        only the particles at the locations of True in the supplied boolean vector.
        """
        return Population(self.models[keep], self.weights[keep], self.samples[keep, :], self.dists[keep], self.M)

    def combine(self, other: "Population") -> "Population":
        """
        Combine this population with another to create one larger population.
        """
        ms = np.concatenate([self.models, other.models])
        samples = np.concatenate([self.samples, other.samples], axis=0)
        dists = np.concatenate([self.dists, other.dists])

        # Some care needs to be taken to adjust the weights when combining.
        # See Appendix B.1 of Leah South's PhD thesis.
        # https://eprints.qut.edu.au/132155/1/Leah_South_Thesis.pdf
        popWeights = np.array((self.total_ess(), other.total_ess()), dtype=np.float64)
        popWeights /= sum(popWeights)
        weights = np.concatenate([popWeights[0] * self.weights, popWeights[1] * other.weights])

        return Population(ms, weights, samples, dists, self.M)
