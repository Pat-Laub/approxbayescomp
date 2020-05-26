# -*- coding: utf-8 -*-
"""
@author: Pat and Pierre-O
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice, default_rng
from scipy.stats import gaussian_kde

from .weighted import weighted_distplot

################################################################
# This was the first function I wrote to show the results of an
# ABC-SMC fit. It zooms out to show the entire prior distribution
# so the posterior looks very peaked.
#
# TO DO: The bootstrapping leads to incorrect bandwidth selection
#   for the KDE (and histograms). This is fixed in the following
#   function.
# ###############################################################
def _plot_results(
    samples, weights, prior, momentEst=None, boot=False, filename=None, thetaTrue=None
):
    numSamples, numTheta = samples.shape
    if boot:
        indices = choice(numSamples, size=10 * numSamples, p=weights)
        samplesBoot = samples[indices, :]
    else:
        samplesBoot = samples

    meanAPEsts = np.dot(weights, samples)
    print("Posterior Mean estimates:", meanAPEsts)

    fig, axs = plt.subplots(1, prior.dim)

    for i, priorI in enumerate((prior.marginals)):
        priorL = priorI.isf(1)
        if priorL > 0:
            xlimL = priorL * 0.9
        elif priorL == 0:
            xlimL = -1
        else:
            xlimL = priorL * 1.1

        priorR = priorI.isf(0)
        xlimR = priorR * 1.1

        xs = np.linspace(xlimL, xlimR, 100)
        xs = np.sort(
            np.concatenate(
                (
                    xs,
                    [
                        priorL,
                        priorL - 1e-8,
                        priorL + 1e-8,
                        priorR,
                        priorR - 1e-8,
                        priorR + 1e-8,
                    ],
                )
            )
        )

        ax = axs[i]
        ax.set_title(prior.names[i])

        (priorLine,) = ax.plot(xs, priorI.pdf(xs), label="Prior")
        ax.axvline(
            priorI.mean(),
            color=priorLine.get_color(),
            linestyle="--",
            label="Prior Mean",
        )

        xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        kde = gaussian_kde(samples[:, i], weights=weights)
        (kdeLine,) = ax.plot(xs, kde(xs), label="ABC Posterior (KDE)")

        ax.hist(
            samplesBoot[:, i],
            bins=20,
            density=True,
            color=kdeLine.get_color(),
            alpha=0.2,
        )

        ax.axvline(meanAPEsts[i], color="r", label="ABC Posterior Mean")

        if momentEst and momentEst[i]:
            ax.axvline(momentEst[i], color="g", label="MOM")

        if thetaTrue:
            ax.axvline(thetaTrue[i], color="k", label="True Value")

    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
    )
    plt.tight_layout()
    if filename:
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()


def plot_abc_fit(fit):
    numSamples = len(fit.samples)
    models = np.unique(fit.models)

    fig, axs = plt.subplots(len(models), len(fit.samples[0]))

    for mInd, m in enumerate(models):
        samples = np.array(
            [fit.samples[i] for i in range(numSamples) if fit.models[i] == m]
        )
        weights = fit.weights[fit.models == m]
        weights /= np.sum(weights)
        numTheta = samples.shape[1]

        for i in range(numTheta):
            ax = axs[mInd, i] if len(models) > 1 else axs[i]
            weighted_distplot(samples[:, i], weights, ax=ax)

    return axs
