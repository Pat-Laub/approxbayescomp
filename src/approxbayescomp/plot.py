# -*- coding: utf-8 -*-
"""
@author: Patrick Laub and Pierre-O Goffard
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice, default_rng
from scipy.stats import gaussian_kde  # type: ignore

from .weighted import iqr, resample, resample_and_kde


# A 'recommended' number of bins for a histogram of weighted data.
def freedman_diaconis_bins(data, weights, maxBins=50):
    if len(data) < 2:
        return 1
    neff = 1 / np.sum(weights**2)
    h = 2 * iqr(data, weights) / (neff ** (1 / 3))

    # fall back to sqrt(data.size) bins if iqr is 0
    if h == 0:
        return min(int(np.sqrt(data.size)), maxBins)
    else:
        return min(int(np.ceil((data.max() - data.min()) / h)), maxBins)


# This is supposed to be a replacement for the seaborn library
# function 'distplot' which works correctly for weighted samples.
def weighted_distplot(
    data, weights, ax=None, cut=3, clip=(-np.inf, np.inf), seed=1, repeats=10, hist=True, despine=True
):
    if not ax:
        ax = plt.gca()

    # Pull out the next color
    (line,) = ax.plot(data.mean(), 0)
    color = line.get_color()
    line.remove()

    # Plot the histogram
    if hist:
        rng = default_rng(seed)
        dataResampled = data[resample(rng, weights, repeats=repeats)]
        bins = freedman_diaconis_bins(data, weights)
        ax.hist(dataResampled, bins=bins, color=color, density=True, alpha=0.4)

    # Choose support for KDE
    neff = 1 / sum(weights**2)
    scott = neff ** (-1.0 / 5)
    cov = np.cov(data, bias=False, aweights=weights)
    bw = scott * np.sqrt(cov)

    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])

    xs = np.linspace(support_min, support_max, 200)

    # Plot the KDE
    K = gaussian_kde(data.T, weights=weights)
    ys = K(xs)
    ax.plot(xs, ys, color=color)

    if despine:
        seaborn_despine()


def draw_prior(prior, axs, color="tab:purple"):
    lines = []
    for i, priorI in enumerate((prior.marginals)):
        priorL = priorI.isf(1)
        priorR = priorI.isf(0)

        if not np.isfinite(priorL):
            priorL = priorI.isf(0.95)
        if not np.isfinite(priorR):
            priorR = priorI.isf(0.05)

        priorWidth = priorR - priorL

        # Want to have x axis be 10% padding, then 80% prior, then 10% padding.
        xAxisWidth = priorWidth / 0.8
        padding = xAxisWidth - priorWidth
        xlimL = priorL - padding / 2
        xlimR = priorR + padding / 2

        xs = np.linspace(xlimL, xlimR, 100)
        xs = np.sort(np.concatenate((xs, [priorL, priorL - 1e-8, priorL + 1e-8, priorR, priorR - 1e-8, priorR + 1e-8])))

        (priorLine,) = axs[i].plot(xs, priorI.pdf(xs), label="Prior", color=color, alpha=0.75, zorder=0)
        axs[i].set_xlim([xlimL, xlimR])
        lines.append(priorLine)

    return lines


def plot_posteriors(
    fit,
    prior,
    subtitles=[],
    refLines=None,
    figsize=(5.0, 2.0),
    dpi=350,
    refStyle={"color": "black", "linestyle": "--"},
    removeYAxis=None,
):
    numThetas = len(prior.marginals)
    fig, axs = plt.subplots(1, numThetas, tight_layout=True, figsize=figsize, dpi=dpi)

    if removeYAxis is None:
        removeYAxis = numThetas > 4

    if numThetas == 1:
        axs = [axs]
    
    if len(subtitles) == 0 and prior.names is not None:
        subtitles = prior.names

    for i in range(numThetas):
        pLims = [prior.marginals[i].isf(1), prior.marginals[i].isf(0)]

        dataResampled, xs, ys = resample_and_kde(fit.samples[:, i], fit.weights, clip=pLims)
        axs[i].plot(xs, ys)

        if refLines is not None:
            axs[i].axvline(refLines[i], **refStyle)

        if i < len(subtitles):
            axs[i].set_title(subtitles[i])

        if removeYAxis:
            axs[i].set_yticks([])

    draw_prior(prior, axs)
    seaborn_despine()
    if removeYAxis:
        seaborn_despine(left=True)


################################################################
# This was the first function I wrote to show the results of an
# ABC-SMC fit. It zooms out to show the entire prior distribution
# so the posterior looks very peaked.
#
# TO DO: The bootstrapping leads to incorrect bandwidth selection
#   for the KDE (and histograms). This is fixed in the following
#   function.
# ###############################################################
def _plot_results(samples, weights, prior, momentEst=None, boot=False, filename=None, thetaTrue=None):
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
        xs = np.sort(np.concatenate((xs, [priorL, priorL - 1e-8, priorL + 1e-8, priorR, priorR - 1e-8, priorR + 1e-8])))

        ax = axs[i]
        ax.set_title(prior.names[i])

        (priorLine,) = ax.plot(xs, priorI.pdf(xs), label="Prior")
        ax.axvline(priorI.mean(), color=priorLine.get_color(), linestyle="--", label="Prior Mean")

        xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        kde = gaussian_kde(samples[:, i], weights=weights)
        (kdeLine,) = ax.plot(xs, kde(xs), label="ABC Posterior (KDE)")

        ax.hist(samplesBoot[:, i], bins=20, density=True, color=kdeLine.get_color(), alpha=0.2)

        ax.axvline(meanAPEsts[i], color="r", label="ABC Posterior Mean")

        if momentEst and momentEst[i]:
            ax.axvline(momentEst[i], color="g", label="MOM")

        if thetaTrue:
            ax.axvline(thetaTrue[i], color="k", label="True Value")

    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels))
    plt.tight_layout()
    if filename:
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.show()


def plot_abc_fit(fit):
    numSamples = len(fit.samples)
    models = np.unique(fit.models)

    fig, axs = plt.subplots(len(models), len(fit.samples[0]))

    for mInd, m in enumerate(models):
        samples = np.array([fit.samples[i] for i in range(numSamples) if fit.models[i] == m])
        weights = fit.weights[fit.models == m]
        weights /= np.sum(weights)
        numTheta = samples.shape[1]

        for i in range(numTheta):
            ax = axs[mInd, i] if len(models) > 1 else axs[i]
            weighted_distplot(samples[:, i], weights, ax=ax)

    return axs


# Copied directly from https://github.com/mwaskom/seaborn/blob/77e3b6b03763d24cc99a8134ee9a6f43b32b8e7b/seaborn/utils.py#L291
def seaborn_despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).
    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.
    Returns
    -------
    None
    """
    # Get references to the axes we want
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(("outward", val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.minorTicks)
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.minorTicks)
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # clip off the parts of the spines that extend past major ticks
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()), xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()), xticks)[-1]
                ax_i.spines["bottom"].set_bounds(firsttick, lasttick)
                ax_i.spines["top"].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()), yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()), yticks)[-1]
                ax_i.spines["left"].set_bounds(firsttick, lasttick)
                ax_i.spines["right"].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)
