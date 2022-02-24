import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
import copy


"""
This module handles the visualization of imagery, spectral libraries and their overlap in image feature space.
"""


def plot_image(image,
               vmin=None,
               vmax=None,
               ax=None,
               show=True,
               origin='upper',
               axis=False,
               out=None):

    image = copy.deepcopy(image)

    if not vmin:
        vmin = np.min(image)

    if not vmax:
        vmax = np.max(image)

    image[image > vmax] = vmax
    image[image < vmin] = vmin
    image /= vmax - vmin

    if not ax:
        plt.clf()
        ax = plt.gca()

    ax.imshow(image, origin=origin)

    if not axis:
        ax.axis('off')

    if show:
        plt.show()
    elif out:
        plt.savefig(out, dpi=1000)

    return ax


def plot_spectra(spectra, wavelengths,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 title_fontsize=8,
                 xlabel_fontsize=8,
                 ylabel_fontsize=8,
                 tick_fontsize=6,
                 ylim=None,
                 colors=None,
                 linestyles=None,
                 linewidth=None,
                 labels=None,
                 unmeasured_intervals=None,
                 ax=None,
                 show=True,
                 show_axes=True,
                 out=None,
                 bbox_to_anchor=None,
                 transparent=False,
                 figsize=None):

    n_spectra = spectra.shape[0]

    if not ax:
        f = plt.figure(figsize=figsize)
        ax = plt.gca()

    if colors is None:
        colors = [None] * n_spectra

    if linestyles is None:
        linestyles = ['-'] * n_spectra

    if labels is not None:

        # only retain the first occurrence of each label, set the rest to None
        # this avoids redundant labels in the legend
        unique_labels = np.unique(labels)

        for l, label in enumerate(labels):

            if label in unique_labels:
                unique_labels = unique_labels[unique_labels != label]
            else:
                labels[l] = None

    # don't plot unmeasured intervals of the spectrum by introducing nan values in the spectra and/or wavelengths
    if unmeasured_intervals is not None:

        if unmeasured_intervals == 'auto':

            diff = np.diff(wavelengths)
            iqr = np.quantile(diff, 0.75) - np.quantile(diff, 0.25)
            ind = np.where(diff > diff.mean() + 1.5 * iqr)[0]
            ext = 0

            for i in ind:

                spectra = np.concatenate((spectra[:, :i + 1 + ext],
                                          np.array([np.nan] * spectra.shape[0]).reshape(-1, 1),
                                          spectra[:, i + 1 + ext:]),
                                         axis=1)
                wavelengths = np.concatenate((wavelengths[:i + 1 + ext],
                                              np.array([np.nan]),
                                              wavelengths[i + 1 + ext:]))
                ext += 1

        else:

            for ui in unmeasured_intervals:

                xmin = ui[0]
                xmax = ui[1]
                con = np.logical_and(wavelengths > xmin, wavelengths < xmax)

                if np.any(con):

                    ind = np.where(con)[0]
                    wavelengths[ind] = np.nan

                elif np.any(wavelengths < xmin) and np.any(wavelengths > xmax):

                    ind = np.where(wavelengths < xmin)[0][-1]
                    spectra = np.concatenate((spectra[:, :ind + 1],
                                              np.array([np.nan] * spectra.shape[0]).reshape(-1, 1),
                                              spectra[:, ind + 1:]),
                                             axis=1)
                    wavelengths = np.concatenate((wavelengths[:ind + 1],
                                                  np.array([np.nan]),
                                                  wavelengths[ind + 1:]))

    # plot the spectra with correct color and label
    for s in range(spectra.shape[0]):

        spectrum = spectra[s, :]

        label = None
        if labels is not None:
            label = labels[s]

        ax.plot(wavelengths, spectrum.squeeze(),
                color=colors[s],
                linestyle=linestyles[s],
                linewidth=linewidth,
                label=label,
                zorder=0)

    if show_axes:

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('0.5')
        ax.spines['bottom'].set_color('0.5')
        ax.tick_params(axis='x',
                       colors='0.5',
                       labelsize=tick_fontsize)
        ax.tick_params(axis='y',
                       colors='0.5',
                       labelsize=tick_fontsize)

    else:

        ax.axis('off')

    if labels is not None:
        ax.legend(loc='right',
                  bbox_to_anchor=bbox_to_anchor,
                  bbox_transform=ax.transAxes,
                  fontsize=11)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, color='0.2')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, color='0.2')

    if ylim:
        ax.set_ylim(ylim)

    if out:
        plt.savefig(out,
                    dpi=1000,
                    transparent=transparent,
                    bbox_inches='tight')
    elif show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_spectra_imagepc(image, spectra,
                         n_components=3,
                         figsize=None,
                         show=True,
                         out=None):

    shape = image.shape
    image = copy.deepcopy(image)
    image = image.reshape((shape[0] * shape[1], shape[2]))
    spectra = copy.deepcopy(spectra)

    # fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(image)

    # transform image and spectra
    image_trans = pca.transform(image)
    spectra_trans = pca.transform(spectra)

    # get binary PC combinations
    combs = list(combinations(range(n_components), 2))
    ncombs = len(combs)

    f = plt.figure(figsize=figsize)

    # plot spectra on feature space for each binary PC combination
    for C, comb in enumerate(combs):

        ax = plt.subplot(1, ncombs, C + 1)

        pcx = comb[0]
        pcy = comb[1]

        # make 2D histogram
        xhist = image_trans[:, pcx].squeeze()
        yhist = image_trans[:, pcy].squeeze()

        ax.hist2d(xhist, yhist,
                  bins=(50, 50),
                  cmap='Greens_r',
                  cmin=1)

        # plot spectra
        xscat = spectra_trans[:, pcx].squeeze()
        yscat = spectra_trans[:, pcy].squeeze()
        ax.scatter(xscat, yscat, s=1.5, color='red', alpha=0.5)
        # ax.set_aspect('equal')

        # axis labels
        ax.set_xlabel('PC{}'.format(pcx + 1), fontsize=15)
        ax.set_ylabel('PC{}'.format(pcy + 1), fontsize=15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('0.5')
        ax.spines['bottom'].set_color('0.5')
        ax.tick_params(axis='x',
                       colors='0.5')
        ax.tick_params(axis='y',
                       colors='0.5')

    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=1000)
    elif show:
        plt.show()
