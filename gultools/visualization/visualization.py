import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
import copy


def plot_image(image,
               vmin=0,
               vmax=1,
               ax=None,
               show=True,
               origin='upper',
               axis=False,
               out=None):

    image = copy.deepcopy(image)
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


def plot_spectra(library, wavelengths,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 colors=None,
                 labels=None,
                 white_spaces=None,
                 ax=None,
                 show=True,
                 out=None):

    n_spectra = library.shape[0]
    ymax = library.max(axis=None)

    if not ax:
        ax = plt.gca()

    if not colors:
        colors = [None] * n_spectra

    if labels is not None:

        # only retain the first occurrence of each label, set the rest to None
        # this is done to avoid redundant labels in the legend
        unique_labels = np.unique(labels)

        for l, label in enumerate(labels):

            if label in unique_labels:
                unique_labels = unique_labels[unique_labels != label]
            else:
                labels[l] = None

    # plot the spectra with correct color and label
    for s, spectrum in enumerate(library):

        ax.plot(wavelengths, spectrum,
                color=colors[s],
                label=labels[s])

    if white_spaces is not None:

        for ws in white_spaces:

            xmin = ws[0]
            xmax = ws[1]
            x_ws = [xmin, xmax, xmax, xmin, xmin]
            y_ws = [0, 0, ymax, ymax, 0]
            ax.fill(x_ws, y_ws, 'white')

    if labels is not None:
        ax.legend()

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if out:
        plt.tight_layout()
        plt.savefig(out, dpi=1000)
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
