import numpy as np
from tqdm import tqdm
import copy
import sys
from gultools.spectral_distance import *
from gultools.validation import kappa_coefficient
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


def comprehensive(spectra,
                  distance_measure=l1_sam,
                  distance_threshold=0.05,
                  return_indices=False):

    center = spectra.mean(axis=0)
    dist_center = distance_measure(spectra, center)
    loc = np.arange(dist_center.size)
    retained_spectra = []
    ret_loc = []
    stop = False

    while True:

        # stop when all library spectra are either retained or removed
        if spectra.shape[0] == 0:
            break

        # retain the spectrum located furthest away from the center
        ind = np.argmax(dist_center)
        ret = copy.deepcopy(spectra[ind, :])
        retained_spectra.append(ret.reshape(1, -1))
        ret_loc.append(loc[ind])

        # remove the retained spectrum
        spectra = np.delete(spectra, ind, axis=0)
        loc = np.delete(loc, ind)
        dist_center = np.delete(dist_center, ind)

        # stop when all library spectra are either retained or removed
        if spectra.shape[0] == 0:
            break

        # Remove similar spectra from the library to avoid redundancy
        dist = distance_measure(ret, spectra)
        del_ind = np.where(dist < distance_threshold)[0]
        if del_ind.size > 0:
            spectra = np.delete(spectra, del_ind, 0)
            loc = np.delete(loc, del_ind)
            dist_center = np.delete(dist_center, del_ind)

    spectra = np.concatenate(retained_spectra, axis=0)

    output = spectra
    if return_indices:
        ret_loc = np.array(ret_loc)
        output = (spectra, ret_loc)

    return output


def pairwise(spectra,
             distance_measure=auc_sam,
             distance_threshold=0.1,
             return_indices=False):
    
    # produce a square matrix containing distances between each spectrum pair
    dist = pdist(spectra, distance_measure)
    dist = squareform(dist)

    # set the diagonal element to the highest finite positive value
    dist[np.arange(dist.shape[0]), np.arange(dist.shape[0])] = np.finfo(float).max

    stop = False
    indices = np.arange(spectra.shape[0])

    while not stop:

        # in case of excessive spectral similarity, remove the most redundant spectrum,
        # i.e. having the lowest row/column average distance
        if np.any(dist < distance_threshold):

            ind = np.unravel_index(np.argmin(dist), dist.shape)
            spec1 = ind[0]
            spec2 = ind[1]
            avg1 = dist[spec1, :].mean()
            avg2 = dist[spec2, :].mean()

            if avg1 > avg2:

                indices = np.delete(indices, spec2)
                dist = np.delete(dist, spec2, axis=0)
                dist = np.delete(dist, spec2, axis=1)

            elif avg1 < avg2:

                indices = np.delete(indices, spec1)
                dist = np.delete(dist, spec1, axis=0)
                dist = np.delete(dist, spec1, axis=1)

            else:

                del_ind = np.array([spec1, spec2])
                indices = np.delete(indices, del_ind)
                dist = np.delete(dist, del_ind, axis=0)
                dist = np.delete(dist, del_ind, axis=1)
        else:

            stop = True

    library_optimized = spectra[indices, :]

    if not return_indices:
        return library_optimized
    else:
        return library_optimized, indices


def ies(spectra, labels, classifier,
        return_indices=False):

    # find the pair of spectra that best models all spectra
    kappas = []
    combs = combinations(labels, 2)

    for comb in combs:

        spectra_temp = spectra[np.array(comb), :]
        labels_temp = np.concatenate((labels[comb[0]], labels[comb[1]]))
        model = classifier.fit(spectra_temp, labels_temp)
        predict = model.predict(spectra)
        kappas.append(kappa_coefficient(predict, labels))

    kappas = np.array(kappas)
    ind_max = np.argmax(kappas)
    kappa_prev = copy.deepcopy(kappas[ind_max])
    retained = np.array(combs[ind_max])
    candidates = np.arange(labels.size)
    candidates = np.delete(candidates, retained)

    while True:

        addition = False
        removal = False

        # check which new spectrum adds the highest improvement to the model, if any
        kappas = []

        for cand in candidates:

            spectra_temp = spectra[retained, :]
            spectra_temp = np.concatenate((spectra_temp, spectra[cand, :].reshape(1, -1)), axis=0)
            labels_temp = labels[retained]
            labels_temp = np.concatenate((labels_temp, labels[cand]))
            model = classifier.fit(spectra_temp, labels_temp)
            predict = model.predict(spectra)
            kappas.append(kappa_coefficient(predict, labels))

        kappas = np.array(kappas)
        ind_max = np.argmax(kappas)
        kappa_max = kappas[ind_max]

        if kappa_max > kappa_prev:

            retained = np.concatenate((retained, [candidates[ind_max]]))
            candidates = np.delete(candidates, ind_max)
            kappa_prev = copy.deepcopy(kappa_max)
            addition = True

        # check which retained spectrum yields the highest improvement to the model when removed, if any
        kappas = []

        for ret in retained:

            retained_temp = np.delete(retained, ret)
            spectra_temp = spectra[retained_temp, :]
            labels_temp = labels[retained_temp]
            model = classifier.fit(spectra_temp, labels_temp)
            predict = model.predict(spectra)
            kappas.append(kappa_coefficient(predict, labels))

        kappas = np.array(kappas)
        ind_max = np.argmax(kappas)
        kappa_max = kappas[ind_max]

        if kappa_max > kappa_prev:

            candidates = np.concatenate((candidates, [retained[ind_max]]))
            retained = np.delete(retained, ind_max)
            kappa_prev = copy.deepcopy(kappa_max)
            removal = True

        # stop the iteration if no spectra were added to or removed from the set of retained spectra
        if not (addition or removal):
            break

    output = spectra[retained, :]
    if return_indices:
        output = spectra[retained, :], retained

    return output


def image_comprehensive(spectra, image,
                        n_eigenvector=None,
                        thres_perc_variance=None):

    """
    This function essentially calculates the Euclidean distance between library spectra and the subspace formed by the
    first n eigenvectors of the covariance matrix of an image. The technique is described in detail in 'MUSIC-CSR:
    Hyperspectral Unmixing via Multiple Signal Classification and Collaborative Sparse Regression' by Marian-Daniel
    Iordache et al. (2014).

    Note that this is a very simplified implementation of MUSIC that leaves out certain important aspects,
    e.g. determining the optimal image subspace. There is no guarantee that it performs as described in
    the original paper.

    :param spectra: 2D array of shape (n_spectra, bands)
    :param image: 3D array of shape (rows, columns, bands)
    :param n_eigenvector: integer, the first n eigenvectors of the image to be retained, must be equal to or smaller
    than the number of bands in image
    :return: 1D array containing MUSIC distances of each spectra spectrum relative to the image subspace
    """

    if spectra.shape[1] != image.shape[2]:
        raise ValueError('number of bands in library and image must be equal')

    if not n_eigenvector:
        n_eigenvector = spectra.shape[1]

    # check that n_eigenvectors doesn't exceed the number of image bands, adjust if needed
    if n_eigenvector > spectra.shape[1]:
        n_eigenvector = spectra.shape[1]

    shape = image.shape
    image = copy.deepcopy(image)
    image = image.reshape((shape[0] * shape[1], shape[2]))
    spectra = copy.deepcopy(spectra)

    # brightness normalize the image and spectra
    image /= image.sum(axis=1).reshape(-1, 1)
    spectra /= spectra.sum(axis=1).reshape(-1, 1)

    # get eigenvalues and eigenvectors of the image-derived covariance matrix, i.e. covariance between image bands
    cov = np.cov(image, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov)

    # calculate % variance described by eigenvectors
    perc_variance = eigval / np.sum(eigval)

    # sort eigenvectors from high to low
    ind = np.argsort(perc_variance)[::-1]
    eigvect = eigvect[:, ind]

    if thres_perc_variance:
        perc_variance = perc_variance[ind]
        cs_perc_variance = np.cumsum(perc_variance)
        n_eigenvector = np.where(cs_perc_variance >= thres_perc_variance)[0][0]

    """
    In the original MUSIC paper by Iordache, a technique called HySime, itself based on a paper titled 'Hyperspectral
    Subspace Identification' by Bioucas-Dias & Nascimento (2008), is used to determine the optimal number of retained
    eigenvectors. See the respective papers for more info.
    """
    eigvect = eigvect[:, :n_eigenvector]

    # calculate Euclidean distances between spectra and the hyperplane formed by the retained eigenvectors
    p = np.diag(np.ones(image.shape[1])) - np.dot(eigvect, eigvect.T)
    dist = np.sum((np.dot(p, spectra.T) ** 2), axis=0) ** 0.5
    dist /= np.sum(spectra ** 2, axis=1).squeeze() ** 0.5

    return dist


def image_pairwise(spectra, image,
                   distance_measure=l1_sam,
                   return_mindist=False,
                   mask=None):

    image = copy.deepcopy(image)
    rows, cols, bands = image.shape
    image = image.reshape(rows * cols, bands)

    mindist = np.ones(rows * cols).reshape(-1, 1)
    labels = np.empty(rows * cols).reshape(-1, 1)

    for s, spec in tqdm(tuple(enumerate(spectra))):

        temp = distance_measure(spec, image).reshape(-1, 1)
        labels[temp < mindist] = s
        mindist = np.minimum(temp, mindist)

    labels = labels.reshape(rows, cols)
    mindist = mindist.reshape(rows, cols)

    if mask is not None:
        labels[mask == 0] = -1
        mindist[mask == 0] = 0

    uni, cnt = np.unique(labels, return_counts=True)
    total = float(cnt.sum())

    if -1 in uni:
        ind = np.where(uni == -1)[0][0]
        uni = np.delete(uni, ind)
        cnt = np.delete(cnt, ind)

    frac = np.zeros(cnt.shape[0])

    for s in range(cnt.shape[0]):

        if s in uni:
            frac[s] = cnt[s] / total

    if return_mindist:
        output = (frac, mindist)
    else:
        output = frac

    return output
