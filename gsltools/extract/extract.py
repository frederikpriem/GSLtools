import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.ndimage import convolve
from scipy.spatial.distance import pdist, cdist, squareform

from gsltools.denoise import gaussian_noise

"""
This module handles the automated extraction of endmember spectra from imagery.
"""


def iterative_spectral_distancing(image, distance_measure, distance_threshold_purity, distance_threshold_redundancy,
                                  normalize_image=False,
                                  return_candidate_indices=False,
                                  nodata=None):

    """
    Iterative Spectral Distancing

    Extracts endmember spectra from an image.

    Args:
        image: 3D array of shape (rows, cols, bands B) and type float
        distance_measure: function object, distance measure used, see submodule 'distance' for examples possible
            measures
        distance_threshold_purity: float, distance threshold used to find spectrally pure patches
        distance_threshold_redundancy: float, distance threshold used to assess redundancy among candidates
        normalize_image: bool, whether to brightness normalize each pixel in the image
        return_candidate_indices: bool, whether to return indices of all candidate EM
        nodata: float, no data value to exclude pixels as candidate endmembers

    Returns:
        em: 2D array of shape (endmembers E, B) and type float, the endmember spectra
        em_rows: 1D array of size E and type int, row indices of the endmember spectra
        em_cols: 1D array of size E and type int, column indices of the endmember spectra
    """

    image_orig = copy.deepcopy(image)
    image = copy.deepcopy(image)
    rows, cols, bands = image.shape
    image_array = image.reshape(rows * cols, bands)
    loc = np.arange(image_array.shape[0])  # this vector is used to trace the location in the original image

    # normalize image
    if normalize_image:
        image = image / np.expand_dims(image.mean(axis=2), axis=2)
        image_array = image_array / np.expand_dims(image_array.mean(axis=1), axis=1)

    # get the center of mass of the image feature space
    center = image_array.mean(axis=0)

    # add pads to the horizontal dimensions of the image
    image_pad = np.pad(image, 1,
                       mode='constant',
                       constant_values=0)
    image_pad = image_pad[:, :, 1:-1]

    # _check if neighbouring pixel spectra are similar to the central pixel spectrum
    shifts = np.arange(-1, 2)
    check = []

    for v in shifts:

        for h in shifts:

            con1 = v != 0 or h != 0
            con2 = True

            if con1 and con2:

                image_shift = np.roll(image_pad, (v, h), axis=(0, 1))
                image_shift = image_shift[1:-1, 1:-1]
                image_shift_array = image_shift.reshape(rows * cols, bands)
                dist = distance_measure(image_array, image_shift_array).reshape(-1, 1)
                check.append(dist < distance_threshold_purity)

    # if all neighbouring spectra are similar, retain the central pixel as a candidate
    # remove no data pixels from candidates if applicable
    check = np.hstack(check)
    check = np.all(check, axis=1)
    if nodata is not None:
        check *= np.all(image_array != nodata, axis=1)
    cand_ind = np.where(check)[0]
    image_array = image_array[cand_ind, :]
    loc = loc[cand_ind]
    
    # retain endmembers
    em_loc = []
    stop = False

    # calculate distance between image feature space center and remaining candidates
    dist_center = distance_measure(image_array, center)

    while not stop:

        # retain the spectrum located furthest away as an endmember
        ind_em = np.argmax(dist_center)
        em = image_array[ind_em, :]
        em_loc.append(loc[ind_em])

        # remove the endmember and similar spectra from the array to avoid redundancy
        dist = distance_measure(em, image_array)
        del_ind = np.where(dist < distance_threshold_redundancy)[0]
        image_array = np.delete(image_array, del_ind, 0)
        loc = np.delete(loc, del_ind)
        dist_center = np.delete(dist_center, del_ind)

        # stop when all candidate spectra are processed
        if image_array.shape[0] == 0:
            stop = True

    em_rows, em_cols = np.unravel_index(em_loc, (rows, cols))
    em = image_orig[em_rows, em_cols, :]

    output = (em, em_rows, em_cols)
    if return_candidate_indices:
        cand_rows, cand_cols = np.unravel_index(cand_ind, (rows, cols))
        output = (em, em_rows, em_cols, cand_rows, cand_cols)

    return output


def synthetic_image(image, num_em,
                    conv_footprint=None,
                    clust_seed=None,
                    noise_seed=None,
                    snr_db=30,
                    min_value=0,
                    max_value=1
                    ):

    # define the default convolution footprint
    if conv_footprint is None:
        conv_footprint = [[0.2, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 0.2]]

    # reshape the image to a 2D array and cast it to float
    image = image.astype(float)
    shape = image.shape
    image = image.reshape(shape[0] * shape[1], -1)

    # fit the cluster on the image and construct the synthetic endmembers
    kmeans = KMeans(n_clusters=num_em,
                    random_state=clust_seed)
    kmeans.fit(image)
    synth_em = np.array(kmeans.cluster_centers_)

    # apply the cluster labels and corresponding synthetic endmembers to produce the synthetic image
    cluster = kmeans.predict(image)
    cluster = cluster.reshape(shape[0], shape[1], 1)
    synth_image = synth_em[cluster, :]
    synth_image = synth_image.reshape(shape)

    # apply convolution on the synthetic image to simulate mixing on the edges between clusters
    conv_footprint = np.array(conv_footprint)
    conv_footprint /= conv_footprint.sum(axis=None)
    footprint = np.expand_dims(conv_footprint, axis=2)
    synth_image = convolve(synth_image, footprint)

    # add Gaussian noise with a specified SNR to the synthetic image to further enhance its realism
    noise = gaussian_noise(synth_image, snr_db,
                           noise_seed=noise_seed)
    synth_image += noise

    # cut off values out of range
    if min_value:
        synth_image[synth_image < min_value] = min_value
    if max_value:
        synth_image[synth_image > max_value] = max_value

    return synth_image, synth_em


def em_coverage(em_ref, em_pred, dist_meas):

    dist_ap = cdist(em_ref, em_pred, dist_meas)
    c = np.mean(dist_ap.min(axis=1))

    return c


def em_redundancy(em_pred, dist_meas):

    dist_pp = pdist(em_pred, dist_meas)
    dist_pp = squareform(dist_pp)
    r = np.tril(dist_pp, -1)
    r = r.mean()

    return r
