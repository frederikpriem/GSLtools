import numpy as np
import copy
from gsltools.distance import *
import matplotlib.pyplot as plt

"""
This module handles automated endmember extract from imagery.
"""


def endmember_extraction(image,
                         normalize_image=False,
                         return_candidate_indices=False,
                         distance_measure=auc_sam,
                         distance_threshold_filter=0.0001,
                         distance_threshold_redundancy=0.0005,
                         filter_type='Moore',
                         normalize_distance=True,
                         nodata=None):

    """
    extracts EM spectra from image in two steps:
    1. Find pure patches in the image using neighbourhood analysis. The central pixel is compared with all its
     neighbouring pixels. If all pixels are similar, the central pixel in these neighbourhood becomes a candidate EM.
    2. Iteratively retain candidate EM's located farthest away from the center of mass of all candidates. Remove spectra
     located near retained EM at the end of each iteration. Repeat until all candidates are either retained or removed.
    :param image: 3D-array of shape (rows, cols, bands)
    :param normalize_image: bool, whether to brightness normalize each pixel in the image
    :param return_candidate_indices: bool, whether to return indices of all candidate EM
    :param distance_measure: object, distance measure used (see module 'spectral distance' for possible measures)
    :param distance_threshold_filter: float, distance threshold used to find pure patches
    :param distance_threshold_redundancy: float, distance threshold used to assess redundancy among candidates
    :param filter_type: type of neighbourhood used to perform spatial filter, either 'Neumann' or 'Moore'
    :param normalize_distance: bool, whether to normalize distances
    :param nodata: float, no data value
    :return:
        em: endmember spectra (2D-array, float, shape (m endmembers, b bands))
        em_rows: endmember rows (1D-array, size m, int),
        em_cols: endmember columns (1D-array, size m, int)
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

    # check if neighbouring pixel spectra are similar to the central pixel spectrum
    shifts = np.arange(-1, 2)
    check = []

    for v in shifts:

        for h in shifts:

            con1 = v != 0 or h != 0

            # apply the chosen neighbourhood filter
            if filter_type == 'Moore':
                con2 = True
            elif filter_type == 'Neumann':
                con2 = np.abs(v) == 0 or np.abs(h) == 0

            if con1 and con2:

                image_shift = np.roll(image_pad, (v, h), axis=(0, 1))
                image_shift = image_shift[1:-1, 1:-1]
                image_shift_array = image_shift.reshape(rows * cols, bands)
                dist = distance_measure(image_array, image_shift_array,
                                        norm=normalize_distance).reshape(-1, 1)
                check.append(dist < distance_threshold_filter)

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
        dist = distance_measure(em, image_array,
                                norm=normalize_distance)
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


def find_match(spectra, ref,
               distance_measure=auc_sam,
               distance_threshold=0.01,
               retention='closest',
               normalize_distance=True,
               return_new=False,
               return_weights=False
               ):

    """
    finds spectral match between set of spectra and reference library
    :param spectra: 2D-array of floats with shape (n target spectra, b bands), target spectra to be matched
    :param ref: 2D-array of floats with shape (r reference spectra, b bands), reference spectra
    :param distance_measure: distance measure used to match spectra
    :param distance_threshold: distance threshold used to determine (dis)similarity
    :param retention: either 'closest' or 'all'. If 'closest' only the closest match is retained. If 'all', all
    reference spectra located within the defined distance threshold are retained, and weights are defined for each match
    in the reference library based on its proximity to the target spectrum.
    :param normalize_distance: bool, whether to normalize distances
    :param return_new: bool, whether to return indices of target spectra without match in the reference library
    :param return_weights: bool, whether to return weights, only applicable if retention='all'
    :return:
        matched: indices of matched spectra (list of int),
        match_ref: index/indices of reference spectrum/spectra matched to target spectrum (list of int or list of
        1D-array of int), see retention
    """

    matched = []
    match_ref = []
    new = []
    weights = []

    for s, spec in enumerate(spectra):

        dist = distance_measure(spec, ref,
                                norm=normalize_distance)
        if np.all(dist >= distance_threshold):
            new.append(s)
        else:
            matched.append(s)
            if retention == 'closest':
                match_ref.append(np.argmin(dist))
            elif retention == 'all':
                match_ref.append(np.where(dist < distance_threshold)[0])
                w = dist[dist < distance_threshold]
                w = distance_threshold - w
                weights.append(w / w.sum())

    output = (matched, match_ref)
    if return_weights:
        output = (matched, match_ref, weights)
        if return_new:
            output = (matched, match_ref, weights, new)
    elif return_new:
        output = (matched, match_ref, new)

    return output
