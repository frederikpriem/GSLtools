import numpy as np
import copy
from gultools.spectral_distance import *
import matplotlib.pyplot as plt


def endmember_extraction(image,
                         brightness_normalize=False,
                         return_candidate_indices=False,
                         distance_measure=l1_sam,
                         distance_threshold_filter=0.01,
                         distance_threshold_redundancy=0.05):

    image_orig = copy.deepcopy(image)
    image = copy.deepcopy(image)
    rows, cols, bands = image.shape
    image_array = image.reshape(rows * cols, bands)
    loc = np.arange(image_array.shape[0])  # this vector is used to trace the location in the original image

    # brightness normalize image
    if brightness_normalize:
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

            if v != 0 or h != 0:

                image_shift = np.roll(image_pad, (v, h), axis=(0, 1))
                image_shift = image_shift[1:-1, 1:-1]
                image_shift_array = image_shift.reshape(rows * cols, bands)
                dist = distance_measure(image_array, image_shift_array).reshape(-1, 1)
                check.append(dist < distance_threshold_filter)

    # if all neighbouring spectra are similar, retain the central pixel as a candidate
    check = np.hstack(check)
    cand_ind = np.where(np.all(check, axis=1))[0]
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
