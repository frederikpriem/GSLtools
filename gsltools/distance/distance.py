import numpy as np
import copy
import gsltools._check as check

eps = np.finfo(float).eps

"""
This module contains several conventional and hybrid distance measures that assess (dis)similarity between pairs of
spectra.
"""


def _check_dims_values(array1, array2):

    check.is_array_like(array1,
                        dtype=float,
                        object_name='array1')

    check.is_array_like(array2,
                        dtype=float,
                        object_name='array2')

    array1 = array1.astype(np.float32)
    array2 = array2.astype(np.float32)

    if len(array1.shape) == 1:
        array1 = array1.reshape(1, -1)

    if len(array2.shape) == 1:
        array2 = array2.reshape(1, -1)

    if len(array1.shape) > 2 or len(array2.shape) > 2:
        raise Exception("arrays must be 1D or 2D")

    if array1.shape[1] != array2.shape[1]:
        raise Exception("the number of bands must be equal in both arrays")

    if (array1.shape[0] != array2.shape[0]) and ((array1.shape[0] > 1) and (array2.shape[0] > 1)):
        raise Exception("the number of rows in both arrays must be equal or one of the arrays must have a single row")

    if (np.any(array1 < 0, axis=None)) or (np.any(array2 < 0, axis=None)):
        raise Exception("reflectance values can't be negative")

    if (np.any(array1 > 1, axis=None)) or (np.any(array2 > 1, axis=None)):
        raise Exception("reflectance values can't be higher than 1")

    return array1, array2


"""
Conventional distance measures
"""


def mean_brightness_difference(array1, array2):

    """
    Mean Brightness Difference

    Sensitive to brightness
    Insensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist = np.abs(array1.sum(axis=1) - array2.sum(axis=1)) / array1.shape[1]

    return dist.squeeze()


def l1_distance(array1, array2):

    """
    L1 distance

    Sensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist = np.sum(np.abs(array1 - array2), axis=1)

    return dist.squeeze()


def l2_distance(array1, array2):

    """
    L2 distance

    Sensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist = np.sum((array1 - array2) ** 2, axis=1)**0.5

    return dist.squeeze()


def correlation_distance(array1, array2):

    """
    Correlation distance

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)

    array1 = copy.deepcopy(array1)
    array2 = copy.deepcopy(array2)
    array1 -= array1.mean(axis=1)
    array2 -= array2.mean(axis=1)
    nom = np.sum(array1 * array2, axis=1)
    denom = (np.sum(array1**2, axis=1) * np.sum(array2**2, axis=1))**0.5
    dist = 1 - nom / denom

    return dist.squeeze()


def cosine_distance(array1, array2):

    """
    Cosine distance

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    nom = np.sum(array1 * array2, axis=1)
    denom1 = np.sum(array1**2, axis=1)**0.5
    denom2 = np.sum(array2**2, axis=1)**0.5
    dist = nom / (eps + denom1 * denom2)
    dist[dist > 1] = 1
    dist[dist < -1] = -1

    return dist.squeeze()


def spectral_angle(array1, array2):

    """
    Spectral angle

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    dist = cosine_distance(array1, array2)
    dist = np.arccos(dist)

    return dist.squeeze()


def spectral_information_divergence(array1, array2):

    """
    Spectral Information Divergence

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    array1 = copy.deepcopy(array1)
    array2 = copy.deepcopy(array2)
    array1[array1 == 0] = eps
    array2[array2 == 0] = eps
    array1 = array1 / (np.expand_dims(array1.sum(axis=1), 1))
    array2 = array2 / (np.expand_dims(array2.sum(axis=1), 1))
    term1 = array1 * np.log(array1 / array2)
    term2 = array2 * np.log(array2 / array1)
    dist = np.sum(term1 + term2, axis=1)

    return dist.squeeze()


def mahalanobis_distance(array1, array2):

    """
    Mahalanobis distance

    Note that this implementation uses the Mahalanobis distance form that compares 1D probability distributions

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    array_all = np.concatenate((array1, array2), axis=0)
    cov = np.cov(array_all, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    diff = array1 - array2
    dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)**0.5

    return dist.squeeze()


def bhattacharyya_distance(array1, array2):

    """
    Bhattacharyya distance

    Note that this implementation uses the Bhattacharyya distance form that compares 1D probability distributions

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    array1 = copy.deepcopy(array1)
    array2 = copy.deepcopy(array2)
    array1 = array1 / (np.expand_dims(array1.sum(axis=1), 1) + eps) + eps
    array2 = array2 / (np.expand_dims(array2.sum(axis=1), 1) + eps) + eps
    dist = - np.log(np.sum((array1 * array2)**0.5, axis=1))
    dist[dist < 0] = 0

    return dist.squeeze()


def jeffries_matusita_distance(array1, array2):

    """
    Jeffries-Matusita Distance

    Note that this implementation uses the Jeffries-Matusita distance form that compares 1D probability distributions

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist = bhattacharyya_distance(array1, array2)
    dist = (2 * (1 - np.exp(-dist)))**0.5

    return dist.squeeze()


"""
Hybrid distance measures
"""


def mean_brightness_difference_spectral_angle(array1, array2):

    """
    Mean Brightness Difference - Spectral Angle

    Sensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist1 = mean_brightness_difference(array1, array2)
    dist2 = spectral_angle(array1, array2)
    dist = (1 + dist1) * (1 + np.sin(dist2)) - 1

    return dist.squeeze()


def spectral_information_divergence_spectral_angle(array1, array2):

    """
    Spectral Information Divergence - Spectral Angle

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist1 = spectral_information_divergence(array1, array2)
    dist2 = spectral_angle(array1, array2)
    dist = dist1 * np.sin(dist2)

    return dist.squeeze()


def jeffries_matusita_distance_spectral_angle(array1, array2):

    """
    Jeffries-Matusita Distance - Spectral Angle

    Note that this implementation uses the Jeffries-Matusita distance form that compares 1D probability distributions

    Insensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist1 = jeffries_matusita_distance(array1, array2)
    dist2 = spectral_angle(array1, array2)
    dist = dist1 * np.sin(dist2)

    return dist.squeeze()


def spectral_similarity(array1, array2):

    """
    Spectral Similarity

    Sensitive to brightness
    Sensitive to spectral shape

    The input arrays must have the same number of columns (bands)
    The input arrays must have the same number of rows (spectra) or one of the arrays must have a single row

    Args:
        array1: 2D reflectance array of type float, rows = spectra, cols = bands
        array2: 2D reflectance array of type float, rows = spectra, cols = bands

    Returns:
        dist: 1D distance array of type float with size = max(array1.shape[0], array2.shape[0])
    """

    array1, array2 = _check_dims_values(array1, array2)
    dist1 = l2_distance(array1, array2) / array1.shape[1]**0.5
    dist2 = correlation_distance(array1, array2)
    dist = (dist1**2 + (1-dist2)**2)**0.5

    return dist.squeeze()

