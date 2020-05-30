import numpy as np
import copy

eps = np.finfo(float).eps


def check_dims_values(array1, array2):

    if len(array1.shape) == 1:
        array1 = array1.reshape(1, -1)

    if len(array2.shape) == 1:
        array2 = array2.reshape(1, -1)

    if array1.shape[1] != array2.shape[1]:
        raise ValueError("the number of bands must be equal in both arrays")

    if (len(array1.shape) > 2) or (len(array2.shape) > 2):
        raise ValueError("arrays can't have more than 2 dimensions")

    if (array1.shape[0] == 0) or (array2.shape[0] == 0):
        raise ValueError("arrays can't be empty")

    if (array1.shape[0] != array2.shape[0]) and ((array1.shape[0] > 1) and (array2.shape[0] > 1)):
        raise ValueError("the number of rows in both arrays must be equal or one of the arrays must have a single row")

    if (np.any(array1 < 0, axis=None)) or (np.any(array2 < 0, axis=None)):
        raise ValueError("reflectance values can't be negative")

    if (np.any(array1 > 1, axis=None)) or (np.any(array2 > 1, axis=None)):
        raise ValueError("reflectance values can't be higher than 1")

    array1 = array1.astype(np.float32)
    array2 = array2.astype(np.float32)

    return array1, array2


def normalize_distance(distance_measure, distance, n_dims):

    signal = np.zeros(n_dims)
    signal[0:n_dims + 1:2] = 1
    signal_complement = 1.0 - signal
    signal = signal.reshape(1, -1)
    signal_complement = signal_complement.reshape(1, -1)
    max_dist = distance_measure(signal, signal_complement)
    distance_norm = distance / max_dist

    return distance_norm


def auc(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _auc(array1, array2):
        dist = np.abs(array1.sum(axis=1) - array2.sum(axis=1))
        return dist

    dist = _auc(array1, array2)
    if norm:
        dist /= array1.shape[1]

    return dist.squeeze()


def l1(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _l1(array1, array2):
        dist = np.sum(np.abs(array1 - array2), axis=1)
        return dist

    dist = _l1(array1, array2)
    if norm:
        dist = normalize_distance(_l1, dist, array1.shape[1])

    return dist.squeeze()


def l2(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _l2(array1, array2):
        dist = np.sum((array1 - array2) ** 2, axis=1)**0.5
        return dist

    dist = _l2(array1, array2)
    if norm:
        dist = normalize_distance(_l2, dist, array1.shape[1])

    return dist.squeeze()


def correlation(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    array1 = copy.deepcopy(array1)
    array2 = copy.deepcopy(array2)
    array1 -= array1.mean(axis=1)
    array2 -= array2.mean(axis=1)
    nom = np.sum(array1 * array2, axis=1)
    denom = (np.sum(array1**2, axis=1) * np.sum(array2**2, axis=1))**0.5
    dist = nom / denom

    return dist.squeeze()


def sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _sam(array1, array2):
        nom = np.sum(array1 * array2, axis=1)
        denom1 = np.sum(array1**2, axis=1)**0.5
        denom2 = np.sum(array2**2, axis=1)**0.5
        dist = nom / (eps + denom1 * denom2)
        dist[dist > 1] = 1
        dist[dist < -1] = -1
        dist = np.arccos(dist)
        return dist

    dist = _sam(array1, array2)
    if norm:
        dist = normalize_distance(_sam, dist, array1.shape[1])

    return dist.squeeze()


def sid(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _sid(array1, array2):
        array1 = copy.deepcopy(array1)
        array2 = copy.deepcopy(array2)
        array1[array1 == 0] = eps
        array2[array2 == 0] = eps
        array1 = array1 / (np.expand_dims(array1.sum(axis=1), 1))
        array2 = array2 / (np.expand_dims(array2.sum(axis=1), 1))
        term1 = array1 * np.log(array1 / array2)
        term2 = array2 * np.log(array2 / array1)
        dist = np.sum(term1 + term2, axis=1)
        return dist

    dist = _sid(array1, array2)
    if norm:
        dist = normalize_distance(_sid, dist, array1.shape[1])

    return dist.squeeze()


def mahalanobis(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _mahalanobis(array1, array2):
        array_all = np.concatenate((array1, array2), axis=0)
        cov = np.cov(array_all, rowvar=False)
        cov_inv = np.linalg.inv(cov)
        diff = array1 - array2
        dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)**0.5
        return dist

    dist = _mahalanobis(array1, array2)
    if norm:
        dist = normalize_distance(_mahalanobis, dist, array1.shape[1])

    return dist.squeeze()


def bhattacharyya(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _bhattacharyya(array1, array2):
        array1 = copy.deepcopy(array1)
        array2 = copy.deepcopy(array2)
        array1 = array1 / (np.expand_dims(array1.sum(axis=1), 1) + eps) + eps
        array2 = array2 / (np.expand_dims(array2.sum(axis=1), 1) + eps) + eps
        dist = - np.log(np.sum((array1 * array2)**0.5, axis=1))
        dist[dist < 0] = 0
        return dist

    dist = _bhattacharyya(array1, array2)
    if norm:
        dist = normalize_distance(_bhattacharyya, dist, array1.shape[1])

    return dist.squeeze()


def jmd(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _jmd(array1, array2):
        dist = bhattacharyya(array1, array2, norm=False)
        dist = (2 * (1 - np.exp(-dist)))**0.5
        return dist

    dist = _jmd(array1, array2)
    if norm:
        dist = normalize_distance(_jmd, dist, array1.shape[1])

    return dist.squeeze()

"""
Hybrid distance measures
"""


def auc_sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    dist1 = auc(array1, array2, norm=True)
    dist2 = sam(array1, array2, norm=False)
    dist = (1 + dist1) * (1 + np.sin(dist2)) - 1

    return dist.squeeze()


def l1_sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _l1_sam(array1, array2):
        dist1 = l1(array1, array2, norm=False)
        dist2 = sam(array1, array2, norm=False)
        dist = dist1 * np.sin(dist2)
        return dist

    dist = _l1_sam(array1, array2)
    if norm:
        dist = normalize_distance(_l1_sam, dist, array1.shape[1])

    return dist.squeeze()


def l2_sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _l2_sam(array1, array2):
        dist1 = l2(array1, array2, norm=False)
        dist2 = sam(array1, array2, norm=False)
        dist = dist1 * np.sin(dist2)
        return dist

    dist = _l2_sam(array1, array2)
    if norm:
        dist = normalize_distance(_l2_sam, dist, array1.shape[1])

    return dist.squeeze()


def sid_sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _sid_sam(array1, array2):
        dist1 = sid(array1, array2, norm=False)
        dist2 = sam(array1, array2, norm=False)
        dist = dist1 * np.sin(dist2)
        return dist

    dist = _sid_sam(array1, array2)
    if norm:
        dist = normalize_distance(_sid_sam, dist, array1.shape[1])

    return dist.squeeze()


def jmd_sam(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _jmd_sam(array1, array2):
        dist1 = jmd(array1, array2, norm=False)
        dist2 = sam(array1, array2, norm=False)
        dist = dist1 * np.sin(dist2)
        return dist

    dist = _jmd_sam(array1, array2)
    if norm:
        dist = normalize_distance(_jmd_sam, dist, array1.shape[1])

    return dist.squeeze()


def spectral_similarity(array1, array2, norm=True):

    array1, array2 = check_dims_values(array1, array2)

    def _spectral_similarity(array1, array2):
        dist1 = l2(array1, array2, norm=False) / array1.shape[1]**0.5
        dist2 = correlation(array1, array2, norm=False)
        dist = (dist1**2 + (1 - dist2)**2)**0.5
        return dist

    dist = _spectral_similarity(array1, array2)
    if norm:
        dist = normalize_distance(_spectral_similarity, dist, array1.shape[1])

    return dist.squeeze()

