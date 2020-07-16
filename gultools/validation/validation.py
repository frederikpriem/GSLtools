import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import norm


def kappa_coefficient(est, ref, labels=None):

    """
    Computes overall kappa coefficient based on estimated and reference set of class labels.
    :param est: 1D-array of int or str, estimated class labels
    :param ref: 1D-array of int or str, reference class labels
    :param labels: iterable of int or str, subset of classes to use to calculate kappa
    :return:
        kappa: float, kappa coefficient
        var: float, estimated variance of kappa coefficient
    """

    # calculate the confusion matrix
    cm = confusion_matrix(ref, est, labels=labels)
    cm = cm.T  # transpose confusion matrix so that rows = estimates and columns = reference (Congalton & Green style)
    cm = cm.astype(float)

    # calculate some confusion matrix derived metrics
    k = cm.shape[0]
    n = cm.sum(axis=None)
    nii = np.diag(cm)
    ni = cm.sum(axis=1)
    nj = cm.sum(axis=0)

    # estimate the Kappa coefficient
    kappa = (n * nii.sum() - np.sum(ni * nj)) / (n**2 - np.sum(ni * nj))

    # estimate the variance of the Kappa coefficient using the Delta method
    t1 = nii.sum() / n
    t2 = np.sum(ni * nj) / n**2
    t3 = np.sum(nii * (ni + nj)) / n**2
    t4 = 0

    for i in range(k):

        for j in range(k):

            t4 += cm[i, j] * (ni[i] + nj[j])**2

    t4 /= n**3

    var1 = t1 * (1 - t1) / (1 - t2)**2
    var2 = 2 * (1 - t1) * (2 * t1 * t2 - t3) / (1 - t2)**3
    var3 = (1 - t1)**2 * (t4 - 4 * t2**2) / (1 - t2)**4
    var = (var1 + var2 + var3) / n

    return kappa, var


def conditional_kappa_coefficients(est, ref, labels=None):

    """
    Computes conditional class-wise kappa coefficients based on estimated and reference set of class labels.
    :param est: 1D-array of int or str, estimated class labels
    :param ref: 1D-array of int or str, reference class labels
    :param labels: iterable of int or str, subset of classes to use to calculate kappa
    :return:
        kappas: 1D-array of floats, conditional class-wise kappa coefficients
        vars: 1D-array of floats, estimated variances of conditional class-wise kappa coefficients
    """

    # calculate the confusion matrix
    cm = confusion_matrix(ref, est, labels=labels)
    cm = cm.T  # transpose confusion matrix so that rows = estimates and columns = reference (Congalton & Green style)
    cm = cm.astype(float)

    # calculate some confusion matrix derived metrics
    k = cm.shape[0]
    n = cm.sum(axis=None)
    nii = np.diag(cm)
    ni = cm.sum(axis=1)
    nj = cm.sum(axis=0)

    # calculate the actual conditional Kappa coefficients and their estimated variances
    kappas = np.empty(k)
    vars = np.empty(k)

    for i in range(k):

        kappas[i] = (n * nii[i] - ni[i] * nj[i]) / (n * ni[i] - ni[i] * nj[i])

        var_nom1 = n * (ni[i] - nii[i])
        var_nom2 = (ni[i] - nii[i]) * (ni[i] * nj[i] - n * nii[i]) + n * nii[i] * (n - ni[i] - nj[i] + nii[i])
        var_denom = ni[i] * (n - nj[i])
        var_denom = var_denom**3
        vars[i] = var_nom1 * var_nom2 / var_denom

    return kappas, vars


def test_significance(kappa, var, confidence=0.05):

    """
    Tests if (conditional) kappa coefficients are significantly different from zero.
    :param kappa: float or iterable of floats, (conditional) kappa coefficient(s)
    :param var: float or iterable of floats, estimated variances of (conditional) kappa coefficient(s)
    :param confidence: float, desired confidence level
    :return:
        z: float or iterable of floats, z-value(s) corresponding to difference(s)
        crit: float, critical value, if z > crit -> reject H0: kappa1 = kappa2
    """

    # if z > crit -> reject H0: kappa = 0
    z = kappa / var**0.5
    crit = norm().ppf(1 - confidence / 2)

    return z, crit


def test_signifance_difference(kappa1, kappa2, var1, var2, confidence=0.05):

    """
    Tests significance of difference between (conditional) kappa coefficients.
    :param kappa1: float or iterable of floats, first (conditional) kappa coefficient(s)
    :param kappa2: float or iterable of floats, second (conditional) kappa coefficient(s)
    :param var1: float or iterable of floats, first estimated variances of (conditional) kappa coefficient(s)
    :param var2: float or iterable of floats, second estimated variances of (conditional) kappa coefficient(s)
    :param confidence: float, desired confidence level
    :return:
        z: float or iterable of floats, z-value(s) corresponding to difference(s)
        crit: float, critical value, if z > crit -> reject H0: kappa1 = kappa2
    """

    z = np.abs(kappa1 - kappa2) / (var1 + var2)**0.5
    crit = norm().ppf(1 - confidence / 2)

    return z, crit
