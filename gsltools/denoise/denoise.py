import numpy as np
import copy
from tqdm import tqdm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

"""
This module handles spectral and spatial denoising/smoothing of image and library data.
"""


def signal_noise_decomposition(image):

    """
    Decomposes an image in its signal and noise components using linear regression. This implementation draws on the
    method proposed by Bioucas-Dias & Nascimento (2008).
    """

    if len(image.shape) > 3 or len(image.shape) < 2:
        err_msg = """image must either be a 2D array of shape (spectra, bands)
        or a 3D array of shape (rows, columns, bands)"""
        raise ValueError(err_msg)

    if len(image.shape) == 3:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)
    else:
        bands = image.shape[1]
        rows = None
        cols = None

    image = copy.deepcopy(image)
    signal = np.empty(image.shape)
    noise = np.empty(image.shape)

    for b in range(bands):

        lr = LinearRegression(fit_intercept=False)
        lr.fit(np.delete(image, b, axis=1), image[:, b])
        signal_temp = lr.predict(np.delete(image, b, axis=1))

        noise_temp = image[:, b] - signal_temp
        noise_temp = noise_temp.squeeze()
        signal[:, b] = signal_temp
        noise[:, b] = noise_temp

    if rows is not None:
        signal = signal.reshape((rows, cols, bands))
        noise = noise.reshape((rows, cols, bands))

    return signal, noise


def hysime(image):

    """
    HYSIME determines the optimal dimensionality of an image subspace formed by its eigenvectors. It is a completely
    unsupervised algorithm, proposed by Bioucas-Dias & Nascimento (2008). It draws on signal_noise_decomposition.
    """

    if len(image.shape) > 3 or len(image.shape) < 2:
        err_msg = """image must either be a 2D array of shape (spectra, bands)
        or a 3D array of shape (rows, columns, bands)"""
        raise ValueError(err_msg)

    if len(image.shape) == 3:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)

    signal, noise = signal_noise_decomposition(image)
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)
    noise = (noise - noise.mean(axis=0)) / noise.std(axis=0)
    image = copy.deepcopy(image)
    image = (image - image.mean(axis=0)) / image.std(axis=0)
    ry = np.dot(image.T, image) / image.shape[0]
    rn = np.dot(noise.T, noise) / image.shape[0]
    rx = np.dot(signal.T, signal) / image.shape[0]

    ev, eig = np.linalg.eig(rx)
    ind = np.argsort(ev)[::-1]
    eig = eig[ind]
    delta = []

    for k in range(image.shape[1]):

        delta_temp = 0

        for j in range(k + 1):

            delta_temp -= np.dot(np.dot(eig[:, j].T, ry), eig[:, j])
            delta_temp += 2 * np.dot(np.dot(eig[:, j].T, rn), eig[:, j])

        delta.append(delta_temp)

    n_subspace = len(np.where(np.array(delta) < 0)[0])
    subspace = eig[:, :n_subspace].T

    return n_subspace, subspace


def spectral_smoothing_gaussian(spectra, wavelengths, std=0.01):

    """This function performs spectral smoothing using spectral convolution with a Gaussian filter"""

    if len(spectra.shape) != 2:
        err_msg = """spectra must be a 2D array of shape (spectra, bands)"""
        raise ValueError(err_msg)

    if spectra.shape[1] != wavelengths.shape[0]:
        err_msg = """number of bands and wavelengths must be equal"""
        raise ValueError(err_msg)

    if std <= 0:
        err_msg = """std must be an integer or float greater than zero"""
        raise ValueError(err_msg)

    spectra_smooth = copy.deepcopy(spectra)
    n_bands = spectra.shape[1]

    for band in range(n_bands):

        wav = wavelengths[band]
        w = norm(wav, std).pdf(wavelengths)
        w /= w.sum()
        w = w.reshape(1, -1)
        spectra_smooth[:, band] = np.sum(w * spectra, axis=1)

    return spectra_smooth


def gaussian_noise(spectra, snr_db,
                   noise_seed=0):

    np.random.seed(noise_seed)
    noise = np.random.randn(*spectra.shape)
    noise *= spectra / 10 ** (snr_db / 10)

    return noise


def iterative_adaptive_smoothing(image, distance_measure, distance_threshold,
                                 iterations=5):

    """
    performs spatially adaptive smoothing on an image over several iterations

    :param image: 3D-array of shape (rows, cols, bands)
    :param iterations: int
    :param distance_measure: object, spectral distance measure used to assess (dis)similarity between neighbouring pixels
    :param distance_threshold: float, distance threshold used to determine whether two pixels are similar
    :return:
    """

    image[image > 1] = 1
    image[image < 0] = 0
    rows, cols, bands = image.shape
    image = np.pad(image, 1,
                   mode='constant',
                   constant_values=0)
    image = image[:, :, 1:-1]
    image_array = image.reshape((rows + 2) * (cols + 2), bands)

    vshifts = [-1, 0, 1]
    hshifts = [-1, 0, 1]
    iterations = list(range(iterations))

    for it in tqdm(iterations):

        new_image = copy.deepcopy(image)
        weights = np.ones((rows + 2, cols + 2))

        for h in hshifts:

            for v in vshifts:

                if v != 0 or h != 0:

                    image_shift = np.roll(image, (v, h), axis=(0, 1))
                    image_shift_array = image_shift.reshape((rows + 2) * (cols + 2), bands)

                    check = distance_measure(image_array, image_shift_array)
                    check = check < distance_threshold
                    check = check.astype(float)
                    check = check.reshape(rows + 2, cols + 2)
                    weights += check
                    check = np.expand_dims(check, axis=2)
                    new_image += check * image_shift

        weights = np.expand_dims(weights, axis=2)
        image = new_image / weights

    image = image[1:-1, 1:-1, :]

    return image
