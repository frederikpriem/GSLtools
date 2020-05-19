import numpy as np
import copy
from tqdm import tqdm
from gultools.spectral_distance import sid_sam
from sklearn.linear_model import LinearRegression


def iterative_adaptive_smoothing(image,
                                 iterations=5,
                                 distance_measure=sid_sam,
                                 distance_threshold=0.000001):

    image = copy.deepcopy(image)
    image[image > 1] = 1
    image[image < 0] = 0
    rows, cols, bands = image.shape
    total = iterations * rows * cols

    with tqdm(total=total) as t:

        for it in range(iterations):

            new_image = copy.deepcopy(image)

            for row in range(rows):

                for col in range(cols):

                    spectrum = image[row, col, :]

                    if spectrum.sum != 0 and spectrum.sum != spectrum.size:

                        rmin = max(0, row - 1)
                        rmax = row + 2
                        cmin = max(0, col - 1)
                        cmax = col + 2
                        neigh = image[rmin:rmax, cmin:cmax, :]
                        neigh = neigh.reshape(neigh.shape[0] * neigh.shape[1], bands)

                        dist = distance_measure(spectrum, neigh)
                        weights = np.where(dist < distance_threshold, 1.0, 0.0)
                        weights /= weights.sum()
                        new_image[row, col, :] = np.sum(np.expand_dims(weights, 1) * neigh, axis=0)

                        t.update()

            image = copy.deepcopy(new_image)

    return image


def estimate_snr(image, homogeneous_areas):

    bands = image.shape[2]
    area_ids = np.unique(homogeneous_areas)
    area_ids = area_ids[area_ids > 0]
    n_area = area_ids.size

    snr_collection = np.zeros((n_area, bands))

    for a, area_id in enumerate(area_ids):

        rows, cols = np.where(homogeneous_areas == area_id)
        spectra = image[rows, cols, :]
        signal = spectra.mean(axis=0)
        noise = spectra.std(axis=0)
        snr = signal / noise
        snr[np.isinf(snr)] = 0
        snr[np.isnan(snr)] = 0
        snr_collection[a, :] = snr

    snr_mean = snr_collection.mean(axis=0)
    snr_min = snr_collection.min(axis=0)
    snr_max = snr_collection.max(axis=0)
    if n_area > 1:
        snr_std = snr_collection.std(axis=0)
    else:
        snr_std = None

    return snr_mean, snr_std, snr_min, snr_max


def signal_noise_decomposition(image,
                               ss=1000):

    rows, cols, bands = image.shape
    image = image.reshape(rows * cols, bands)
    signal = np.zeros((rows * cols, bands))
    noise = np.zeros((rows * cols, bands))

    if ss == 'all':
        ss = rows * cols

    sample = np.random.choice(range(rows * cols),
                              size=ss,
                              replace=False)

    band_list = list(range(bands))

    for b in tqdm(band_list):

        band = image[:, b]
        other_bands = np.delete(image, b, axis=1)
        y = band[sample]
        x = other_bands[sample, :]
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        band_signal = model.predict(other_bands)
        band_noise = band - band_signal
        signal[:, b] = copy.deepcopy(band_signal)
        noise[:, b] = copy.deepcopy(band_noise)

    signal = signal.reshape(rows, cols, bands)
    noise = noise.reshape(rows, cols, bands)

    return signal, noise