import numpy as np
import copy
from tqdm import tqdm
from gultools.spectral_distance import sid_sam


def iterative_adaptive_smoothing(image,
                                 iterations=5,
                                 distance_measure=sid_sam,
                                 distance_threshold=0.000001,
                                 normalize_distance=True):

    """
    performs spatially adaptive smoothing on image over several iterations
    :param image: 3D-array of shape (rows, cols, bands)
    :param iterations: int
    :param distance_measure: object, spectral distance measure used to assess (dis)similarity between neighbouring pixels
    :param distance_threshold: float, distance threshold used to determine whether two pixels are similar
    :param normalize_distance: bool, whether to use normalized distance measure
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

                    check = distance_measure(image_array, image_shift_array,
                                             norm=normalize_distance)
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
