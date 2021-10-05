import numpy as np


"""
This module handles synthetic mixing of discrete single class label spectra
"""


def synthetic_mixing(spectra, names, size,
                     complexity=(0.5, 0.5),
                     include_original=False,
                     sampling='stratified',
                     background=None,
                     only_within_class_mixing=False,
                     only_between_class_mixing=False,
                     return_dominant_class=False,
                     ):
    
    """
    Randomly generates synthetic linear mixtures based on a given set of spectra.
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param names: 1D-array of strings with shape (n spectra,), contains class label of each spectrum
    :param size: int, number of synthetic mixtures to generate
    :param complexity: iterable of floats, weights given to generating binary, ternary, quaternary ... mixtures
    :param include_original: bool, whether to include the original spectra in the output array of spectra
    :param sampling: str, type of sampling used, either 'random' or 'stratified'
    :param background: str or iterable of strings, classes considered as background whose fractions are not to be taken
    in account
    :param only_within_class_mixing: bool, False by default, whether to only perform withing class mixing
    :param only_between_class_mixing: bool, False by default, whether to only perform between class mixing
    :param return_dominant_class: bool, False by default, whether to return the class labels corresponding to the
    highest class fraction of each generated mixture.
    :return:
        mix: 2D-array of floats with shape (m mixtures, b bands), containing reflectance values of synthetic mixtures
        frac: 2D-array of floats with shape (m mixtures, c classes), containing fraction labels of each mixture
    """

    numspec = spectra.shape[0]
    numbands = spectra.shape[1]

    # get spectra names
    classnames, counts = np.unique(names, return_counts=True)
    numclass = classnames.size

    # produce weights for sampling
    weights = np.zeros(names.size, dtype=float)

    if sampling == 'random':

        weights = np.ones(names.size, dtype=float)

    elif sampling == 'stratified':

        for name, count in zip(classnames, counts):

            weights[names == name] = float(numspec) / count

    elif isinstance(sampling, dict):

        for name in classnames:

            weights[names == name] = sampling[name]

    weights /= weights.sum()

    # make sure that the complexity vector sums to 1
    complexity = np.array(complexity)
    complexity /= complexity.sum()

    # create empty output arrays
    mix = np.zeros((size, numbands))
    frac = np.zeros((size, numclass))

    # make complexity cumsum vector to sample the number of spectra to be used in mixtures
    cs = np.cumsum(complexity)

    # generate mixtures iteratively
    for s in np.arange(size):

        # sample number of spectra to be used in mixture
        rnd = np.random.rand()
        num = np.argmax(cs >= rnd) + 2

        # randomly generate fractions
        randfrac = np.random.rand(num)

        for n in range(1, num):

            if n == num - 1:
                randfrac[n] = 1 - np.sum(randfrac[:n])
            else:
                randfrac[n] = np.random.uniform(0, 1 - np.sum(randfrac[:n]))

        # randomly assign library spectra to each fraction
        # use weights to perform stratified sampling
        if only_within_class_mixing:

            specind1 = np.random.choice(range(numspec),
                                        size=1,
                                        p=weights)

            p = np.where(names == names[specind1], 1.0, 0.0)
            if p.sum() > 1:
                p[specind1] = 0
            p *= weights
            p /= p.sum()

            specind2 = np.random.choice(range(numspec),
                                        size=num - 1,
                                        replace=False,
                                        p=p)

            specind = np.concatenate((specind1, specind2))

        elif only_between_class_mixing:

            specind1 = np.random.choice(range(numspec),
                                        size=1,
                                        p=weights)

            p = np.where(names != names[specind1], 1.0, 0.0)
            p *= weights
            p /= p.sum()

            specind2 = np.random.choice(range(numspec),
                                        size=num - 1,
                                        replace=False,
                                        p=p)

            specind = np.concatenate((specind1, specind2))
            
        else:

            specind = np.random.choice(range(numspec),
                                       size=num,
                                       replace=False,
                                       p=weights)

        # sum fractions of same class to produce final fractions
        for i, ind in enumerate(specind):
            
            classind = np.where(classnames == names[ind])[0]
            frac[s, classind] += randfrac[i]

        # generate synthetic mixtures
        mixtemp = np.zeros(numbands)
        
        for n in range(num):
            
            mixtemp += randfrac[n] * spectra[specind[n], :]
            
        mix[s, :] = mixtemp

    # include pure endmembers in synthetic mixtures
    if include_original:

        mix = np.concatenate((mix, spectra), axis=0)

        for name in names:

            fractemp = np.where(classnames == name, 1.0, 0.0)
            fractemp = fractemp.reshape(1, -1)
            frac = np.concatenate((frac, fractemp), axis=0)

    # remove background classes from fraction labels
    if background is not None:

        if isinstance(background, str):
            background = [background]

        for b in background:

            classind = np.where(classnames == b)[0]
            frac = np.delete(frac, classind, axis=1)
            classnames = np.delete(classnames, classind)

    output = (mix, frac)
    if return_dominant_class:
        classind = np.argmax(frac, axis=1)
        dominant_class = classnames[classind]
        output = (mix, frac, dominant_class)

    return output
