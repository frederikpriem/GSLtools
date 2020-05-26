import numpy as np


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
    :param libpath: str
    :param size: int, sample size
    :param complexity: list or tuple of floats, first element corresponds to fraction binary, second to fraction ternary
     etc.
    :param include_original: bool
    :param sampling: str or dict, 'random', or 'stratified' or dict {class name:weight}
    :param background: str or list of str, classes not included in fraction labels
    :return: 
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

    # make sure that complexity vector sums to 1
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
        num = np.argmax(cs <= rnd) + 2

        # randomly generate fractions
        randfrac = np.random.rand(num)

        for n in range(1, num):
            
            randfrac[n] = 1 - np.sum(randfrac[:n])

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
                                        replace=True,
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


def perturbate_spectra(spectra, intensity,
                       mode='relative'):

    perturbated = np.zeros(spectra.shape)

    for s, spec in enumerate(spectra):

        if mode == 'absolute':
            perturbated[s, :] = spec + np.random.normal(0, intensity, spec.size)
        else:

            for r, refl in enumerate(spec):

                perturbated[s, r] = refl + np.random.normal(0, r * intensity)

    perturbated[perturbated < 0] = 0
    perturbated[perturbated > 1] = 1

    return perturbated
