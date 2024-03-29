import numpy as np


def synthetic_mixing(spectra, labels, n_mixtures,
                     target_classes=None,
                     mixing_complexity=(0.5, 0.5),
                     bilinear=False,
                     bilinear_scale=.05,
                     include_original=False,
                     sampling='stratified',
                     only_within_class_mixing=False,
                     only_between_class_mixing=False,
                     return_dominant_class=False,
                     seed=0,
                     ):

    np.random.seed(seed)

    if target_classes is None:
        target_classes = np.unique(labels)
    else:
        target_classes = np.array(target_classes)

    numspec = spectra.shape[0]
    numbands = spectra.shape[1]

    # get spectra names
    unique_labels, counts = np.unique(labels, return_counts=True)

    # produce weights for sampling
    weights = np.zeros(labels.size, dtype=float)

    if sampling == 'random':

        weights = np.ones(labels.size, dtype=float)

    elif sampling == 'stratified':

        for unique_label, count in zip(unique_labels, counts):

            weights[labels == unique_label] = float(numspec) / count

    elif isinstance(sampling, dict):

        for unique_label in unique_labels:

            if unique_label in sampling:
                weights[labels == unique_label] = sampling[unique_label] / float(len(labels[labels == unique_label]))

    weights /= weights.sum()

    # make the complexity vector sum to 1
    mixing_complexity = np.array(mixing_complexity)
    mixing_complexity /= mixing_complexity.sum()

    # create empty output arrays
    mix = np.zeros((n_mixtures, numbands))
    frac = np.zeros((n_mixtures, len(target_classes)))

    # make complexity cumsum vector to sample the number of spectra to be used in mixtures
    cs = np.cumsum(mixing_complexity)

    # generate mixtures iteratively
    for s in np.arange(n_mixtures):

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
        if only_within_class_mixing:

            specind1 = np.random.choice(range(numspec),
                                        size=1,
                                        p=weights)

            p = np.where(labels == labels[specind1], 1.0, 0.0)
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

            p = np.where(labels != labels[specind1], 1.0, 0.0)
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

        # sum the fractions belonging to the same target classes to produce the final fractions
        for i, ind in enumerate(specind):

            target_class_ind = np.where(target_classes == labels[ind])[0]

            if len(target_class_ind) > 0:
                frac[s, target_class_ind[0]] += randfrac[i]

        # generate the synthetic mixture
        mixtemp = np.zeros(numbands)

        for n in range(num):
            mixtemp += randfrac[n] * spectra[specind[n], :]

        # add the bilinear component to the mixture
        # the bilinear component is generated with the first two spectra of the mixture
        if bilinear:
            b_randfrac = np.random.exponential(scale=bilinear_scale)
            mixtemp += b_randfrac * spectra[specind[0], :] * spectra[specind[1], :]

        mix[s, :] = mixtemp

    # include the original pure endmember spectra in the synthetic mixtures
    if include_original:

        mix = np.concatenate((mix, spectra), axis=0)

        for label in labels:

            fractemp = np.where(target_classes == label, 1.0, 0.0)
            fractemp = fractemp.reshape(1, -1)
            frac = np.concatenate((frac, fractemp), axis=0)

    output = (mix, frac)

    if return_dominant_class:

        class_ind = np.argmax(frac, axis=1)
        dominant_class = target_classes[class_ind]
        output = (mix, frac, dominant_class)

    return output
