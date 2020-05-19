import numpy as np
from scipy.ndimage import gaussian_filter
import copy


def spectral_resampling(new_wavs, new_fwhm, old_wavs, old_fwhm, old_refl):

    """
    This function spectrally resamples reflectance data from a source band definition to newly specified one. This code
    is an adaptation from https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py.

    It requires central wavelengths (wavs) and bandwidths (fwhm) of both definitions.

    Make sure that the old and new band definitions overlap at least partially, otherwise, an error will be raised.

    Also make sure that new_wavs, new_fwhm, old_wavs and old_fwhm share the same unit, e.g. micrometer or nanometer.

    Reflectance data can be expressed as non-negative floating point numbers or integers.

    :param new_wavs: 1D-array, target wavelenghts, i.e. band central wavelengths
    :param new_fwhm: 1D-array, target full widths at half maximum, i.e. bandwidth
    :param old_wavs: 1D-array, source wavelenghts, corresponding to old_refl
    :param old_fwhm: 1D-array, source full widths at half maximum, corresponding to old_refl
    :param old_refl: 2D-array, source reflectance data, rows = observations, columns = bands

    :return: new_refl: 2D-array, resampled reflectance data in new band definition,
    rows = observations, columns = bands
    """

    # check that for every new band there is at least 1 (partially) overlapping old band
    for newband in range(new_wavs.shape[0]):

        conleft1 = (new_wavs[newband] - new_fwhm[newband] / 2) >= (old_wavs - old_fwhm / 2)
        conleft2 = (new_wavs[newband] - new_fwhm[newband] / 2) <= (old_wavs + old_fwhm / 2)
        conleft = np.any(np.logical_and(conleft1, conleft2))

        conright1 = (new_wavs[newband] + new_fwhm[newband] / 2) >= (old_wavs - old_fwhm / 2)
        conright2 = (new_wavs[newband] + new_fwhm[newband] / 2) <= (old_wavs + old_fwhm / 2)
        conright = np.any(np.logical_and(conright1, conright2))

        if not (conleft or conright):
            raise ValueError("The new bands specified must overlap at least"
                             "partially with the old band definitions.")

    # make output array
    new_refl = np.zeros((old_refl.shape[0], new_wavs.size))

    start = 0
    stop = 0

    # Calculate new reflectance values band per band
    for newband in range(new_wavs.shape[0]):

        # Find first old band that is (partially) covered by the new band
        while (new_wavs[newband] - new_fwhm[newband] / 2) > (old_wavs[start] + old_fwhm[start] / 2):
            start += 1

        # Find last old band that is (partially) covered by the new band
        while (old_wavs[stop] - old_fwhm[stop] / 2) < (new_wavs[newband] + new_fwhm[newband] / 2):
            if stop == old_wavs.size - 1:
                break
            else:
                stop += 1

        # If the new band is fully within one old band, they are considered identical
        if stop == start:

            new_refl[:, newband] = old_refl[:, start]

        # Otherwise, multiply the widths of the first and last old bands by their covered fractions
        else:

            start_factor = np.abs((old_wavs[start] + old_fwhm[start] / 2) - (new_wavs[newband] - new_fwhm[newband] / 2))
            start_factor /= old_fwhm[start]

            end_factor = np.abs((new_wavs[newband] + new_fwhm[newband] / 2) - (old_wavs[stop] - old_fwhm[stop] / 2))
            end_factor /= old_fwhm[stop]

            widths = copy.deepcopy(old_fwhm[start:stop])
            widths[0] *= start_factor
            widths[-1] *= end_factor

            # calculate reflectance values for new band
            values = widths * old_refl[:, start:stop]
            new_refl[:, newband] = values.sum(axis=1)
            new_refl[:, newband] /= widths.sum()

    return new_refl


def bandclust(spectra, wavelengths, bandwidths,
              subbands_start=None,
              nbins='FD',
              sigma=1):

    def mutual_information(x, y):

        # determine the optimal number of bins for MI estimation using the Freedman-Diaconis rule
        if nbins == 'FD':

            n = x.size
            iqr_x = np.quantile(x, 0.75) - np.quantile(x, 0.25)
            iqr_y = np.quantile(y, 0.75) - np.quantile(y, 0.25)
            width_x = 2 * iqr_x / n**(1 / 3)
            width_y = 2 * iqr_y / n**(1 / 3)
            nbins_x = int((x.max() - x.min()) / width_x)
            nbins_y = int((y.max() - y.min()) / width_y)
            bins = (nbins_x, nbins_y)

        else:
            bins = (nbins, nbins)

        # get joint histogram
        jh = np.histogram2d(x, y, bins=bins)[0]
        jh = jh + np.finfo(jh.dtype).eps
        sh = np.sum(jh)
        jh = jh / sh

        # smooth joint histogram with a gaussian filter
        # gaussian_filter(jh,
        #                 sigma=sigma,
        #                 mode='constant',
        #                 output=jh)

        # get marginal histograms
        mh1 = np.sum(jh, axis=0).reshape((-1, jh.shape[1]))
        mh2 = np.sum(jh, axis=1).reshape((jh.shape[0], -1))

        # get joint entropy
        je = np.sum(-jh * np.log(jh))

        # get marginal entropies
        me1 = np.sum(-mh1 * np.log(mh1))
        me2 = np.sum(-mh2 * np.log(mh2))

        # get mutual information
        mi = me1 + me2 - je

        return mi

    def split_subband(spectra, b, bmin, bmax):

        left = spectra[:, bmin:b].mean(axis=1).squeeze()
        right = spectra[:, b:bmax].mean(axis=1).squeeze()

        return left, right

    def mi_subband(spectra, bmin, bmax):

        bands = np.arange(bmin + 1, bmax)
        mi = np.empty(bands.size)

        for B, b in enumerate(bands):

            left, right = split_subband(spectra, b, bmin, bmax)
            mi[B] = mutual_information(left, right)

        # smooth the MI vector
        gaussian_filter(mi,
                        sigma=sigma,
                        mode='constant',
                        output=mi)

        return mi

    def optimal_split(mi):

        # find the local minima in the MI vector
        copy.deepcopy(mi)
        d = np.diff(mi)
        d = np.sign(d)
        d = np.diff(d)
        lmin = np.where(d == 2)[0]
        lmin += 1

        # the optimal split band corresponds to the local minimum with the lowest MI
        if lmin.size > 0:
            ind = np.argmin(mi[lmin])
            split = lmin[ind]
        else:
            split = None

        return split

    # the initial subband covers all bands
    if not subbands_start:
        subbands = [0, spectra.shape[1]]
    else:
        subbands = copy.deepcopy(subbands_start)
        subbands[-1] = spectra.shape[1]
    splits = [0]
    skip = []

    # iteratively split subbands in two
    # stop when no more local MI minima can be found or when the subband set is equal to the original band set
    while len(splits) > 0:

        splits = []

        for B, b in enumerate(subbands[:-1]):

            # get edges of current subband
            bmin = copy.deepcopy(b)
            bmax = copy.deepcopy(subbands[B + 1])

            if (not [bmin, bmax] in skip) and ((bmax - bmin) > 1):

                mi = mi_subband(spectra, bmin, bmax)
                split_temp = optimal_split(mi)

                if split_temp:
                    split_temp += bmin
                    splits.append(split_temp)
                else:
                    skip.append([bmin, bmax])

        # update subband definitions
        subbands += splits
        subbands = sorted(subbands)

    # produce the new band definition based on the clustering result
    wavelengths = np.array(wavelengths)
    bandwidths = np.array(bandwidths)
    new_wavelengths = []
    new_bandwidths = []

    for B, b in enumerate(subbands[:-1]):

        bmin = copy.deepcopy(b)
        bmax = copy.deepcopy(subbands[B + 1])

        if B == len(subbands[:-1]):
            bmax += 1

        new_wavelength = np.mean([wavelengths[bmin], wavelengths[bmax - 1]])
        new_bandwidth = wavelengths[bmax - 1] + bandwidths[bmax - 1] / 2 - (wavelengths[bmin] - bandwidths[bmin] / 2)
        new_wavelengths.append(new_wavelength)
        new_bandwidths.append(new_bandwidth)

    new_wavelengths = np.array(new_wavelengths)
    new_bandwidths = np.array(new_bandwidths)

    return new_wavelengths, new_bandwidths
