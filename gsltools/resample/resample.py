import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import gsltools._check as check
import copy


"""
This module handles spectral resampling and clustering of imagery and libraries.
"""


def spectral_resampling(new_cwav, new_fwhm, old_cwav, old_fwhm, old_refl,
                        band_overlap_threshold=0.5,
                        fill_insufficient_overlap=None,
                        raise_insufficient_overlap=True,
                        error_prefix=''):

    """
    This function uses spectral convolution to spectraly resample libraries or images to other band definitions.
    It assumes that bands are characterized by a Gaussian Spectral Response Function (SRF) that can be parametrized with
    the given central wavelengths and FWHM.
    To avoid resampling in case of insufficient overlap between new and old band definitions, the overlap threshold is
    used. The default value of 0.5 means that at least 1 old band must fall within the FWHM range of each new band,
    i.e. the wavelength range of the SRF where min. intensity = 0.5 * max. Lowering the threshold eases this restriction
    and vice versa. Setting the threshold to zero effectively negates the constraint, setting it to 1 means that there
    must be an exact match between an old and new central wavelength for the latter to be computed.
    """

    check.is_not_none(new_cwav,
                      object_name='new_cwav')
    check.is_not_none(new_fwhm,
                      object_name='new_fwhm')
    check.is_not_none(old_cwav,
                      object_name='old_cwav')
    check.is_not_none(old_fwhm,
                      object_name='old_fwhm')
    check.is_not_none(old_refl,
                      object_name='old_refl')
    check.is_array_like(new_cwav,
                        dimensions=1,
                        repetition_allowed=False,
                        dtype=float,
                        object_name='new_cwav')
    check.is_array_like(new_fwhm,
                        dimensions=1,
                        size=len(new_cwav),
                        dtype=float,
                        object_name='new_fwhm')
    check.is_array_like(old_cwav,
                        dimensions=1,
                        repetition_allowed=False,
                        dtype=float,
                        object_name='old_cwav')
    check.is_array_like(old_fwhm,
                        dimensions=1,
                        size=len(old_cwav),
                        dtype=float,
                        object_name='old_fwhm')
    check.is_array_like(old_refl,
                        dimensions=2,
                        dtype=float,
                        object_name='old_refl')
    check.is_float(band_overlap_threshold,
                   ge=0,
                   le=1,
                   object_name='overlap_threshold')
    check.is_float(fill_insufficient_overlap,
                   object_name='fill_insufficient_overlap')
    check.is_bool(raise_insufficient_overlap,
                  object_name='raise_insufficient_overlap')
    check.is_str(error_prefix,
                 object_name='error_prefix')
    if old_refl.shape[1] != len(old_cwav) and old_refl.shape[1] != len(old_fwhm):
        raise Exception('{}the lengths of old_cwav and old_fwhm and the number of columns in old_refl must be the same'.format(error_prefix))

    # calculate reflectance values for each new band
    new_refl = np.zeros((old_refl.shape[0], new_cwav.size))

    for band, (cwav, fwhm) in enumerate(zip(new_cwav, new_fwhm)):
        
        std = fwhm / 2.355
        w1 = norm(cwav, std).pdf(old_cwav).reshape(1, -1)
        w2 = np.array(old_fwhm).reshape(1, -1)

        if np.all(w1 < band_overlap_threshold * norm(cwav, std).pdf(cwav)) and fill_insufficient_overlap is not None:
            new_refl[:, band] = fill_insufficient_overlap
        elif np.all(w1 < band_overlap_threshold * norm(cwav, std).pdf(cwav)) and fill_insufficient_overlap is None:
            if raise_insufficient_overlap:
                raise ValueError('{}insufficient overlap between new band (cwav={}, fwhm={}) and old bands'.format(error_prefix, cwav, fwhm))
            else:
                new_refl[:, band] = np.nan
        else:
            new_refl[:, band] = np.sum(w1 * w2 * old_refl, axis=1) / np.sum(w1 * w2, axis=None)

    return new_refl


def bandclust(spectra, wavelengths, bandwidths,
              subbands_start=None,
              nbins=None,
              sigma=1.):

    """
    Clusters adjacent bands based on Mutual Information (MI). See corresponding paper for more information.
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param wavelengths: 1D-array of float of shape (b bands), containing the central wavelength of each band
    :param bandwidths: 1D-array of float of shape (b bands), containing bandwidths for each band
    :param subbands_start: Iterable containing int values, initial subband definition. Values must range between 0 and
    b - 1. This can be used to avoid clustering of bands over a spectral interval that is not covered by the sensor.
    :param nbins: int , number of bins used to estimate mutual information, if None then nbins is estimated with
    Freedman-Diaconis rule.
    :param sigma: float, parameter used to smooth the MI curve prior to subband splitting.
    :return:
        new_wavelengths: 1D-array of floats with shape (b bands,), central wavelengths of new clustered band
        definition
        new_bandwidths: 1D-array of floats with shape (b bands,), bandwidths of new clustered band definition

    """

    check.is_not_none(spectra,
                      object_name='spectra')
    check.is_not_none(wavelengths,
                      object_name='wavelengths')
    check.is_not_none(bandwidths,
                      object_name='bandwidths')
    check.is_array_like(spectra,
                        dimensions=2,
                        dtype=float,
                        object_name='spectra')
    check.is_array_like(wavelengths,
                        dimensions=1,
                        dtype=float,
                        object_name='wavelengths')
    check.is_array_like(bandwidths,
                        dimensions=1,
                        dtype=float,
                        object_name='bandwidths')
    check.is_array_like(subbands_start,
                        dimensions=1,
                        dtype=int,
                        object_name='subbands_start')

    if subbands_start is None:
        subbands = [0, spectra.shape[1] - 1]
    else:
        subbands = copy.deepcopy(subbands_start)

    for s in subbands:

        check.is_int(s,
                     ge=0,
                     le=spectra.shape[1] - 1,
                     object_name='subbands_start elements')

    check.is_int(nbins,
                 object_name='nbins')
    check.is_float(sigma,
                   g=0,
                   object_name='sigma')

    spectra = np.array(spectra, dtype=float)
    wavelengths = np.array(wavelengths, dtype=float)
    bandwidths = np.array(bandwidths, dtype=float)

    def mutual_information(x, y):

        # determine the optimal number of bins for MI performance using the Freedman-Diaconis rule
        if nbins is None:

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

    # main code block
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

        # look up moments of mixed distributions to understand this step
        new_wavelength = np.sum(wavelengths[bmin:bmax] * bandwidths[bmin:bmax]) / np.sum(bandwidths[bmin:bmax])
        new_bandwidth = bandwidths[bmin:bmax]**2 + wavelengths[bmin:bmax]**2 - new_wavelength**2
        new_bandwidth = np.sum(new_bandwidth * bandwidths[bmin:bmax])
        new_bandwidth /= np.sum(bandwidths[bmin:bmax])
        new_bandwidth = new_bandwidth**0.5

        # look up conversion std to fwhm to understand this step
        new_bandwidth = new_bandwidth * 2.355

        new_wavelengths.append(new_wavelength)
        new_bandwidths.append(new_bandwidth)

    new_wavelengths = np.array(new_wavelengths)
    new_bandwidths = np.array(new_bandwidths)

    return new_wavelengths, new_bandwidths
