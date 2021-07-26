import os
import copy
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import gsltools._check as check
from gsltools.io import read_envi_library, read_envi_header, save_envi_library, envi_to_dtype, dtype_to_envi
from gsltools.resample import spectral_resampling


class MetadataAttribute:

    def __init__(self, aid, atype, dtype,
                 description=None,
                 mandatory=False,
                 value_domain=None,
                 value_hierarchy=None,
                 regex=None,
                 max_value=None,
                 min_value=None,
                 nodata_value=None,
                 multivalued=0,
                 encapsulators='[]',
                 separator=',',
                 repetition_allowed=True,
                 _skip_checks=False
                 ):

        if not _skip_checks:

            check.is_not_none(aid, object_name='aid')
            check.is_not_none(atype, object_name='atype')
            check.is_not_none(dtype, object_name='dtype')
            check.is_int_or_str(aid,
                                object_name='aid')
            check.is_not_one_of(aid, check.protected_attributes,
                                object_name='aid')
            check.is_one_of(atype, ['l', 's', 'b'],
                            object_name='atype')
            check.is_one_of(dtype, check.allowed_dtypes,
                            object_name='dtype')
            check.is_str(description,
                         object_name='description')
            check.is_bool(mandatory,
                          object_name='mandatory')
            check.is_array_like(value_domain,
                                dimensions=1,
                                dtype=dtype,
                                object_name='value_domain')
            check.check_hierarchy(value_hierarchy,
                                  dtype=dtype,
                                  object_name='value_hierarchy')
            check.is_str(regex,
                         object_name='regex')
            if regex:
                dtype = str
            check.is_of_dtype(max_value, dtype,
                              object_name='max_value')
            check.is_of_dtype(min_value, dtype,
                              object_name='min_value')
            if not mandatory:
                check.is_not_none(nodata_value,
                                  object_name='nodata_value')
            check.is_of_dtype(nodata_value, dtype,
                              object_name='nodata_value')
            check.is_not_none(multivalued,
                              object_name='multivalued')
            check.is_one_of(multivalued, [0, 1, 2],
                            object_name='multivalued')
            if (atype == 's' or atype == 'b') and multivalued == 0:
                multivalued = 1
            if multivalued == 2:
                dtype = str
            check.is_one_of(encapsulators, ['[]', '()', '{}', '<>'],
                            object_name='encapsulators')
            check.is_str(separator,
                         length=1,
                         object_name='separator')
            check.is_bool(repetition_allowed,
                          object_name='repetition_allowed')

        self._aid = aid
        self._atype = atype
        self._dtype = dtype
        self._description = description
        self._mandatory = mandatory
        self._value_domain = value_domain
        self._value_hierarchy = value_hierarchy
        self._regex = regex
        self._max_value = max_value
        self._min_value = min_value
        self._nodata_value = nodata_value
        self._multivalued = multivalued
        self._encapsulators = encapsulators
        self._separator = separator
        self._repetition_allowed = repetition_allowed
        self._protected = False

    def print_summary(self):

        txt = '\nMetadata attribute'
        txt += '\n------------------\n\n'
        txt += 'aid = {}\n'.format(self._aid)
        txt += 'attribute type = {}\n'.format(self._atype)
        txt += 'data type = {}\n'.format(self._dtype)
        txt += 'description = {}\n'.format(self._description)
        txt += 'mandatory = {}\n'.format(self._mandatory)
        txt += 'value domain = {}\n'.format(self._value_domain)
        txt += 'value hierarchy = {}\n'.format(self._value_hierarchy)
        txt += 'regular expression = {}\n'.format(self._regex)
        txt += 'max value = {}\n'.format(self._max_value)
        txt += 'min value = {}\n'.format(self._min_value)
        txt += 'no data value = {}\n'.format(self._nodata_value)
        txt += 'multivalued = {}\n'.format(self._multivalued)
        txt += 'encapsulators = {}\n'.format(self._encapsulators)
        txt += 'separators = {}\n'.format(self._separator)
        txt += 'repetition allowed = {}\n'.format(self._repetition_allowed)
        txt += 'protected = {}\n'.format(self._protected)
        print(txt)


class MetadataModel:

    def __init__(self,
                 description=None):

        self._description = description
        self._mdm = dict()

        # add the protected metadata attributes that must be specified for each library
        lid = MetadataAttribute('lid', 'l', str,
                                description='Library ID',
                                mandatory=True,
                                multivalued=0,
                                _skip_checks=True)
        lid._protected = True
        self._mdm['lid'] = lid

        spectra = MetadataAttribute('spectra', 'l', int,
                                    description='The number of spectra included in the library',
                                    mandatory=True,
                                    min_value=1,
                                    multivalued=0,
                                    _skip_checks=True)
        spectra._protected = True
        self._mdm['spectra'] = spectra

        bands = MetadataAttribute('bands', 'l', int,
                                  description='The number of spectral bands included in the library',
                                  mandatory=True,
                                  min_value=1,
                                  multivalued=0,
                                  _skip_checks=True)
        bands._protected = True
        self._mdm['bands'] = bands

        wavelength_scale_factor = MetadataAttribute('wavelength scale factor', 'l', float,
                                                    description='Scale factor needed to convert the library wavelength unit to the SI unit (meter)',
                                                    mandatory=True,
                                                    min_value=np.finfo(float).tiny,
                                                    multivalued=0,
                                                    _skip_checks=True)
        wavelength_scale_factor._protected = True
        self._mdm['wavelength scale factor'] = wavelength_scale_factor

        wavelength = MetadataAttribute('wavelength', 'b', float,
                                       description='Central wavelengths of the library spectral bands, expressed in the unit'
                                                   'corresponding to the specified wavelength scale factor, e.g. micrometer -> 10E-6',
                                       mandatory=True,
                                       min_value=np.finfo(float).tiny,
                                       multivalued=1,
                                       repetition_allowed=False,
                                       _skip_checks=True
                                       )
        wavelength._protected = True
        self._mdm['wavelength'] = wavelength

        # add the protected optional metadata attributes
        sid = MetadataAttribute('sid', 's', str,
                                description='Spectrum ID',
                                mandatory=False,
                                multivalued=1,
                                repetition_allowed=False,
                                _skip_checks=True)
        sid._protected = True
        self._mdm['sid'] = sid

        bid = MetadataAttribute('bid', 'b', str,
                                description='Spectral band ID',
                                mandatory=False,
                                multivalued=1,
                                repetition_allowed=False,
                                _skip_checks=True)
        bid._protected = True
        self._mdm['bid'] = bid

        wavelength_inverse_unit = MetadataAttribute('wavelength inverse unit', 'l', bool,
                                                    description='Boolean indicating whether the library wavelength unit is inverse,'
                                                                'which is the case e.g. with wave numbers',
                                                    mandatory=False,
                                                    multivalued=0,
                                                    _skip_checks=True)
        wavelength_inverse_unit._protected = True
        self._mdm['wavelength inverse unit'] = wavelength_inverse_unit

        fwhm = MetadataAttribute('fwhm', 'b', float,
                                 description='Full Width at Half Maximum (bandwidth) of the library spectral bands, expressed in the unit'
                                             'corresponding to the specified wavelength scale factor, e.g. micrometer -> 10E-6',
                                 mandatory=False,
                                 min_value=np.finfo(float).tiny,
                                 multivalued=1,
                                 repetition_allowed=True,
                                 _skip_checks=True)
        fwhm._protected = True
        self._mdm['fwhm'] = fwhm

        reflectance_scale_factor = MetadataAttribute('reflectance scale factor', 'l', float,
                                                     description='Scale factor needed to convert library reflectance values to the [0, 1] range',
                                                     mandatory=False,
                                                     min_value=np.finfo(float).tiny,
                                                     multivalued=0,
                                                     _skip_checks=True)
        reflectance_scale_factor._protected = True
        self._mdm['reflectance scale factor'] = reflectance_scale_factor

        class_label = MetadataAttribute('class label', 's', str,
                                        description='Class label assigned to each spectrum',
                                        mandatory=False,
                                        multivalued=1,
                                        repetition_allowed=True,
                                        _skip_checks=True)
        class_label._protected = True
        self._mdm['class label'] = class_label

        l_filter = MetadataAttribute('l-filter', 'l', bool,
                                     description='Library filter (False = do not use)',
                                     mandatory=False,
                                     multivalued=0,
                                     _skip_checks=True)
        l_filter._protected = True
        self._mdm['l-filter'] = l_filter

        s_filter = MetadataAttribute('s-filter', 's', bool,
                                     description='Spectrum filter (False = do not use)',
                                     mandatory=False,
                                     multivalued=1,
                                     repetition_allowed=True,
                                     _skip_checks=True)
        s_filter._protected = True
        self._mdm['s-filter'] = s_filter

        b_filter = MetadataAttribute('b-filter', 'b', bool,
                                     description='Spectral band filter (False = do not use)',
                                     mandatory=False,
                                     multivalued=1,
                                     repetition_allowed=True,
                                     _skip_checks=True)
        b_filter._protected = True
        self._mdm['b-filter'] = b_filter

    def add_attribute(self, metadata_attribute):

        if not isinstance(metadata_attribute, MetadataAttribute):
            raise Exception('metadata attributes must be instances of the MetadataAttribute class')

        self._mdm[metadata_attribute._aid] = metadata_attribute

    def remove_attribute(self, aid):

        if aid in check.protected_attributes:
            raise Exception('protected metadata attributes cannot be edited or removed')

        if aid in self._mdm.keys():
            del self._mdm[aid]
        else:
            raise Exception('aid {} not found in the metadata model'.format(aid))

    def get_attribute_ids(self):

        return list(self._mdm.keys())

    def get_mandatory_attribute_ids(self):

        mandatory_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if attribute._mandatory:
                mandatory_aid_list.append(aid)

        return mandatory_aid_list

    def get_protected_attribute_ids(self):

        protected_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if attribute._protected:
                protected_aid_list.append(aid)

        return protected_aid_list

    def get_unprotected_attribute_ids(self):

        unprotected_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if not attribute._protected:
                unprotected_aid_list.append(aid)

        return unprotected_aid_list

    def get_optional_attribute_ids(self):

        optional_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if not attribute._mandatory:
                optional_aid_list.append(aid)

        return optional_aid_list

    def get_library_attribute_ids(self):

        library_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if attribute._atype == 'l':
                library_aid_list.append(aid)

        return library_aid_list

    def get_spectrum_attribute_ids(self):

        spectrum_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if attribute._atype == 's':
                spectrum_aid_list.append(aid)

        return spectrum_aid_list

    def get_band_attribute_ids(self):

        band_aid_list = []

        for aid in self.get_attribute_ids():

            attribute = self.get_attribute(aid)
            if attribute._atype == 'b':
                band_aid_list.append(aid)

        return band_aid_list

    def get_attribute(self, aid):

        check.is_not_none(aid,
                          object_name='aid')
        check.is_str(aid,
                     object_name='aid')
        check.is_one_of(aid, self.get_attribute_ids())

        return copy.deepcopy(self._mdm[aid])

    def print_summary(self):

        txt = '\nMetadata Model'
        txt += '\n--------------'
        txt += '\n\ndescription = {}'.format(self._description)
        print(txt)

        aid_list = self.get_attribute_ids()

        for aid in aid_list:

            attribute = self.get_attribute(aid)
            attribute.print_summary()


class SpectralLibrary:

    def __init__(self, metadata_model):

        if not isinstance(metadata_model, MetadataModel):
            raise Exception('the metadata model must be an instance of the MetadataModel class')

        self._spectra = None
        self._metadata = None
        self._metadata_model = copy.deepcopy(metadata_model)

    def load(self, spectra, metadata,
             error_prefix=''):

        metadata = check.check_metadata(metadata, self._metadata_model, error_prefix)
        check.check_spectra(spectra, metadata, error_prefix)
        self._spectra = spectra
        self._metadata = metadata

    def load_from_envi(self, path, envi_attribute_map):

        check.is_not_none(path,
                          object_name='path')
        check.is_str(path,
                     path=True,
                     object_name='path')
        check.is_not_none(envi_attribute_map,
                          object_name='envi_attribute_map')
        check.is_dict(envi_attribute_map,
                      values=self._metadata_model.get_mandatory_attribute_ids(),
                      object_name='envi_attribute_map')

        spectra, metadata = read_envi_library(path)
        metadata_correct = dict()

        for item in envi_attribute_map.items():

            metadata_correct[item[1]] = copy.deepcopy(metadata[item[0]])

        self.load(spectra, metadata_correct,
                  error_prefix=path + ': ')

    def save_to_envi(self, path, envi_attribute_map,
                     additional_envi_attributes=None):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        check.is_not_none(path,
                          object_name='path')
        check.is_str(path,
                     dir=True,
                     object_name='path')
        check.is_not_none(envi_attribute_map,
                          object_name='envi_attribute_map')
        check.is_dict(envi_attribute_map,
                      values=self._metadata_model.get_mandatory_attribute_ids(),
                      object_name='envi_attribute_map')
        check.is_dict(additional_envi_attributes,
                      object_name='additional_envi_attributes')

        metadata = dict()

        for item in envi_attribute_map.items():

            metadata[item[0]] = copy.deepcopy(self._metadata[item[1]])

        if additional_envi_attributes is not None:

            for item in additional_envi_attributes.items():

                metadata[item[0]] = item[1]

        save_envi_library(path, self.get_spectra(), metadata)

    def get_spectra(self,
                    s_filter=None,
                    b_filter=None):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        check.is_array_like(s_filter,
                            dtype=bool,
                            dimensions=1,
                            size=self._metadata['spectra'],
                            object_name='s_filter')
        check.is_array_like(b_filter,
                            dtype=bool,
                            dimensions=1,
                            size=self._metadata['bands'],
                            object_name='b_filter')

        if s_filter is None:
            s_filter = [True for s in range(self._metadata['spectra'])]
        if b_filter is None:
            b_filter = [True for b in range(self._metadata['bands'])]

        s_ind = np.where(s_filter)[0]
        b_ind = np.where(b_filter)[0]
        spectra = copy.deepcopy(self._spectra)
        spectra = spectra[s_ind, :]
        spectra = spectra[:, b_ind]
        return spectra

    def get_resampled_spectra(self, wavelength, wavelength_scale_factor,
                              fwhm=None,
                              wavelength_inverse_unit=False,
                              drop_nodata=True,
                              fill_nodata=None,
                              band_overlap_threshold=0.5,
                              raise_insufficient_overlap=False,
                              error_prefix='',
                              fill_insufficient_overlap=None,
                              s_filter=None,
                              b_filter=None
                              ):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        check.is_not_none(wavelength,
                          object_name='wavelength')
        check.is_not_none(wavelength_scale_factor,
                          object_name='wavelength_scale_factor')
        check.is_array_like(wavelength,
                            dtype=float,
                            dimensions=1,
                            object_name='wavelength',
                            repetition_allowed=False)
        wavelength = np.array(wavelength, dtype=float)
        check.is_float(wavelength_scale_factor,
                       g=0,
                       object_name='wavelength_scale_factor')

        if fwhm is None:
            diff = np.diff(wavelength)
            diff1 = np.append(diff[0], diff)
            diff2 = np.append(diff, diff[-1])
            fwhm = np.minimum(diff1, diff2)

        check.is_bool(wavelength_inverse_unit,
                      object_name='wavelength_inverse_unit')
        check.is_bool(drop_nodata,
                      object_name='drop_nodata')
        check.is_float(fill_nodata,
                       object_name='fill_nodata')
        check.is_not_one_of(fill_nodata, [np.nan],
                            object_name='fill_nodata')

        spectra = self.get_spectra(s_filter=s_filter,
                                   b_filter=b_filter)
        spectra *= self._metadata['reflectance scale factor']

        # get the library's spectral bands wavelength and FWHM in the specified unit
        # first convert them to the SI unit
        wsf = float(self._metadata['wavelength scale factor'])
        wl = np.array(self._metadata['wavelength']) * wsf
        fl = np.array(self._metadata['fwhm']) * wsf
        if self._metadata['wavelength inverse unit']:
            wl = wl ** -1
            fl = fl ** -1

        # then convert them to the specified unit
        wl /= wavelength_scale_factor
        fl /= wavelength_scale_factor
        if wavelength_inverse_unit:
            wl = wl ** -1
            fl = fl ** -1

        # address nodata values
        spectra[np.isnan(spectra)] = -1
        spectra[spectra > 1] = -1
        spectra[spectra < 0] = np.nan

        if fill_nodata is not None:
            spectra[np.isnan(spectra)] = fill_nodata

        if np.any(np.isnan(spectra), axis=None) and drop_nodata:
            del_ind = np.where(np.any(np.isnan(spectra), axis=0))[0]
            wl = np.delete(wl, del_ind)
            fl = np.delete(fl, del_ind)
            spectra = np.delete(spectra, del_ind, axis=1)

        # perform the spectral resampling resampling
        resampled_spectra = spectral_resampling(wavelength, fwhm, wl, fl, spectra,
                                                band_overlap_threshold=band_overlap_threshold,
                                                fill_insufficient_overlap=fill_insufficient_overlap,
                                                raise_insufficient_overlap=raise_insufficient_overlap,
                                                error_prefix=error_prefix)

        return resampled_spectra

    def get_metadata(self):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        return copy.deepcopy(self._metadata)

    def get_metadata_model(self):

        return copy.deepcopy(self._metadata_model)

    def get_attribute_ids(self):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        return copy.deepcopy(list(self._metadata.keys()))

    def get_spectrum_ids(self):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        return copy.deepcopy(self._metadata['sid'])

    def get_band_ids(self):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        return copy.deepcopy(self._metadata['bid'])

    def get_library_id(self):

        if self._spectra is None:
            raise Exception('the spectral library must have loaded spectra and metadata to perform this operation')

        return copy.deepcopy(self._metadata['lid'])


class SpectralLibraryCollection:

    def __init__(self, metadata_model,
                 description=None):

        if not isinstance(metadata_model, MetadataModel):
            raise Exception('the metadata model must be an instance of the MetadataModel class')

        check.is_str(description,
                     object_name='description')

        # initialize the instance attributes
        self._slc = {}
        self._merged_library = None
        self.description = description
        self._metadata_model = copy.deepcopy(metadata_model)

        # initialize the indices
        l_index_columns = []
        s_index_columns = ['lid']
        b_index_columns = ['lid']

        for aid in self._metadata_model.get_attribute_ids():

            attribute = self._metadata_model.get_attribute(aid)

            if attribute._atype == 'l' and attribute._aid not in l_index_columns:
                l_index_columns.append(aid)
            if attribute._atype == 's' and attribute._aid not in s_index_columns:
                s_index_columns.append(aid)
            if attribute._atype == 'b' and attribute._aid not in b_index_columns:
                b_index_columns.append(aid)

        # l-index
        self._l_index = pd.DataFrame(data=None,
                                     columns=l_index_columns)

        # s-index
        self._s_index = pd.DataFrame(data=None,
                                     columns=s_index_columns)

        # b-index
        self._b_index = pd.DataFrame(data=None,
                                     columns=b_index_columns)

    def _update_indices(self,
                        lid=None):

        if isinstance(lid, list) or isinstance(lid, tuple) or isinstance(lid, np.ndarray):
            lids = lid
        elif lid is None:
            lids = list(self._slc.keys())
        else:
            lids = [lid]

        for lid in lids:

            if lid not in list(self._slc.keys()):
                raise Exception('library with lid {} not found'.format(lid))

        for lid in lids:

            # delete rows in the indices corresponding to lid
            if lid in self._l_index['lid'].values:
                self._l_index = self._l_index.loc[self._l_index['lid'] != lid]
            if lid in self._s_index['lid'].values:
                self._s_index = self._s_index.loc[self._s_index['lid'] != lid]
            if lid in self._b_index['lid'].values:
                self._b_index = self._b_index.loc[self._b_index['lid'] != lid]

            # get the library metadata
            metadata = self._slc[lid].get_metadata()
            metadata_model_aid_list = self._metadata_model.get_attribute_ids()
            library_aid_list = self._slc[lid].get_attribute_ids()

            # produce the updated part of the library index and merge with the existing index
            columns = self._l_index.columns.values
            l_index = pd.DataFrame(data=None, columns=columns, index=range(1))

            for aid in metadata_model_aid_list:

                # if the attribute is present in the library metadata, add it to the index
                # else use the nodata value (which must be specified for non-mandatory attributes)
                attribute = self._metadata_model.get_attribute(aid)
                if attribute._atype == 'l':
                    if aid in library_aid_list:
                        l_index[aid] = metadata[aid]
                    else:
                        l_index[aid] = attribute._nodata_value
                    l_index[aid] = l_index[aid].astype(attribute._dtype)

            self._l_index = pd.concat((self._l_index, l_index), axis=0, ignore_index=True)

            # produce the updated part of the spectrum index and merge with the existing index
            columns = self._s_index.columns.values
            s_index = pd.DataFrame(data=None, columns=columns, index=range(metadata['spectra']))

            for aid in metadata_model_aid_list:

                # if the attribute is present in the library metadata, add it to the index
                # else use the nodata value (which must be specified for unprotected optional attributes)
                attribute = self._metadata_model.get_attribute(aid)
                if attribute._atype == 's':
                    if aid in library_aid_list:
                        s_index[aid] = metadata[aid]
                    else:
                        s_index[aid] = attribute._nodata_value
                    s_index[aid] = s_index[aid].astype(attribute._dtype)

            s_index['lid'] = lid
            self._s_index = pd.concat((self._s_index, s_index), axis=0, ignore_index=True)

            # produce the updated part of the band index and merge it with the existing index
            columns = self._b_index.columns.values
            b_index = pd.DataFrame(data=None, columns=columns, index=range(metadata['bands']))

            for aid in metadata_model_aid_list:

                # if the attribute is present in the library metadata, add it to the index
                # else use the nodata value (which must be specified for non-mandatory attributes)
                attribute = self._metadata_model.get_attribute(aid)
                if attribute._atype == 'b':
                    if aid in library_aid_list:
                        b_index[aid] = metadata[aid]
                    else:
                        b_index[aid] = attribute._nodata_value
                    b_index[aid] = b_index[aid].astype(attribute._dtype)

            b_index['lid'] = lid
            self._b_index = pd.concat((self._b_index, b_index), axis=0, ignore_index=True)

    def get_library_ids(self):

        return list(self._slc.keys())

    def add_library(self, spectral_library,
                    overwrite=False):

        if not isinstance(spectral_library, SpectralLibrary):
            raise Exception('spectral_library must be an instance of the SpectralLibrary class')

        if spectral_library._spectra is None:
            raise Exception('spectral_library must have loaded spectra and metadata')

        if check.compare_metadata_models(spectral_library._metadata_model, self._metadata_model):
            raise Exception('spectral_library must have the same metadata model as the Spectral Library Collection')

        check.is_bool(overwrite,
                      object_name='overwrite')

        lid_list = self.get_library_ids()
        lid = spectral_library.get_library_id()

        if (lid not in lid_list) or (lid in lid_list and overwrite):
            self._slc[lid] = copy.deepcopy(spectral_library)
            self._update_indices(lid=lid)
        elif lid in lid_list and not overwrite:
            raise Exception('Spectral library with lid {} is already present in the spectral library collection. '
                            'Set the keyword argument overwrite to True to replace the existing library.'.format(lid))

    def add_library_from_envi(self, path, envi_attribute_map,
                              overwrite=False):

        spectral_library = SpectralLibrary(self._metadata_model)
        spectral_library.load_from_envi(path, envi_attribute_map)
        self.add_library(spectral_library,
                         overwrite=overwrite)

    def get_library(self, lid):

        check.is_not_none(lid,
                          object_name='lid')
        check.is_str(lid,
                     object_name='lid')
        lid_list = self.get_library_ids()
        if lid not in lid_list:
            raise Exception('lid {} not found in the spectral library collection')

        return copy.deepcopy(self._slc[lid])

    def remove_library(self, lid):

        check.is_not_none(lid,
                          object_name='lid')
        check.is_str(lid,
                     object_name='lid')
        lid_list = self.get_library_ids()
        if lid not in lid_list:
            raise Exception('lid {} not found in the spectral library collection')

        del self._slc[lid]
        self._l_index = self._l_index.loc[self._l_index['lid'].values != lid]
        self._s_index = self._s_index.loc[self._s_index['lid'].values != lid]
        self._b_index = self._b_index.loc[self._b_index['lid'].values != lid]

    def get_l_index(self):

        return copy.deepcopy(self._l_index)

    def get_s_index(self):

        return copy.deepcopy(self._s_index)

    def get_b_index(self):

        return copy.deepcopy(self._b_index)

    def set_l_filter(self, dataframe):

        check.is_not_none(dataframe,
                          object_name='dataframe')
        check.is_dataframe(dataframe,
                           columns=['lid', 'l-filter'],
                           object_name='dataframe')
        dataframe = dataframe[['lid', 'l-filter']]

        # use pandas merge to join the input to the original index
        # the intermediate conversion to float is needed to cope with missing values (np.nan is of type float)
        dataframe.loc[:, 'l-filter'] = dataframe.loc[:, 'l-filter'].values.astype(float)
        self._l_index = pd.merge(self._l_index, dataframe, how='left', on='lid')

        # fill missing rows with original values
        ind = np.where(np.isnan(self._l_index['l-filter_y'].values))[0]
        self._l_index.loc[ind, 'l-filter_y'] = self._l_index.loc[ind, 'l-filter_x']

        # convert back to bool and remove the intermediate columns
        self._l_index['l-filter'] = self._l_index['l-filter_y'].values.astype(bool)
        self._l_index.drop(['l-filter_x', 'l-filter_y'],
                           axis=1,
                           inplace=True)

    def set_s_filter(self, dataframe):

        check.is_not_none(dataframe,
                          object_name='dataframe')
        check.is_dataframe(dataframe,
                           columns=['lid', 'sid', 's-filter'],
                           object_name='dataframe')
        dataframe = dataframe[['lid', 'sid', 's-filter']]

        # use pandas merge to join the input to the original index
        # the intermediate conversion to float is needed to cope with missing values (np.nan is of type float)
        dataframe['s-filter'] = dataframe['s-filter'].values.astype(float)
        self._s_index = pd.merge(self._s_index, dataframe,
                                 how='left',
                                 on=['lid', 'sid'])

        # fill missing rows with original values
        ind = np.where(np.isnan(self._s_index['s-filter_y'].values))[0]
        self._s_index.loc[ind, 's-filter_y'] = self._s_index.loc[ind, 's-filter_x']

        # convert back to bool and remove the intermediate columns
        self._s_index['s-filter'] = self._s_index['s-filter_y'].values.astype(bool)
        self._s_index.drop(['s-filter_x', 's-filter_y'],
                           axis=1,
                           inplace=True)

    def set_b_filter(self, dataframe):

        check.is_not_none(dataframe,
                          object_name='dataframe')
        check.is_dataframe(dataframe,
                           columns=['lid', 'bid', 'b-filter'],
                           object_name='dataframe')
        dataframe = dataframe[['lid', 'bid', 'b-filter']]

        # use pandas merge to join the input to the original index
        # the intermediate conversion to float is needed to cope with missing values (np.nan is of type float)
        dataframe['b-filter'] = dataframe['b-filter'].values.astype(float)
        self._b_index = pd.merge(self._b_index, dataframe,
                                 how='left',
                                 on=['lid', 'bid'])

        # fill missing rows with original values
        ind = np.where(np.isnan(self._b_index['b-filter_y'].values))[0]
        self._b_index.loc[ind, 'b-filter_y'] = self._b_index.loc[ind, 'b-filter_x']

        # convert back to bool and remove the intermediate columns
        self._b_index['b-filter'] = self._b_index['b-filter_y'].values.astype(bool)
        self._b_index.drop(['b-filter_x', 'b-filter_y'],
                           axis=1,
                           inplace=True)

    def reset_filters(self):

        self.reset_l_filter()
        self.reset_s_filter()
        self.reset_b_filter()

    def reset_l_filter(self):

        for lid in self.get_library_ids():

            con = self._l_index['lid'].values == lid
            l_filter = self.get_library(lid).get_metadata()['l-filter']
            self._l_index.loc[con, 'l-filter'] = l_filter

    def reset_s_filter(self):

        for lid in self.get_library_ids():

            con = self._s_index['lid'].values == lid
            s_filter = self.get_library(lid).get_metadata()['s-filter']
            self._s_index.loc[con, 's-filter'] = s_filter

    def reset_b_filter(self):

        for lid in self.get_library_ids():

            con = self._b_index['lid'].values == lid
            b_filter = self.get_library(lid).get_metadata()['b-filter']
            self._b_index.loc[con, 'b-filter'] = b_filter

    def set_class_labels(self, dataframe):

        check.is_not_none(dataframe,
                          object_name='dataframe')
        check.is_dataframe(dataframe,
                           columns=['lid', 'sid', 'class label'],
                           object_name='dataframe')
        dataframe = dataframe[['lid', 'sid', 'class label']]

        # use pandas merge to join the input to the original index
        self._s_index = pd.merge(self._s_index, dataframe,
                                 how='left',
                                 on=['lid', 'sid'])

        # fill missing rows with original values
        ind = np.where(self._s_index['class label_y'].values == 'nan')[0]
        self._s_index.loc[ind, 'class label_y'] = self._s_index.loc[ind, 'class label_x']

        # remove the intermediate columns
        self._s_index['class label'] = self._s_index['class label_y'].values.astype(str)
        self._s_index.drop(['class label_y', 'class label_x'],
                           axis=1,
                           inplace=True)

    def set_class_labels_with_attribute(self, aid,
                                        hierarchy_level=0):

        check.is_not_none(aid,
                          object_name='aid')
        check.is_str(aid,
                     object_name='aid')
        check.is_one_of(aid, self._metadata_model.get_attribute_ids())

        attribute = self._metadata_model.get_attribute(aid)
        if not attribute._atype == 's':
            raise Exception('the specified attribute must be spectra-specific')

        self._s_index['class label'] = self._s_index[aid].values

        # if the attribute is hierarchical, relabel to the specified level
        if attribute._value_hierarchy is not None:

            check.is_int(hierarchy_level,
                         object_name='hierarchy_level',
                         ge=-1)
            hierarchy = np.array(attribute._value_hierarchy)

            if hierarchy_level == -1:
                hierarchy_level = hierarchy.shape[1] - 1

            if hierarchy_level > hierarchy.shape[1] - 1:
                hierarchy_level = hierarchy.shape[1] - 1

            class_labels = self._s_index['class label'].values
            class_labels_unique = np.unique(class_labels)
            level_labels = hierarchy[:, hierarchy_level]

            for clu in class_labels_unique:

                con = self._s_index['class label'] == clu

                if clu not in level_labels:
                    rows, cols = np.where(hierarchy == clu)
                    col = cols.max()
                    row = rows[0]
                    if col > hierarchy_level:
                        new_label = hierarchy[row, hierarchy_level]
                        self._s_index.loc[con, 'class label'] = new_label.encode('utf-8').decode()
                    else:
                        self._s_index.loc[con, 'class label'] = np.nan

    def reset_class_labels(self):

        for lid in self.get_library_ids():

            con = self._s_index['lid'].values == lid
            class_labels = self.get_library(lid).get_metadata()['class label']
            self._s_index.loc[con, 'class label'] = class_labels

    def get_metadata_model(self):

        return copy.deepcopy(self._metadata_model)

    def merge_libraries(self, new_lid, wavelength, wavelength_scale_factor,
                        fwhm=None,
                        wavelength_inverse_unit=False,
                        drop_nodata=True,
                        fill_nodata=None,
                        band_overlap_threshold=0.5,
                        raise_insufficient_overlap=False,
                        fill_insufficient_overlap=None,
                        drop_unlabeled=False,
                        fill_unlabeled=None,
                        use_filters=False,
                        source_lid_in_sid=True,
                        id_separator='_',
                        drop_optional_spectrum_attributes=False):

        if len(self._slc) == 0:
            raise Exception('The spectral library collection must contain at least one library to perform this'
                             'operation')

        check.is_not_none(new_lid,
                          object_name='new_lid')
        check.is_bool(drop_unlabeled,
                      object_name='drop_unlabeled')
        check.is_str(fill_unlabeled,
                     object_name='fill_unlabeled')
        check.is_bool(use_filters,
                      object_name='use_filters')
        check.is_bool(source_lid_in_sid,
                      object_name='source_lid_in_sid')
        check.is_str(id_separator,
                     length=1,
                     object_name='id_separator')
        check.is_bool(drop_optional_spectrum_attributes,
                      object_name='drop_optional_spectrum_attributes')

        if fwhm is None:
            diff = np.diff(wavelength)
            diff1 = np.append(diff[0], diff)
            diff2 = np.append(diff, diff[-1])
            fwhm = np.minimum(diff1, diff2)

        spectra_coll = []
        s_index_coll = []
        lid_list = self.get_library_ids()

        for lid in lid_list:

            # apply the library filter
            stop = False
            if use_filters:
                con = self._l_index['lid'].values == lid
                filter_value = self._l_index.loc[con, 'l-filter'].values[0]
                if not filter_value:
                    stop = True

            if not stop:

                # get the library and its metadata
                library = self.get_library(lid)
                metadata = library.get_metadata()

                # get copies of the parts of the s-index and b-index that correspond to the library
                s_index = self.get_s_index()
                b_index = self.get_b_index()
                s_con = s_index['lid'].values == lid
                b_con = b_index['lid'].values == lid
                s_index = s_index.loc[s_con]
                b_index = b_index.loc[b_con]

                # get the s- and b-filter
                if use_filters:
                    s_filter = s_index['s-filter'].values
                    b_filter = b_index['b-filter'].values
                else:
                    s_filter = np.array([True for s in range(metadata['spectra'])])
                    b_filter = np.array([True for b in range(metadata['bands'])])

                # filter or fill out unlabelled spectra
                if drop_unlabeled:
                    class_labels = s_index['class label'].values.astype(str)
                    ind = np.where(class_labels == 'nan')[0]
                    s_filter[ind] = False
                if fill_unlabeled is not None:
                    class_labels = s_index['class label'].values.astype(str)
                    ind = np.where(class_labels == 'nan')[0]
                    s_index.loc[s_index.index[ind], 'class label'] = fill_unlabeled

                # _check if there are still spectra and bands left after applying the filters
                if (np.sum(s_filter) > 0) and (np.sum(b_filter) > 0):

                    # get the resampled spectra
                    resampled_spectra = library.get_resampled_spectra(
                        wavelength, wavelength_scale_factor,
                        fwhm=fwhm,
                        wavelength_inverse_unit=wavelength_inverse_unit,
                        drop_nodata=drop_nodata,
                        fill_nodata=fill_nodata,
                        band_overlap_threshold=band_overlap_threshold,
                        raise_insufficient_overlap=raise_insufficient_overlap,
                        fill_insufficient_overlap=fill_insufficient_overlap,
                        s_filter=s_filter,
                        b_filter=b_filter)

                    # _check if there are no-data values in the resampled spectra, drop the library if there are
                    if not np.any(np.isnan(resampled_spectra), axis=None):

                        # add the resampled spectra to the collections
                        spectra_coll.append(resampled_spectra)

                        # get the unfiltered part of the s-index, adjust its spectrum identifiers, and add it to the
                        # collection
                        ind = np.where(s_filter)[0]
                        s_index = s_index.loc[s_index.index[ind]]
                        sid_list = s_index['sid'].values
                        if source_lid_in_sid:
                            new_sid = [lid + id_separator + s for s in sid_list]
                            s_index['sid'] = new_sid
                        s_index_coll.append(s_index)

        # merge the collections
        spectra = np.concatenate(spectra_coll, axis=0)
        s_index = pd.concat(s_index_coll,
                            axis=0,
                            ignore_index=True)
        s_index.reset_index(inplace=True)

        # make a spectral library to hold the merged library
        # use a copy of the metadata model from which the unprotected library and band attributes are removed
        metadata_model = self.get_metadata_model()
        attribute_ids = metadata_model.get_attribute_ids()

        for aid in attribute_ids:

            attribute = metadata_model.get_attribute(aid)
            con1 = (attribute._atype == 'l') or (attribute._atype == 'b')
            con2 = not attribute._protected
            if con1 and con2:
                metadata_model.remove_attribute(aid)

        merged_library = SpectralLibrary(metadata_model)

        # add the protected attributes to the metadata dictionary
        metadata = dict()
        metadata['lid'] = new_lid
        metadata['wavelength'] = wavelength
        metadata['fwhm'] = fwhm
        metadata['wavelength scale factor'] = wavelength_scale_factor
        metadata['wavelength inverse unit'] = wavelength_inverse_unit
        metadata['reflectance scale factor'] = 1.
        metadata['spectra'] = spectra.shape[0]
        metadata['bands'] = spectra.shape[1]
        metadata['sid'] = s_index['sid'].values
        metadata['class label'] = s_index['class label'].values
        metadata['l-filter'] = True
        metadata['s-filter'] = [True for s in range(spectra.shape[0])]
        metadata['b-filter'] = [True for b in range(spectra.shape[1])]

        # add the remaining spectrum attributes
        spectrum_attribute_ids = self.get_metadata_model().get_spectrum_attribute_ids()

        for aid in spectrum_attribute_ids:

            attribute = self.get_metadata_model().get_attribute(aid)
            if not attribute._protected and attribute._mandatory:
                metadata[aid] = s_index[aid].values
            if not attribute._protected and not attribute._mandatory:
                if not drop_optional_spectrum_attributes:
                    metadata[aid] = s_index[aid].values

        # load the spectra and metadata to the library
        merged_library.load(spectra, metadata)
        self._merged_library = merged_library

    def get_merged_library(self):

        if self._merged_library is None:
            raise Exception('The merged library must first be created to perform this operation')

        return copy.deepcopy(self._merged_library)
