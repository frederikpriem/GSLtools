import os
import copy
import pickle
import numpy as np
import pandas as pd
from gsltools.io import read_envi_library, read_envi_header, save_envi_library, envi_to_dtype, dtype_to_envi
from gsltools.resample import spectral_resampling


class SpectralLibrary:

    def __init__(self, lid, name, spectra, metadata):

        self.lid = lid
        self.name = name
        self.spectra = spectra
        self.metadata = metadata


class GenericSpectralLibrary:

    def __init__(self, class_system, metadata_model):

        """
        the metadata model is a dictionary containing at least all mandatory and no reserved attributes
        the classification system is a Pandas data frame conforming to the following rules:

            1. each row corresponds to a class sequence, each column to a classification level
            2. the data frame columns are ordered from general (left) to specific (right)
            3. each row and column is unique
            4. classes are nested, i.e. a higher level class can only fit in 1 class of each lower level
            5. specificity is non-descending, e.g. ... vegetation -> vegetation -> grass ... is allowed but
               ... vegetation -> grass -> vegetation ... isn't
            6. class specification is exhaustive, i.e. splitting a class in subclasses or changing its label in a
               higher level implies that the resulting higher-level labels are all different from the original
               lower-level label, e.g. the following rows
               ...  level n      level n + 1   ...
               ...  vegetation   grass         ...
               ...  vegetation   vegetation    ...
               are not allowed to co-exist in the same classification system
        """

        # initialize the class instance attributes needed to perform initial checks
        mandatory_attributes = ['l_wavelength_unit',
                                'l_reflectance_scale_factor',
                                'b_wavelength',
                                'b_fwhm',
                                's_class_label'
                                ]
        self._mandatory_attributes = np.array(mandatory_attributes)

        reserved_attributes = ['lid',
                               'sid',
                               'bid',
                               's_class_relabel',
                               'l_filter',
                               's_filter',
                               'b_filter']
        self._reserved_attributes = np.array(reserved_attributes)

        allowed_wavelength_units = ['micrometer',
                                    'micrometers',
                                    'micrometre',
                                    'micrometres',
                                    'nanometer',
                                    'nanometers',
                                    'nanometre',
                                    'nanometres',
                                    'wavenumber',
                                    'wavenumbers',
                                    'wave number',
                                    'wave numbers']
        self._allowed_wavelength_units = np.array(allowed_wavelength_units)

        # check the classification system
        self._check_class_system(class_system)

        # check the metadata model
        self._check_metadata_model(metadata_model)

        # initialize the remaining class instance attributes
        self.entries = {}
        self.merged_state = None
        self._class_system = class_system
        self._metadata_model = metadata_model
        self.l_index = None
        self.s_index = None
        self.b_index = None

        si_scale_factor = [10 ** -6,
                           10 ** -6,
                           10 ** -6,
                           10 ** -6,
                           10 ** -9,
                           10 ** -9,
                           10 ** -9,
                           10 ** -9,
                           10 ** -2,
                           10 ** -2,
                           10 ** -2,
                           10 ** -2
                           ]
        self._si_scale_factor = np.array(si_scale_factor)

        inverse_unit = [False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True]
        self._inverse_unit = np.array(inverse_unit)

        # initialize the GSL indices
        self._initialize_indices()

    @staticmethod
    def _check_class_system(class_system):

        if not isinstance(class_system, pd.DataFrame):
            raise TypeError('class_system must be a Pandas Dataframe object')
        else:

            # check if classification system rows and columns are unique
            uni, h_cnt = np.unique(class_system.values,
                                   return_counts=True,
                                   axis=0)
            uni, v_cnt = np.unique(class_system.values,
                                   return_counts=True,
                                   axis=1)
            if np.any(h_cnt > 1) or np.any(v_cnt > 1):
                raise ValueError('the classification system rows and columns must be unique')

            # check if classes are nested
            rows, cols = class_system.values.shape

            for col in range(1, cols):

                col_values = class_system.values[:, col]
                col_values_unique = np.unique(col_values)

                for cvu in col_values_unique:

                    rows = np.where(col_values == cvu)[0]
                    prev_col_values = class_system.values[rows, col - 1]
                    uni_prev_col_values = np.unique(prev_col_values)
                    if len(uni_prev_col_values) > 1:
                        raise ValueError('the classification system must be hierarchical (general -> specific) and nested')

            # check if specificity is non-descending
            for row in range(rows):

                prev_class = []

                for col in range(1, cols):

                    if class_system.values[row, col] != class_system.values[row, col - 1]:
                        prev_class.append(class_system.values[row, col - 1])
                    if class_system.values[row, col] in prev_class:
                        raise ValueError('the classification system must have non-decreasing specificity')

            # check if class specifications are exhaustive
            for col in range(cols - 1):

                col_values = class_system.values[:, col]
                col_values_unique = np.unique(col_values)

                for cvu in col_values_unique:

                    rows = np.where(col_values == cvu)[0]
                    next_col_values = class_system.values[rows, col + 1]
                    if (cvu in next_col_values) and (len(np.unique(next_col_values)) > 1):
                        raise ValueError('class specification must be exhaustive')

        return 0

    def _check_metadata_model(self, metadata_model):

        if not isinstance(metadata_model, dict):
            raise TypeError('metadata_model must be a dictionary')

        for attr in self._mandatory_attributes:

            if attr not in metadata_model.keys():
                raise ValueError('{} must be defined in the metadata model'.format(attr))

        for attr in metadata_model.keys():

            if not isinstance(attr, str):
                raise TypeError('metadata attribute names must be strings')
            if attr[0] not in ['l', 's', 'b']:
                raise ValueError("metadata attribute names must start with 'l' (library-specific), 's' (spectra-specific) or 'b' (band-specific)")
            if attr in self._reserved_attributes:
                raise ValueError("{} is a reserved metadata attribute name".format(attr))

        return 0

    def _initialize_indices(self):

        l_index_columns = ['lid',
                           'l_filter'
                           ]

        s_index_columns = ['lid',
                           'sid',
                           's_filter',
                           's_class_relabel'
                           ]

        b_index_columns = ['lid',
                           'bid',
                           'b_filter'
                           ]

        for attr in self._metadata_model.keys():

            if attr[0] == 'l':
                l_index_columns.append(attr)
            if attr[0] == 's':
                s_index_columns.append(attr)
            if attr[0] == 'b':
                b_index_columns.append(attr)

        self.l_index = pd.DataFrame(data=None, columns=l_index_columns)
        self.s_index = pd.DataFrame(data=None, columns=s_index_columns)
        self.b_index = pd.DataFrame(data=None, columns=b_index_columns)

    def _update_indices(self, lid=None):

        # process and check the input
        if isinstance(lid, list) or isinstance(lid, tuple) or isinstance(lid, np.ndarray):
            lids = lid
        elif lid is None:
            lids = self.entries.keys()
        else:
            lids = [lid]

        for lid in lids:

            if lid not in self.entries.keys():
                raise ValueError('library with lid {} not found'.format(lid))

        # get the metadata model
        mdm = self._metadata_model

        for lid in lids:

            # delete entries in indices corresponding to lid
            if lid in self.l_index['lid'].values:
                self.l_index = self.l_index.loc[self.l_index['lid'] != lid]
            if lid in self.s_index['lid'].values:
                self.s_index = self.s_index.loc[self.s_index['lid'] != lid]
            if lid in self.b_index['lid'].values:
                self.b_index = self.b_index.loc[self.b_index['lid'] != lid]

            # get the library metadata
            md = self.entries[lid].metadata

            # update the library index
            l_index = pd.DataFrame(data=None)
            l_index['lid'] = lid
            l_index['l_filter'] = True

            l_attr = self.l_index.columns.values
            l_attr = l_attr[l_attr != 'lid']
            l_attr = l_attr[l_attr != 'l_filter']

            for attr in l_attr:

                l_index[attr] = md[mdm[attr]]

            self.l_index = pd.concat((self.l_index, l_index), axis=0, ignore_index=True)

            # update the spectra index
            n_spectra = len(md[mdm['class_label']])
            s_index = pd.DataFrame(data=None)
            s_index['lid'] = np.array([lid] * n_spectra)
            s_index['sid'] = np.arange(n_spectra).astype(int)
            s_index['s_filter'] = True
            s_index['s_class_relabel'] = None

            s_attr = self.s_index.columns.values
            s_attr = s_attr[s_attr != 'lid']
            s_attr = s_attr[s_attr != 'sid']
            s_attr = s_attr[s_attr != 's_filter']
            s_attr = s_attr[s_attr != 's_class_relabel']

            for attr in s_attr:

                s_index[attr] = md[mdm[attr]]

            self.s_index = pd.concat((self.s_index, s_index), axis=0, ignore_index=True)

            # update the band index
            n_bands = len(md[mdm['b_wavelength']])
            b_index = pd.DataFrame(data=None)
            b_index['lid'] = np.array([lid] * n_bands)
            b_index['bid'] = np.arange(n_bands).astype(int)
            b_index['b_filter'] = True

            b_attr = self.b_index.columns.values
            b_attr = b_attr[b_attr != 'lid']
            b_attr = b_attr[b_attr != 'bid']
            b_attr = b_attr[b_attr != 'b_filter']

            for attr in b_attr:

                b_index[attr] = md[mdm[attr]]

            self.b_index = pd.concat((self.b_index, b_index), axis=0, ignore_index=True)
        
    def _check_data(self, lid, name, spectra, metadata,
                    raise_error_if_missing=True):
        
        msg_start = 'library {} ({}): '.format(name, lid)

        # check lid
        if lid is None:
            raise TypeError(msg_start + "lid mustn't be None")
        if len(str(lid)) == 0:
            raise ValueError(msg_start + "lid mustn't be empty")

        # check spectra
        if not isinstance(spectra, np.ndarray):
            raise TypeError(msg_start + 'spectra must be a Numpy array')
        if not len(spectra.shape) == 2:
            raise ValueError(msg_start + 'spectra must have two dimensions, i.e. observations (rows) and bands (columns)')
        if spectra.shape[0] == 0 or spectra.shape[1] == 0:
            raise ValueError(msg_start + "the number of rows and columns in spectra must be greater than zero")
        try:
            spectra.astype(float)
        except:
            raise TypeError(msg_start + 'spectra reflectance values must be integer or float, and void of undefined values')

        # check metadata
        mdm = self._metadata_model
        n_spectra, n_bands = spectra.shape

        # check if mandatory attributes are present
        for attr in self._mandatory_attributes:

            if mdm[mdm[attr]] not in metadata:
                raise ValueError(msg_start + '{} ({}) attribute not found in metadata'.format(attr, mdm[attr]))

        # check the consistency of each attribute
        for attr in mdm.keys():

            # check if attribute is missing, and fill values if applicable
            if (mdm[attr] not in metadata) and raise_error_if_missing:
                raise ValueError(msg_start + '{} ({}) attribute not found in metadata'.format(attr, mdm[attr]))
            elif (mdm[attr] not in metadata) and not raise_error_if_missing:
                if attr[0] == 'l':
                    metadata[mdm[attr]] = None
                if attr[0] == 's':
                    metadata[mdm[attr]] = np.array([None] * n_spectra)
                if attr[0] == 'b':
                    metadata[mdm[attr]] = np.array([None] * n_bands)

            md_entry = metadata[mdm[attr]]
            is_array = isinstance(md_entry, list) or isinstance(md_entry, tuple) or isinstance(md_entry, np.ndarray)

            # check data types/value domains of mandatory attributes
            if attr == 'l_wavelength_unit':
                if not isinstance(md_entry, str):
                    raise TypeError(msg_start + "{} ({}) attribute must be a string".format(attr, mdm[attr]))
                awu = self._allowed_wavelength_units
                if md_entry.lower() not in awu:
                    raise TypeError(msg_start + "{} ({}) attribute must be one of {}".format(attr, mdm[attr], awu))
            if attr == 'l_reflectance_scale_factor':
                try:
                    float(md_entry)
                except ValueError:
                    raise TypeError(msg_start + "{} ({}) attribute must be a float or integer".format(attr, mdm[attr]))
            if attr in ['b_wavelength', 'b_fwhm']:
                try:
                    np.array(md_entry, dtype=float)
                except ValueError:
                    raise TypeError(msg_start + "{} ({}) attribute must contain floats or integers".format(attr, mdm[attr]))

            # check format and length of each attribute
            if attr[0] == 'l' and is_array:
                raise TypeError(msg_start + "{} ({}) attribute mustn't be an array-like".format(attr, mdm[attr]))

            if attr[0] == 's':
                if not is_array:
                    raise TypeError(msg_start + '{} ({}) attribute must be an array-like'.format(attr, mdm[attr]))
                if len(md_entry) != n_spectra:
                    raise ValueError(msg_start + '{} ({}) attribute length must be equal to the number of spectra in the library'.format(attr, mdm[attr]))

            if attr[0] == 'b':
                if not is_array:
                    raise TypeError(msg_start + '{} ({}) attribute must be an array-like'.format(attr, mdm[attr]))
                if len(md_entry) != n_bands:
                    raise ValueError(msg_start + '{} ({}) attribute length must be equal to the number of bands in the library'.format(attr, mdm[attr]))

    def load_library(self, lid, name, spectra, metadata,
                     raise_error_if_missing=True,
                     overwrite=False):

        self._check_data(lid, name, spectra, metadata, raise_error_if_missing=raise_error_if_missing)
        current_lids = self.entries.keys()

        if (lid not in current_lids) or (lid in current_lids and overwrite):
            self.entries[lid] = SpectralLibrary(lid, name, spectra, metadata)
            self._update_indices(lid=lid)

    def load_libraries_from_source(self, libs,
                                   raise_error_if_missing=True,
                                   overwrite=False,
                                   format_iter=None):

        """
        libs is an iterable containing any amount of iterables with 3 elements: lid, name and path to source
        only ENVI libraries (sli and hdr file) can be loaded with this method
        the path may refer to the sli or hdr file
        raise_missing indicates whether an error should be raised for missing optional metadata attributes
        overwrite indicates whether old entries should be overwritten if lids overlap
        format_iter is a dictionary used to cast non-default ENVI header entries to specific types
        """

        # check input
        if not (isinstance(libs, list) or isinstance(libs, tuple)):
            raise TypeError('libs must be a list or tuple')

        for lib in libs:

            if not (isinstance(lib, list) or isinstance(lib, tuple) or isinstance(lib, np.ndarray)):
                raise TypeError('all elements of libs must be array-like')
            if len(lib) != 3:
                raise ValueError('all elements of libs must be array-like with length 3, i.e. lid, name and path')
            if not os.path.exists(lib[2]):
                raise ValueError('{} not found'.format(lib[2]))

        # add the libraries to the GSL
        for lib in libs:

            spectra, metadata = read_envi_library(lib[2], format_iter=format_iter)
            self.load_library(lib[0], lib[1], spectra, metadata,
                              raise_error_if_missing=raise_error_if_missing,
                              overwrite=overwrite)

    def remove_library(self, lid):

        # process and check the input
        if lid is None:
            raise ValueError('lid must be defined')

        lids = None
        if isinstance(lid, list) or isinstance(lid, tuple) or isinstance(lid, np.ndarray):
            lids = lid
        else:
            lids = [lid]

        for lid in lids:

            if lid not in self.entries.keys():
                raise ValueError('library with lid {} not found'.format(lid))

        # remove the library/libraries from the GSL entries and all corresponding rows from the indices
        for lid in lids:

            del self.entries[lid]
            self.l_index = self.l_index.loc[self.l_index['lid'].values != lid]
            self.s_index = self.s_index.loc[self.s_index['lid'].values != lid]
            self.b_index = self.b_index.loc[self.b_index['lid'].values != lid]

    def reset_filter(self, which):

        # process and check the input
        if which not in (None, 'all', 'l_filter', 's_filter', 'b_filter'):
            raise ValueError('if specified which must be all, l_filter, s_filter or b_filter')
        if which is None:
            which = 'all'

        if (which == 'all') or (which == 'l_filter'):
            self.l_index['l_filter'] = True
        if (which == 'all') or (which == 's_filter'):
            self.s_index['s_filter'] = True
        if (which == 'all') or (which == 'b_filter'):
            self.b_index['b_filter'] = True

    def relabel_spectra(self, labeling):

        """
        label is either an array-like of class labels corresponding to unique rows in the classification system or a
        column name of the classification system
        """

        # process the input
        if labeling is None:
            raise ValueError('label must be defined')
        level = None
        if not (isinstance(labeling, list) or isinstance(labeling, tuple) or isinstance(labeling, np.ndarray)):
            level = copy.deepcopy(labeling)
            labeling = None

        # check the input
        if labeling is not None:

            # verify that each given label exists in the classification system
            # also verify that each entry of label corresponds to unique rows in the classification system
            # i.e. you can't include class label corresponding to more than one column on the same row
            rows = []

            for lab in labeling:

                if lab not in self._class_system.values:
                    raise ValueError('{} not found in the classification system'.format(lab))

                rows.append(np.unique(np.where(self._class_system.values == lab)[0]))

            rows = np.concatenate(rows)
            uni, cnt = np.unique(rows, return_counts=True)
            if np.any(cnt > 1):
                raise ValueError("class label mustn't overlap")

        if level is not None:

            # verify that the given level is a column name in the classification system
            if level not in self._class_system.columns.values:
                raise ValueError('{} not found in classification system column names'.format(level))

        # start relabeling
        # first reset previous relabeling
        self.s_index['s_class_relabel'] = None

        # relabel using label
        if labeling is not None:

            class_label = self.s_index['l_class_label'].values
            class_label_unique = np.unique(class_label)

            for clu in class_label_unique:

                con = self.s_index['l_class_label'] == clu

                if clu in labeling:
                    self.s_index.loc[con, 's_class_relabel'] = clu
                elif clu in self._class_system.values:
                    ind = np.where(self._class_system.values == clu)

                    for row, col in zip(ind[0], ind[1]):

                        options = self._class_system.values[row, :col]
                        if len(options) > 0:
                            if np.any(np.isin(options, labeling)):
                                ind2 = np.where(np.isin(options, labeling))[0][0]
                                self.s_index.loc[con, 's_class_relabel'] = options[ind2]
                                break

                    self.s_index.loc[con, 's_class_relabel'] = None
                else:
                    self.s_index.loc[con, 's_class_relabel'] = None
                    
        # relabel using level
        else:

            class_label = self.s_index['l_class_label'].values
            class_label_unique = np.unique(class_label)
            level_label = self._class_system[level].values
            level_col = np.where(self._class_system.columns.values == level)[0][0]

            for clu in class_label_unique:

                con = self.s_index['l_class_label'] == clu

                if clu in level_label:
                    self.s_index.loc[con, 's_class_relabel'] = clu
                elif clu in self._class_system.values:
                    rows, cols = np.where(self._class_system.values == clu)
                    colmax = cols.max()
                    if colmax >= level_col:
                        new_label = self._class_system.values[rows[0], level_col]
                        self.s_index.loc[con, 's_class_relabel'] = new_label
                    else:
                        self.s_index.loc[con, 's_class_relabel'] = None
                else:
                    self.s_index.loc[con, 's_class_relabel'] = None

    def merge(self, wavelength, fwhm, wavelength_unit,
              labeling=None,
              use_original_labels=False,
              reflectance_scale_factor=1,
              reflectance_dtype=np.float32,
              drop_unlabeled=True,
              drop_out_of_bounds=True,
              use_filters=True,
              resample_threshold=0.5,
              fill_missing=False,
              raise_error_if_missing=False):

        # process and check input
        for check in [wavelength, fwhm]:

            if not (isinstance(check, list) or isinstance(check, tuple) or isinstance(check, np.ndarray)):
                raise TypeError("wavelengths and FWHM must be array-like")
            try:
                np.array(check, dtype=float)
            except ValueError:
                raise TypeError("wavelengths and FWHM must be (convertible to) floats")

        if not isinstance(wavelength_unit, str):
            raise TypeError('wavelength_unit must be a string')
        if wavelength_unit.lower() not in self._allowed_wavelength_units:
            raise ValueError('wavelength_unit must be in {}'.format(self._allowed_wavelength_units))

        # relabel the spectra
        if not (use_original_labels or labeling is None):
            self.relabel_spectra(labeling)
        if use_original_labels:
            self.s_index['s_class_relabel'] = self.s_index['s_class_label'].values

        # start collecting the filtered libraries and spectrally resample them
        spec_coll = []
        s_ind_coll = []
        lids = self.entries.keys()
        mdm = self._metadata_model

        for lid in lids:

            # apply library filter
            cont = True
            if use_filters:
                con = self.l_index['lid'] == lid
                filter_value = self.l_index.loc[con, 'l_filter'].values[0]
                if not filter_value:
                    cont = False

            if cont:

                # get this spectral library's metadata, spectrum identifiers and s-index
                sl = self.entries[lid]
                md = sl.metadata
                con = self.s_index['lid'].values == lid
                temp = self.s_index.loc[con, 'sid']
                sid = temp.values
                s_ind = temp.index.values

                # get the spectra with reflectance values scaled as specified
                # first convert to [0, 1] range
                spectra = sl.spectra
                rsf = md[mdm['l_reflectance_scale_factor']]
                spectra /= float(rsf)

                # then convert to the specified range
                spectra *= reflectance_scale_factor

                # make sure that spectra and labels are correctly ordered
                spectra = spectra[sid, :]

                # get the library's wavelengths and FWHM in the specified unit
                # first convert them to SI units
                wu = md[mdm['l_wavelength_unit']].lower()
                wl = np.array(md[mdm['b_wavelength']])
                fw = np.array(md[mdm['b_fwhm']])
                unit_ind = np.where(self._allowed_wavelength_units == wu)[0][0]
                if self._inverse_unit[unit_ind]:
                    wl = wl ** -1
                    fw = fw ** -1
                wl = wl * self._si_scale_factor[unit_ind]
                fw = fw * self._si_scale_factor[unit_ind]

                # then convert them to the specified unit
                unit_ind = np.where(self._allowed_wavelength_units == wavelength_unit.lower())[0][0]
                wl = wl / self._si_scale_factor[unit_ind]
                fw = fw / self._si_scale_factor[unit_ind]
                if self._inverse_unit[unit_ind]:
                    wl = wl ** -1
                    fw = fw ** -1

                # get the relabeled class labels of this library
                con = self.s_index['lid'] == lid
                labels = self.s_index.loc[con, 's_class_relabel'].values

                # apply spectrum and band filters
                if use_filters:

                    con1 = self.s_index['lid'].values == lid
                    con2 = self.s_index['s_filter'].values
                    ind = self.s_index.loc[con1 & con2, 'sid'].values
                    spectra = spectra[ind, :]
                    labels = labels[ind]
                    s_ind = s_ind[ind]

                    con1 = self.b_index['lid'].values == lid
                    con2 = self.b_index['b_filter'].values
                    ind = self.b_index.loc[con1 & con2, 'bid'].values
                    spectra = spectra[:, ind]
                    wl = wl[ind]
                    fw = fw[ind]

                # apply remaining spectrum filters
                if drop_unlabeled:
                    ind = np.where(labels == None)[0]
                    spectra = np.delete(spectra, ind, axis=0)
                    s_ind = np.delete(s_ind, ind)

                if drop_out_of_bounds:
                    con1 = spectra < 0
                    con2 = spectra > reflectance_scale_factor
                    ind = np.where(con1 | con2)[0]
                    ind = np.unique(ind)
                    spectra = np.delete(spectra, ind, axis=0)
                    s_ind = np.delete(s_ind, ind)

                # check if there are still spectra and bands left
                if (spectra.shape[0] > 0) and (spectra.shape[1] > 0):

                    # perform the spectral resampling
                    new_spectra = spectral_resampling(wavelength, fwhm, wl, fw, spectra,
                                                      resample_threshold=resample_threshold,
                                                      fill_missing=fill_missing,
                                                      raise_error_if_missing=raise_error_if_missing)

                    # if the spectral resampling worked, add everything to the corresponding collections
                    if new_spectra is not None:
                        new_spectra = new_spectra.astype(reflectance_dtype)
                        spec_coll.append(new_spectra)
                        s_ind_coll.append(s_ind)

        # merge the collections
        new_spectra = np.concatenate(spec_coll, axis=0)
        s_ind = np.concatenate(s_ind_coll)

        # get the corresponding s-index entries
        s_index = self.s_index.loc[s_ind]

        # make the new metadata dictionary
        new_metadata = dict()

        # the mandatory library and band attributes are derived from method input
        new_metadata[mdm['l_wavelength_unit']] = wavelength_unit
        new_metadata[mdm['l_reflectance_scale_factor']] = reflectance_scale_factor
        new_metadata[mdm['b_wavelength']] = wavelength
        new_metadata[mdm['b_fwhm']] = fwhm

        # all spectrum attributes are derived from the resulting s-index entries
        # the relabeled class labels are used instead of the original labels (the latter attribute is dropped)
        new_metadata[mdm['s_class_label']] = s_index['s_class_relabel'].values
        s_attr = s_index.columns.values
        s_attr = s_attr[s_attr != 's_class_label']
        s_attr = s_attr[s_attr != 's_class_relabel']

        for attr in s_attr:

            new_metadata[mdm[attr]] = s_index[attr]
            
        # finally store the merged state of the GSL
        self.merged_state = (new_spectra, new_metadata)

    def merged_state_to_envi(self, path,
                             specify_header_entries=None,
                             add_optional_attributes=True):

        # check input
        if not isinstance(path, str):
            raise TypeError('path must be a string')
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError('{} directory not found'.format(os.path.dirname(path)))
        if not isinstance(specify_header_entries, dict):
            raise TypeError('specify_header_entries must be a dictionary')

        spectra, md = self.merged_state
        mdm = self._metadata_model
        md_new = dict()

        # ENVI numeric data type is inferred from the spectra, giving priority to the most specific match
        dtype = spectra.dtype
        envi_dtype = dtype_to_envi[dtype]

        # add ENVI specific header entries with correct label
        md_new['description'] = None
        md_new['samples'] = spectra.shape[1]
        md_new['lines'] = spectra.shape[0]
        md_new['bands'] = 1
        md_new['header offset'] = 0
        md_new['file type'] = 'ENVI Spectral Library'
        md_new['data type'] = envi_dtype
        md_new['interleave'] = 'bsq'
        md_new['byte order'] = 0
        md_new['wavelength units'] = md[mdm['l_wavelength_unit']]
        md_new['reflectance scale factor'] = md[mdm['l_reflectance_scale_factor']]
        md_new['wavelength'] = md[mdm['b_wavelength']]
        md_new['fwhm'] = md[mdm['b_fwhm']]
        md_new['spectra names'] = md[mdm['s_class_label']]
        md_new['band names'] = [str(w) + ' ' + md_new['wavelength units'] for w in md_new['wavelength']]
        
        # users can optionally specify ENVI header entries (assuming they respect the conventions)
        # note that these entries will overwrite earlier specified entries
        for attr in specify_header_entries.keys():
            
            md_new[attr] = specify_header_entries[attr]

        # all remaining optional metadata entries of the merged GSL are added to the header as is
        # note that in this case priority is given to header entries that are already specified to avoid inconsistencies
        if add_optional_attributes:

            for attr in md.keys():

                if attr not in md_new.keys():
                    md_new[attr] = copy.deepcopy(md[attr])
            
        # save to ENVI spectral library
        save_envi_library(path, spectra, md_new)
