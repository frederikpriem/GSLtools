import os
import copy
import re
import numpy as np
import pandas as pd


protected_attributes = ['lid',  # the first 5 attributes must be included in each library metadata dictionary
                        'spectra',  # absence of one or more of these 5 attributes will raise an error
                        'bands',
                        'wavelength scale factor',
                        'wavelength',
                        'sid',  # the following 9 attributes can be included in each library metadata dictionary
                        'wavelength inverse unit',  # if left unspecified, they will be added with default values
                        'fwhm',
                        'reflectance scale factor',
                        'class label',
                        'bid',
                        'l-filter',
                        's-filter',
                        'b-filter']

allowed_dtypes = [bool,
                  str,
                  int,
                  float,
                  np.str,
                  np.int,
                  np.int8,
                  np.int16,
                  np.int32,
                  np.int64,
                  np.uint,
                  np.uint8,
                  np.uint16,
                  np.uint32,
                  np.uint64,
                  np.float,
                  np.float16,
                  np.float32,
                  np.float64]


def is_not_none(o,
                object_name='object'):

    if o is None:
        raise Exception('{} must not be None'.format(object_name))


def is_bool(o,
            object_name='object'):

    if o is not None:

        if not isinstance(o, bool):
            raise Exception('{} must be a Boolean'.format(object_name))


def is_str(o,
           length=None,
           max_length=None,
           min_length=None,
           regex=None,
           path=False,
           dir=False,
           object_name='object'):

    if o is not None:

        if not isinstance(o, str):
            raise Exception('{} must be a string'.format(object_name))

        if length:
            if len(o) != length:
                raise Exception('{} must have exactly {} character(s)'.format(object_name, length))

        if max_length:
            if len(o) > max_length:
                raise Exception('{} must not be longer than {} character(s)'.format(object_name, max_length))

        if min_length:
            if len(o) < min_length:
                raise Exception('{} must not be shorter than {} character(s)'.format(object_name, min_length))

        if regex:
            if not bool(re.match(regex, o)):
                raise Exception('{} must match the regular expression {}'.format(object_name, regex))

        if path:
            if not os.path.exists(o):
                raise Exception('{} must be an existing file path'.format(object_name))

        if dir:
            if not os.path.isdir(os.path.split(o)[0]):
                raise Exception('{} must be/contain an existing directory path'.format(object_name))


def check_regex(o, regex,
                object_name='object'):

    if o is not None:

        o_array = np.array(o, dtype=str)

        if o_array.size == 1:
            is_str(o_array,
                   regex=regex,
                   object_name=object_name)
        else:
            for s in o_array:
                is_str(s,
                       regex=regex,
                       object_name=object_name)


def is_int(o,
           l=None,
           le=None,
           g=None,
           ge=None,
           object_name='object'):

    if o is not None:

        if not isinstance(o, int):
            raise Exception('{} must be an integer'.format(object_name))

        if l:
            if not o < l:
                raise Exception('{} must be less than {}'.format(object_name, l))

        if le:
            if not o <= le:
                raise Exception('{} must be less than or equal to {}'.format(object_name, le))

        if g:
            if not o > g:
                raise Exception('{} must be greater than {}'.format(object_name, g))

        if ge:
            if not o >= ge:
                raise Exception('{} must be greater than or equal to {}'.format(object_name, ge))


def is_float(o,
             l=None,
             le=None,
             g=None,
             ge=None,
             object_name='object'):

    if o is not None:

        if not isinstance(o, float):
            raise Exception('{} must be a floating-point number'.format(object_name))

        if l:
            if not o < l:
                raise Exception('{} must be less than {}'.format(object_name, l))

        if le:
            if not o <= le:
                raise Exception('{} must be less than or equal to {}'.format(object_name, le))

        if g:
            if not o > g:
                raise Exception('{} must be greater than {}'.format(object_name, g))

        if ge:
            if not o >= ge:
                raise Exception('{} must be greater than or equal to {}'.format(object_name, ge))


def is_int_or_str(o,
                  object_name='object'):

    if o is not None:

        if not (isinstance(o, int) or isinstance(o, str)):
            raise Exception('{} must be an integer or string'.format(object_name))


def is_tuple(o,
             length=None,
             object_name='object'):

    if o is not None:

        if not isinstance(o, tuple):
            raise Exception('{} must be a tuple'.format(object_name))

        if length:
            if not len(o) == length:
                raise Exception('{} must have length {}'.format(object_name, length))


def is_list(o,
            length=None,
            object_name='object'):

    if o is not None:

        if not isinstance(o, list):
            raise Exception('{} must be a list'.format(object_name))

        if length:
            if not len(o) == length:
                raise Exception('{} must have length {}'.format(object_name, length))


def is_list_or_tuple(o,
                     length=None,
                     object_name='object'):

    if o is not None:

        if not (isinstance(o, tuple) or isinstance(o, list)):
            raise Exception('{} must be a list or tuple'.format(object_name))

        if length:
            if not len(o) == length:
                raise Exception('{} must have length {}'.format(object_name, length))


def is_array_like(o,
                  size=None,
                  shape=None,
                  dimensions=None,
                  dtype=None,
                  object_name='object',
                  repetition_allowed=True):

    if o is not None:

        if not (isinstance(o, np.ndarray) or isinstance(o, list) or isinstance(o, tuple)):
            raise Exception('{} must be a list, tuple or numpy array'.format(object_name))

        try:
            o = np.array(o, dtype=dtype)
        except Exception:
            if dtype:
                raise Exception('{} must be convertible to an array with datatype {}'.format(object_name, dtype))
            else:
                raise Exception('{} must be convertible to an array'.format(object_name))

        if size:
            if o.size != size:
                raise Exception('{} must have size {}'.format(object_name, size))

        if shape:
            if o.shape != shape:
                raise Exception('{} must have shape {}'.format(object_name, shape))

        if dimensions:
            if len(o.shape) != dimensions:
                raise Exception('{} must be a {}D array'.format(object_name, dimensions))

        if not repetition_allowed:
            uni, cnt = np.unique(o, return_counts=True)
            if np.any(cnt > 1):
                raise Exception('{} must not have duplicate values'.format(object_name))


def is_dataframe(o,
                 columns=None,
                 object_name='object'):

    if o is not None:

        if not isinstance(o, pd.DataFrame):
            raise Exception('{} must be a Pandas DataFrame object'.format(object_name))

        if columns is not None:

            for column in columns:

                if column not in o.columns.values:
                    raise Exception('{} must have a column with name {}'.format(object_name, column))


def is_dict(o,
            keys=None,
            values=None,
            object_name='object'):

    if o is not None:

        if not isinstance(o, dict):
            raise Exception('{} must be a dictionary'.format(object_name))

        if keys is not None:

            for key in keys:

                if key not in o.keys():
                    raise Exception('{} dictionary must have key {}'.format(object_name, key))

        if values is not None:

            for value in values:

                if value not in o.values():
                    raise Exception('{} dictionary must have value {}'.format(object_name, value))


def is_one_of(o, values,
              object_name='object'):

    if o is not None:

        if not np.all(np.isin(np.array(o), values)):
            raise Exception('{} must be one of the following: {}'.format(object_name, values))


def is_not_one_of(o, values,
                  multivalued=False,
                  encapsulators='[]',
                  separator=',',
                  object_name='object'):

    if o is not None:

        if multivalued:

            new_o = []

            if isinstance(o, str):
                if o[0] == encapsulators[0] and o[-1] == encapsulators[1]:
                    new_o += o[1:-1].split(separator)
                else:
                    new_o.append(o)

            else:
                for elem in o:
                    if elem[0] == encapsulators[0] and elem[-1] == encapsulators[1]:
                        new_o += elem[1:-1].split(separator)
                    else:
                        new_o.append(elem)

            o = new_o

        if np.any(np.isin(np.array(o), values)):
            raise Exception('{} must not be one of the following: {}'.format(object_name, values))


def is_of_dtype(o, dtype,
                object_name='object'):

    if o is not None:

        try:
            np.array(o, dtype=dtype)
        except Exception:
            raise Exception('{} must be of type {}'.format(object_name, dtype))


def is_single_value(o,
                    object_name='object'):

    if o is not None:

        o_array = np.array(o)
        if len(o_array.shape) > 0:
            raise Exception('{} must be a scalar or a string'.format(object_name))


def check_hierarchy(hierarchy, dtype,
                    object_name='object'):

    if hierarchy is not None:

        is_array_like(hierarchy,
                      dimensions=2,
                      dtype=dtype)

        hierarchy = np.array(hierarchy, dtype=dtype)

        # check if hierarchy rows and columns are unique
        uni, h_cnt = np.unique(hierarchy,
                               return_counts=True,
                               axis=0)
        uni, v_cnt = np.unique(hierarchy,
                               return_counts=True,
                               axis=1)
        if np.any(h_cnt > 1) or np.any(v_cnt > 1):
            raise Exception(f'{object_name} rows and columns must be unique')

        # check if hierarchy is nested
        rows, cols = hierarchy.shape

        for col in range(1, cols):

            col_values = hierarchy[:, col]
            col_values_unique = np.unique(col_values)

            for cvu in col_values_unique:

                r = np.where(col_values == cvu)[0]
                prev_col_values = hierarchy[r, col - 1]
                uni_prev_col_values = np.unique(prev_col_values)
                if len(uni_prev_col_values) > 1:
                    raise Exception(f'{object_name} must be organized in a nested fashion, '
                                     'going from general (left) to specific (right)')

        # check if specificity is non-descending
        for row in range(rows):

            prev_class = []

            for col in range(1, cols):

                if hierarchy[row, col] != hierarchy[row, col - 1]:
                    prev_class.append(hierarchy[row, col - 1])
                if hierarchy[row, col] in prev_class:
                    raise Exception(f'each column of {object_name}, besides the first, '
                                     'must be more specific than the column to its left')

        # check if specifications are exhaustive
        for col in range(cols - 1):

            col_values = hierarchy[:, col]
            col_values_unique = np.unique(col_values)

            for cvu in col_values_unique:

                rows = np.where(col_values == cvu)[0]
                next_col_values = hierarchy[rows, col + 1]
                if (cvu in next_col_values) and (len(np.unique(next_col_values)) > 1):
                    raise Exception(f'if a value is split in {object_name}, it '
                                     'must not be repeated in the next column')


def check_metadata(metadata, metadata_model, error_prefix):

    # check if the metadata object is a dictionary and that it has all the mandatory metadata attribute keys (aid)
    mandatory_attributes = metadata_model.get_mandatory_attribute_ids()
    object_name = error_prefix + 'metadata'
    is_dict(metadata,
            keys=mandatory_attributes,
            object_name=object_name)

    # entries that are not included in the metadata model are removed from the library metadata dictionary
    all_aid_list = metadata_model.get_attribute_ids()
    for key in list(metadata.keys()):

        if key not in all_aid_list:
            del metadata[key]

    # check if each metadata entry conforms to the properties of the corresponding metadata model attribute
    aid_list = list(metadata.keys())

    # check spectra, bands and wavelength
    aid_list.pop(aid_list.index('spectra'))
    spectra = metadata['spectra']
    is_int(spectra,
           ge=1,
           object_name='spectra')

    aid_list.pop(aid_list.index('bands'))
    bands = metadata['bands']
    is_int(bands,
           ge=1,
           object_name='bands')

    aid_list.pop(aid_list.index('wavelength'))
    wavelength = metadata['wavelength']
    is_array_like(wavelength,
                  dimensions=1,
                  size=bands,
                  dtype=float,
                  repetition_allowed=False,
                  object_name='wavelength')

    # add optional metadata attributes with default values, if unspecified
    if 'sid' not in aid_list:
        metadata['sid'] = [str(s) for s in range(spectra)]
    if 'bid' not in aid_list:
        metadata['bid'] = [str(b) for b in range(bands)]
    if 'wavelength inverse unit' not in aid_list:
        metadata['wavelength inverse unit'] = False
    if 'fwhm' not in aid_list:
        wavelength = np.array(wavelength, dtype=float)
        diff = np.diff(wavelength)
        diff1 = np.append(diff[0], diff)
        diff2 = np.append(diff, diff[-1])
        fwhm = np.minimum(diff1, diff2)
        metadata['fwhm'] = fwhm
    if 'reflectance scale factor' not in aid_list:
        metadata['reflectance scale factor'] = 1.
    if 'class label' not in aid_list:
        metadata['class label'] = [np.nan for s in range(spectra)]
    if 'l-filter' not in aid_list:
        metadata['l-filter'] = True
    if 's-filter' not in aid_list:
        metadata['s-filter'] = [True for s in range(spectra)]
    if 'b-filter' not in aid_list:
        metadata['b-filter'] = [True for b in range(bands)]

    # check the remaining attributes
    for aid in aid_list:

        if aid in list(metadata.keys()):

            entry = metadata[aid]
            object_name = '{}{}'.format(error_prefix, aid)
            attribute = metadata_model.get_attribute(aid)

            # check the data type and dimensions
            if attribute._atype == 's':
                is_array_like(entry,
                              dimensions=1,
                              size=spectra,
                              dtype=attribute._dtype,
                              repetition_allowed=attribute._repetition_allowed,
                              object_name=object_name)

            elif attribute._atype == 'b':
                is_array_like(entry,
                              dimensions=1,
                              size=bands,
                              dtype=attribute._dtype,
                              repetition_allowed=attribute._repetition_allowed,
                              object_name=object_name)

            else:
                is_of_dtype(entry, attribute._dtype,
                            object_name=object_name)

                is_single_value(entry,
                                object_name=object_name)

            # check the remaining constraints on the attribute value domain
            if attribute._l:
                if not np.all(np.array(entry, dtype=attribute._dtype) < attribute._l):
                    raise Exception('{} must be less than {}'.format(object_name, attribute._l))

            if attribute._le:
                if not np.all(np.array(entry, dtype=attribute._dtype) <= attribute._le):
                    raise Exception('{} must be less than or equal to {}'.format(object_name, attribute._le))

            if attribute._g:
                if not np.all(np.array(entry, dtype=attribute._dtype) > attribute._g):
                    raise Exception('{} must be greater than {}'.format(object_name, attribute._g))

            if attribute._ge:
                if not np.all(np.array(entry, dtype=attribute._dtype) >= attribute._ge):
                    raise Exception('{} must be greater than or equal to {}'.format(object_name, attribute._ge))

            # account for possibly multivalued string attributes
            if attribute._multivalued:

                parsed = []

                for e in entry:

                    if len(e) >= 2:
                        if e[0] == attribute._encapsulators[0] and e[-1] == attribute._encapsulators[1]:
                            parsed += [part.strip() for part in e[1:-1].split(attribute._separator)]
                    else:
                        parsed.append(e)

                entry = copy.deepcopy(parsed)

            if attribute._value_domain is not None:
                is_one_of(entry, attribute._value_domain,
                          object_name=object_name)

            if attribute._value_hierarchy is not None:
                is_one_of(entry, np.unique(attribute._value_hierarchy),
                          object_name=object_name)

            if attribute._regex:
                check_regex(entry, attribute._regex,
                            object_name=object_name)

    return metadata


def check_spectra(spectra, metadata, error_prefix):

    n_spectra = metadata['spectra']
    n_bands = metadata['bands']
    object_name = error_prefix + 'spectra'

    is_array_like(spectra,
                  shape=(n_spectra, n_bands),
                  dtype=float,
                  object_name=object_name)
    spectra = copy.deepcopy(spectra)
    spectra /= float(metadata['reflectance scale factor'])
    spectra[np.isnan(spectra)] = -1
    spectra[spectra > 1] = -1
    spectra[spectra < 0] = -1

    if np.all(spectra == -1, axis=None):
        raise Exception('{} contains only nodata values'.format(error_prefix))

    for b in range(n_bands):

        if np.all(spectra[:, b] == -1):
            bid = metadata['bid'][b]
            print('Warning: {} band with bid {} contains only nodata values.'.format(error_prefix, bid))

    for s in range(n_spectra):

        if np.all(spectra[s, :] == -1):
            sid = metadata['sid'][s]
            raise Exception('{} spectrum with sid {} contains only nodata values.'.format(error_prefix, sid))


def compare_metadata_models(mdm1, mdm2):

    attribute_ids1 = mdm1.get_attribute_ids()
    attribute_ids1.sort()
    attribute_ids2 = mdm2.get_attribute_ids()
    attribute_ids2.sort()
    out_code = 0

    if not attribute_ids1 == attribute_ids2:
        out_code = 1
    else:
        mda_attributes = ['_aid', '_atype', '_dtype', '_description', '_mandatory', '_value_domain', '_value_hierarchy',
                         '_regex', '_l', '_le', '_g', '_ge', '_nodata_value', '_multivalued', '_encapsulators',
                         '_separator', '_repetition_allowed', '_protected']

        for aid in attribute_ids1:

            attribute1 = mdm1.get_attribute(aid)
            attribute2 = mdm2.get_attribute(aid)

            for a in mda_attributes:

                if isinstance(attribute1.__getattribute__(a), np.ndarray):
                    if not np.array_equal(attribute1.__getattribute__(a), attribute2.__getattribute__(a)):
                        out_code = 1
                        break
                else:
                    if not attribute1.__getattribute__(a) == attribute2.__getattribute__(a):
                        out_code = 1
                        break

    return out_code
