import numpy as np
import pandas as pd
import copy
from gsltools.distance import *

"""
This module provides function that support class labelling of spectra
"""


def insert_spectra_labels_in_metadata(ids, labels, metadata):

    """
    Inserts spectra labels in the metadata dictionary of a spectral library. Uses 'ids' to match labels to library
    spectra (assumes that 'ids' is presents as a metadata entry in the metadata dictionary)
    :param ids: 1D-array, ids used to match labels to their corresponding library spectra
    :param labels: 1D-array of strings, labels corresponding to each id
    :param metadata: dictionary, metadata dictionary of the spectral libary in which labels are being inserted
    :return: metadata: updated metadata dictionary
    """

    if len(ids) != len(labels):
        raise ValueError('ids and labels must be of equal length')

    dfright = pd.DataFrame(data=None)
    dfright['labels'] = labels
    dfright['ids'] = ids.astype(str)

    dfleft = pd.DataFrame(data=None)
    dfleft['ids'] = copy.deepcopy(metadata['spectra names'].astype(str))

    df = pd.merge(dfleft, dfright, on='ids', how='left')
    df.fillna('unlabeled', inplace=True)

    metadata['spectra names'] = df['labels'].values

    return metadata


def relabel(metadata, labels_old, labels_new):

    """
    Relabels spectra names in a metadata dictionary based on equal-sized sets of old and new labels.
    :param metadata: dictionary, metadata dictionary of spectral library
    :param labels_old: 1D-array of strings with shape (l labels,), the old labels to be replaced
    :param labels_new: 1D-array of strings with shape (l labels,), the new labels
    :return: metadata: dictionary, updated metadata dictionary
    """

    # Got a weird bug here where only the first two characters of labels were written to output.
    # It has something to do with numpy automatically converting strings containing only integers
    # to dtype('Ux'), where x is the highest number of integer digits in the data. Not sure why ...
    labels = metadata['spectra names'].astype(np.dtype('U1000'))
    labels_output = copy.deepcopy(labels)

    for label_old, label_new in zip(labels_old, labels_new):

        labels_output[labels == label_old] = label_new

    metadata['spectra names'] = labels_output

    return metadata