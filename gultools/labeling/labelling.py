import numpy as np
import pandas as pd
from gultools.spectral_distance import *


def insert_spectra_labels_in_metadata(ids, labels, metadata):

    dfright = pd.DataFrame(data=None)
    dfright['labels'] = labels
    dfright['ids'] = ids.astype(str)

    dfleft = pd.DataFrame(data=None)
    dfleft['ids'] = copy.deepcopy(metadata['spectra names'].astype(str))

    df = pd.merge(dfleft, dfright, on='ids', how='left')
    df.fillna('unlabeled', inplace=True)

    metadata['spectra names'] = df['labels'].values

    return metadata


def multi_criteria_analysis(spectrum, library, labels,
                            distance_measures=None,
                            distance_measures_labels=None,
                            weights=None,
                            normalize_similarities=True):

    spectrum = spectrum.reshape(1, -1)
    unique_labels = np.unique(labels)

    if (distance_measures is not None) and not distance_measures_labels:
        distance_measures_labels = np.array(['meas{}'.format(d) for d in range(len(distance_measures))])

    if not distance_measures:
        distance_measures = [l1,
                             l2,
                             correlation,
                             sam,
                             sid,
                             mahalanobis,
                             bhattacharyya,
                             jmd,
                             l1_sam,
                             l2_sam,
                             sid_sam,
                             jmd_sam,
                             spectral_similarity]

        distance_measures_labels = ['L1',
                                    'L2',
                                    'Corr.',
                                    'SAM',
                                    'SID',
                                    'Mahal.',
                                    'Bhatt.',
                                    'JMD',
                                    'L1-SAM',
                                    'L2-SAM',
                                    'SID-SAM',
                                    'JMD-SAM',
                                    'SpecSim']

    if weights is not None:
        weights = weights.astype(float) / weights.sum()
    else:
        weights = np.ones(len(distance_measures))
        weights /= weights.sum()
    weights = weights.reshape(1, -1)

    n_measures = len(distance_measures)
    rows = library.shape[0]
    dist = np.zeros((rows, n_measures))

    for d, dm in enumerate(distance_measures):

        dist[:, d] = dm(spectrum, library, norm=True)

    similarity = 1 - dist

    # get the maximum similarity for each label-similarity measure combination
    ls = np.zeros((unique_labels.size, n_measures))
    ls_indices = np.zeros((unique_labels.size, n_measures), dtype=int)

    for l, label in enumerate(unique_labels):

        ind = np.where(labels != label)[0]
        similarity_label = copy.deepcopy(similarity)
        similarity_label[ind, :] = 0
        ls[l, :] = similarity_label.max(axis=0)
        ls_indices[l, :] = np.argmax(similarity_label, axis=0)

    # get weighted mean similarities between labels
    ls_mean = np.sum(ls * weights, axis=1)

    # get the best fitting spectrum and corresponding similarity (measure) of the overall best fitting label class
    best_label_ind = np.argmax(ls_mean)
    best_measure_ind = np.argmax(ls[best_label_ind, :])
    best_library_ind = ls_indices[best_label_ind, best_measure_ind]
    
    best_similarity = ls[best_label_ind, best_measure_ind]
    best_similarity_measure = distance_measures_labels[best_measure_ind]
    best_spectrum = library[best_library_ind, :]

    # sort output in descending order
    sortind = np.argsort(ls_mean)
    sortind = sortind[::-1]
    ls = ls[sortind, :]
    ls_mean = ls_mean[sortind]
    unique_labels = unique_labels[sortind]

    if normalize_similarities:
        ls = ls / np.expand_dims(ls.sum(axis=0), axis=1)
        ls_mean /= ls_mean.sum()

    output = (unique_labels, ls, ls_mean, )