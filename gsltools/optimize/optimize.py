import copy
import numpy as np
from tqdm import tqdm
from gsltools.distance import *
from gsltools.validate import kappa_coefficient
from gsltools.denoise import hysime
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


"""
This module addresses library optimize.
"""


def comprehensive(spectra,
                  distance_measure=auc_sam,
                  distance_threshold=0.1,
                  return_indices=False):

    """
    Optimizes a library with the same approach used in the second step of 'endmember extract' (see documentation).
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param distance_measure: object, distance measure used
    :param distance_threshold: float, distance threshold used (see 'endmember extract')
    :param return_indices: bool, False by default, whether to return the indices of retained spectra
    :return: spectra: 2D-array of floats with shape (r retained spectra, b bands)
    """

    center = spectra.mean(axis=0)
    dist_center = distance_measure(spectra, center)
    loc = np.arange(dist_center.size)
    retained_spectra = []
    ret_loc = []

    while True:

        # stop when all library spectra are either retained or removed
        if spectra.shape[0] == 0:
            break

        # retain the spectrum located furthest away from the center
        ind = np.argmax(dist_center)
        ret = copy.deepcopy(spectra[ind, :])
        retained_spectra.append(ret.reshape(1, -1))
        ret_loc.append(loc[ind])

        # remove the retained spectrum
        spectra = np.delete(spectra, ind, axis=0)
        loc = np.delete(loc, ind)
        dist_center = np.delete(dist_center, ind)

        # stop when all library spectra are either retained or removed
        if spectra.shape[0] == 0:
            break

        # Remove similar spectra from the library to avoid redundancy
        dist = distance_measure(ret, spectra)
        del_ind = np.where(dist < distance_threshold)[0]

        if del_ind.size > 0:

            spectra = np.delete(spectra, del_ind, 0)
            loc = np.delete(loc, del_ind)
            dist_center = np.delete(dist_center, del_ind)

    spectra = np.concatenate(retained_spectra, axis=0)

    output = spectra

    if return_indices:
        ret_loc = np.array(ret_loc)
        output = (spectra, ret_loc)

    return output


def pairwise(spectra,
             distance_measure=auc_sam,
             distance_threshold=0.1,
             return_indices=False):

    """
    This is a more generic implementation of EAR/MASA optimize (see corresponding papers). Any distance measure
    can be used.
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param distance_measure: object, distance measure used
    :param distance_threshold: float, distance threshold used
    :param return_indices: bool, False by default, whether to return the indices of retained spectra
    :return: spectra: 2D-array of floats with shape (r retained spectra, b bands)
    """
    
    # produce a square array containing distances between each pair of spectra
    dist = pdist(spectra, distance_measure)
    dist = squareform(dist)

    # set the diagonal element to the highest finite positive value
    dist[np.arange(dist.shape[0]), np.arange(dist.shape[0])] = np.finfo(float).max

    stop = False
    indices = np.arange(spectra.shape[0])

    while not stop:

        # in case of excessive spectral similarity, remove the most redundant spectrum,
        # i.e. having the lowest row/column average distance
        if np.any(dist < distance_threshold):

            ind = np.unravel_index(np.argmin(dist), dist.shape)
            spec1 = ind[0]
            spec2 = ind[1]
            avg1 = dist[spec1, :].mean()
            avg2 = dist[spec2, :].mean()

            if avg1 > avg2:

                indices = np.delete(indices, spec2)
                dist = np.delete(dist, spec2, axis=0)
                dist = np.delete(dist, spec2, axis=1)

            elif avg1 < avg2:

                indices = np.delete(indices, spec1)
                dist = np.delete(dist, spec1, axis=0)
                dist = np.delete(dist, spec1, axis=1)

            else:

                del_ind = np.array([spec1, spec2])
                indices = np.delete(indices, del_ind)
                dist = np.delete(dist, del_ind, axis=0)
                dist = np.delete(dist, del_ind, axis=1)
        else:

            stop = True

    library_optimized = spectra[indices, :]

    if not return_indices:
        return library_optimized
    else:
        return library_optimized, indices


def ies(spectra, labels, classifier,
        random_start=False,
        start_size=None,
        return_indices=False,
        return_kappa=False):

    """
    This is a more generic implementation of Iterative Endmember Selection (IES) optimize (see corresponding paper).
    Any classifier object with a 'fit' and 'predict' method (cfr. Scikit-Learn) can be used.
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param labels: 1D-array with shape (n spectra,)
    :param classifier: object, classifier used, must have Scikit-Learn style fit and predict methods
    :param random_start: bool, whether to start with a randomly sampled subset of the whole set of spectra
    :param start_size: int, size of the starting population, will be sampled randomly
    :param return_indices: bool, False by default, whether to return the indices of retained spectra
    :param return_kappa: bool, False by default, whether to return the kappa index corresponding to the final subset
    :return: spectra: 2D-array of floats with shape (r retained spectra, b bands)
    """

    kappas = []

    if random_start:

        retained = np.random.choice(range(labels.size),
                                    size=start_size,
                                    replace=False)
        spectra_temp = spectra[retained, :]
        labels_temp = labels[retained]
        model = classifier.fit(spectra_temp, labels_temp)
        predict = model.predict(spectra)
        kappa = kappa_coefficient(predict, labels)[0]
        kappa_prev = copy.deepcopy(kappa)

    else:

        # find the inter-class pair of spectra that best predicts all label
        combs = list(combinations(np.arange(labels.size), 2))
        combs = np.array(combs)
        combs_labels = list(combinations(labels, 2))
        combs_labels = np.array(combs_labels)
        con = np.equal(combs_labels[:, 0], combs_labels[:, 1])
        combs = combs[np.where(con)[0]]

        for comb in tqdm(combs,
                         desc='finding the best initial pair of spectra',
                         leave=False):

            spectra_temp = spectra[np.array(comb), :]
            labels_temp = np.array([labels[comb[0]], labels[comb[1]]])
            model = classifier.fit(spectra_temp, labels_temp)
            predict = model.predict(spectra)
            kappa = kappa_coefficient(predict, labels)[0]
            kappas.append(kappa)

        kappas = np.array(kappas)
        ind_max = np.argmax(kappas)
        kappa_prev = copy.deepcopy(kappas[ind_max])
        retained = np.array(combs[ind_max])

    removed = np.arange(labels.size, dtype=int)
    removed = np.delete(removed, retained)

    # start the main iteration
    while True:

        addition = False
        removal = False

        # add the spectrum whose addition yields the highest improvement to the model, if any
        if removed.size > 0:

            kappas = []

            for rem in tqdm(removed,
                            desc='adding spectra',
                            leave=True):

                ind_temp = np.concatenate((retained, [rem]))
                spectra_temp = spectra[ind_temp, :]
                labels_temp = labels[ind_temp]

                try:

                    model = classifier.fit(spectra_temp, labels_temp)
                    predict = model.predict(spectra)
                    kappa = kappa_coefficient(predict, labels)[0]
                    kappas.append(kappa)

                except:

                    kappas.append(-1)

            kappas = np.array(kappas)
            ind_max = np.argmax(kappas)
            kappa_max = kappas[ind_max]

            if kappa_max > kappa_prev:

                retained = np.concatenate((retained, [removed[ind_max]]))
                removed = np.delete(removed, ind_max)
                kappa_prev = copy.deepcopy(kappa_max)
                addition = True

        # remove the spectrum whose removal yields the highest improvement to the model, if any
        if retained.size > 0:

            kappas = []

            for r, ret in tqdm(list(enumerate(retained)),
                               desc='removing spectra',
                               leave=True):

                ind_temp = np.delete(retained, r)
                spectra_temp = spectra[ind_temp, :]
                labels_temp = labels[ind_temp]

                try:

                    model = classifier.fit(spectra_temp, labels_temp)
                    predict = model.predict(spectra)
                    kappa = kappa_coefficient(predict, labels)[0]
                    kappas.append(kappa)

                except:

                    kappas.append(-1)

            kappas = np.array(kappas)
            ind_max = np.argmax(kappas)
            kappa_max = kappas[ind_max]

            if kappa_max > kappa_prev:

                removed = np.concatenate((removed, [retained[ind_max]]))
                retained = np.delete(retained, ind_max)
                kappa_prev = copy.deepcopy(kappa_max)
                removal = True

        # break if no spectra were added/removed during this iteration
        if not (addition or removal):
            break

    output = spectra[retained, :]
    if return_indices:
        output = spectra[retained, :], retained
        if return_kappa:
            output = spectra[retained, :], retained, kappa_prev
    elif return_kappa:
        output = spectra[retained, :], kappa_prev

    return output


def music(image, spectra,
          use_hysime=True):

    """
    This function essentially calculates the Euclidean distance between library spectra and the plane formed by the
    first n eigenvectors of the covariance matrix of an image. The technique is described in detail in 'MUSIC-CSR:
    Hyperspectral Unmixing via Multiple Signal Classification and Collaborative Sparse Regression' by Marian-Daniel
    Iordache et al. (2014).

    Note that this is a simplified implementation of MUSIC that leaves out the HYSIME-based determination of the optimal
    image subspace, i.e. the number of eigenvectors to use. The user can optionally specify how many of the first
    eigenvector must be used to compute MUSIC.

    :param spectra: 2D array of shape (n_spectra, bands)
    :param image: 3D array of shape (rows, columns, bands)
    :param use_hysime: Boolean, use HYSIME to determine the optimal subspace
    :return: 1D array containing MUSIC distances of each spectrum relative to the image subspace
    """

    if len(image.shape) > 3 or len(image.shape) < 2:
        err_msg = """image must either be a 2D array of shape (spectra, bands)
        or a 3D array of shape (rows, columns, bands)"""
        raise ValueError(err_msg)

    if len(image.shape) == 3:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)

    if spectra.shape[1] != image.shape[1]:
        raise ValueError('number of bands in library and image must be equal')

    # brightness normalize the image and spectra
    image = copy.deepcopy(image)
    spectra = copy.deepcopy(spectra)
    image /= image.sum(axis=1).reshape(-1, 1)
    spectra /= spectra.sum(axis=1).reshape(-1, 1)

    # determine the optimal subspace using HYSIME
    if use_hysime:
        n_components, _ = hysime(image)
    else:
        n_components = image.shape[1]

    # get eigenvalues and eigenvectors of the image-derived covariance matrix, i.e. covariance between image bands
    cov = np.cov(image, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov)
    ind = np.argsort(eigval)[::-1]
    eigvect = eigvect[:, ind]
    eigvect = eigvect[:, :n_components]

    # calculate Euclidean distances between the tested spectra and the hyperplane formed by the image eigenvectors
    p = np.diag(np.ones(image.shape[1])) - np.dot(eigvect, eigvect.T)
    dist = np.sum((np.dot(p, spectra.T) ** 2), axis=0) ** 0.5
    dist /= np.sum(spectra ** 2, axis=1).squeeze() ** 0.5

    return dist


def amuses(image, spectra, distance_measure, fmin, fmax, dmin, dmax):

    if len(image.shape) > 3 or len(image.shape) < 2:
        err_msg = """image must either be a 2D array of shape (spectra, bands)
        or a 3D array of shape (rows, columns, bands)"""
        raise ValueError(err_msg)

    if len(image.shape) == 3:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)

    if spectra.shape[1] != image.shape[1]:
        raise ValueError('number of bands in library and image must be equal')

    dmusic = music(image, spectra)

    retain = np.where(dmusic < np.quantile(dmusic, fmin))[0]
    con1 = dmusic >= np.quantile(dmusic, fmin)
    con2 = dmusic < np.quantile(dmusic, fmax)
    maybe = np.where(con1 & con2)[0]

    retained_spectra = spectra[retain, :]

    for m in maybe:

        maybe_spectrum = spectra[m, :]
        dist = distance_measure(maybe_spectrum, retained_spectra)
        d = (dmusic[m] - dmusic.min()) / (dmusic.max() - dmusic.min())
        thres = dmin + (dmax - dmin) * d

        if np.all(dist > thres):
            retain = np.append((retain, [m]))
            retained_spectra = spectra[retain, :]

    return retain


def dice(image, spectra,
         norm='l1',
         n_components=None):

    """
    This is an experimental image-based library optimize technique that considers spectra and their Deviation
    from the Image Center in Eigenspace (DICE). We first fit a PC transformation on the tested image and then apply it
    on the tested spectra. The DICE distance measure is then quantified by taking component-wise ratios between absolute
    PC values of the tested spectra and the standard deviation of the corresponding component. Aggregate statistics are
    obtained for each tested spectrum using L1, L2 or Linf norms.

    DICE can be used to determine if a spectrum is located in a more central or peripheral position in an image point
    cloud. DICE is a relative distance measure (considering the varying distributions of the image point cloud along its
    dimensions) using a well-defined point in the image feature space (the point cloud center) as reference. Similar to
    MUSIC, the user can optionally specify how many of the first eigenvectors must be used to compute DICE.

    :param image: Either 3D-array of floats with size (rows, cols, bands) or 2D-array of floats with shape (rows * cols,
    bands)
    :param spectra: 2D array of shape (n spectra, b bands)
    :param mode: string, denoting the distance measure used to define DIVE, either 'l1', 'l2' or 'max'
    :param n_components: int, None by default, number of PC to use
    :return: d: 1D-array of floats with shape (n spectra,), containing DIVE values for each tested spectrum.
    """

    spectra = copy.deepcopy(spectra)
    image = copy.deepcopy(image)

    if norm not in ['l1', 'l2', 'linf']:
        raise ValueError("mode must be 'l1', 'l2' or 'linf'")

    if len(image.shape) > 3 or len(image.shape) < 2:
        err_msg = """image must either be a 2D array of shape (spectra, bands)
        or a 3D array of shape (rows, columns, bands)"""
        raise ValueError(err_msg)

    if len(image.shape) == 3:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)

    if not n_components:
        n_components = image.shape[1]

    pca = PCA(n_components=n_components)
    pca.fit(image)
    image_pc = pca.transform(image)
    image_pc_mean = image_pc.mean(axis=0)
    image_pc_std = image_pc.std(axis=0)

    spectra_pc = pca.transform(spectra)
    d = (spectra_pc - image_pc_mean.reshape(1, -1)) / image_pc_std.reshape(1, -1)

    if norm == 'l1':
        d = np.mean(np.abs(d), axis=1)
    elif norm == 'l2':
        d = np.mean(d**2, axis=1)**0.5
    elif norm == 'linf':
        d = np.max(np.abs(d), axis=1)

    return d


def genetic_algorithm(spectra, labels, estimator,
                      pop_size=100,
                      n_parents=40,
                      mutation_rate=0.01,
                      n_generations=20,
                      return_indices=False,
                      return_fitness_evolution=False,
                      print_fitness=False,
                      mutation_rate_control=0.5):

    """
    Applies Iterative Endmember Selection (IES) using a Genetic Algorithm (GA) instead of forward/backward selection.
    :param spectra: 2D array of shape (n spectra, b bands)
    :param labels: 1D-array with shape (n spectra,)
    :param estimator: object, classifier used, must have fit and predict methods
    :param pop_size: int, population size used in GA
    :param n_parents: int, number of parents used, must be smaller than pop_size
    :param mutation_rate: float [0, 1], probability of genes mutating during crossover
    :param n_generations: int, number of iterations over which the GA is run
    :param return_indices: bool, whether to return indices of retained spectra
    :param return_fitness_evolution: bool, whether to return the 1D-array containing the subsequent best fitness values
    of each generation
    :param print_fitness: bool, whether to print the best fitness of each generation
    :param mutation_rate_control: float [0, 1], factor by which mutation rate is multiplied/divided if the best fitness
    stagnates/increases over subsequent generations.
    :return:
    """

    def make_new_pop(pop_size):

        pop = np.random.uniform(0, 1, (pop_size, n_genes))
        pop = np.where(pop > 0.5, 1, 0)

        return pop

    def mutate(chromosome):

        n_genes = chromosome.size
        mutation = np.random.uniform(0, 1, n_genes)
        ind = np.where(mutation < mutation_rate)
        chromosome[ind] -= 1
        chromosome = np.abs(chromosome)

        return chromosome

    def assess(chromosome):

        ind = np.where(chromosome)[0]
        spectra_ = spectra[ind, :]
        labels_ = labels[ind]
        try:
            mod = estimator.fit(spectra_, labels_)
            est = mod.predict(spectra)
            fit = kappa_coefficient(est, labels)[0]
        except ValueError:
            fit = -1

        return fit

    def crossover(parents, fitness):

        n_offspring = pop_size - n_parents
        offspring = np.empty((n_offspring, n_genes))

        for o in range(offspring.shape[0]):

            # index of first parent
            parent1_ind = o % n_parents

            # index of second parent
            parent2_ind = (o + 1) % n_parents

            # sample genes based on parent fitness
            weights = fitness[[parent1_ind, parent2_ind]]
            weights = np.array(weights, dtype=float) / sum(weights)

            # crossover
            co = np.random.choice([parent1_ind, parent2_ind],
                                  replace=True,
                                  size=n_genes,
                                  p=weights)
            offspring[o, :] = parents[co, range(n_genes)]

        return offspring

    def mutate_offspring(offspring):

        for o in range(offspring.shape[0]):

            offspring[o, :] = mutate(offspring[o, :].squeeze())

        return offspring

    # main code block
    n_genes = spectra.shape[0]
    pop = make_new_pop(pop_size)
    fitness = np.ones(pop_size) * -1
    generations = list(range(n_generations))
    maxfit = np.empty(n_generations)
    meanfit = np.empty(n_generations)

    for generation in tqdm(generations, leave=False):

        for p in range(pop_size):

            if fitness[p] < 0:
                fitness[p] = assess(pop[p, :])

        meanfit[generation] = fitness.mean()
        maxfit[generation] = fitness.max()

        sortind = np.argsort(fitness)
        sortind = sortind[::-1]
        pop = pop[sortind, :]
        fitness = fitness[sortind]
        parents = pop[:n_parents, :]
        fitness = fitness[:n_parents]

        if print_fitness:
            print('generation {} max. fitness = {}'.format(generation, fitness[0]))

        offspring = crossover(parents, fitness)
        if mutation_rate:
            offspring = mutate_offspring(offspring)
        pop = np.concatenate((parents, offspring), axis=0)
        fitness = np.concatenate((fitness, np.ones(offspring.shape[0]) * -1))

        if generation > 0 and mutation_rate_control:
            if maxfit[generation] == maxfit[generation - 1]:
                mutation_rate *= mutation_rate_control
            elif maxfit[generation] > maxfit[generation - 1]:
                mutation_rate /= mutation_rate_control

    # final fitness assessment
    for p in range(pop_size):

        if fitness[p] < 0:
            fitness[p] = assess(pop[p, :])

    sortind = np.argsort(fitness)
    sortind = sortind[::-1]
    pop = pop[sortind, :]

    best = pop[0, :]
    ind = np.where(best)[0]
    spectra = spectra[ind, :]

    output = spectra
    if return_indices:
        output = (spectra, ind)
        if return_fitness_evolution:
            output = (spectra, ind, (meanfit, maxfit))
    elif return_fitness_evolution:
        output = (spectra, (meanfit, maxfit))

    return output