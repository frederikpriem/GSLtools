import numpy as np
from tqdm import tqdm
from gultools.spectral_distance import *
from gultools.validation import kappa_coefficient
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


"""
This module addresses library optimization.
"""


def comprehensive(spectra,
                  distance_measure=l1_sam,
                  distance_threshold=0.05,
                  return_indices=False):

    """
    Optimizes a library with the same approach used in the second step of 'endmember extraction' (see documentation).
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param distance_measure: object, distance measure used
    :param distance_threshold: float, distance threshold used (see 'endmember extraction')
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
    This is a more generic implementation of EAR/MASA optimization (see corresponding papers). Any distance measure
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
    This is a more generic implementation of Iterative Endmember Selection (IES) optimization (see corresponding paper).
    Any classifier object with a 'fit' and 'predict' method (cfr. Scikit-Learn) can be used.
    :param spectra: 2D-array of floats with shape (n spectra, b bands)
    :param labels: 1D-array with shape (n spectra,)
    :param classifier: object, classifier used, must have fit and predict methods
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
        combs = np.array(combinations(np.arange(labels.size), 2))
        combs_labels = np.array(combinations(labels, 2))
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


def music(spectra, image,
          n_eigenvector=None,
          thres_perc_variance=None):

    """
    This function essentially calculates the Euclidean distance between library spectra and the subspace formed by the
    first n eigenvectors of the covariance matrix of an image. The technique is described in detail in 'MUSIC-CSR:
    Hyperspectral Unmixing via Multiple Signal Classification and Collaborative Sparse Regression' by Marian-Daniel
    Iordache et al. (2014).

    Note that this is a simplified implementation of MUSIC that leaves out certain aspects,
    e.g. determining the optimal image subspace. There is no guarantee that it performs as described in
    the original paper.

    :param spectra: 2D array of shape (n_spectra, bands)
    :param image: 3D array of shape (rows, columns, bands)
    :param n_eigenvector: integer, the first n eigenvectors of the image to be retained, must be equal to or smaller
    than the number of bands in image
    :return: 1D array containing MUSIC distances of each spectrum relative to the image subspace
    """

    if spectra.shape[1] != image.shape[1]:
        raise ValueError('number of bands in library and image must be equal')

    if not n_eigenvector:
        n_eigenvector = spectra.shape[1]

    # check that n_eigenvectors doesn't exceed the number of image bands, adjust if needed
    if n_eigenvector > spectra.shape[1]:
        n_eigenvector = spectra.shape[1]

    image = copy.deepcopy(image)
    spectra = copy.deepcopy(spectra)

    # brightness normalize the image and spectra
    image /= image.sum(axis=1).reshape(-1, 1)
    spectra /= spectra.sum(axis=1).reshape(-1, 1)

    # get eigenvalues and eigenvectors of the image-derived covariance matrix, i.e. covariance between image bands
    cov = np.cov(image, rowvar=False)
    eigval, eigvect = np.linalg.eig(cov)

    # calculate % variance described by eigenvectors
    perc_variance = eigval / np.sum(eigval)

    # sort eigenvectors from high to low
    ind = np.argsort(perc_variance)[::-1]
    eigvect = eigvect[:, ind]

    if thres_perc_variance:
        perc_variance = perc_variance[ind]
        cs_perc_variance = np.cumsum(perc_variance)
        n_eigenvector = np.where(cs_perc_variance >= thres_perc_variance)[0][0]

    """
    In the original MUSIC paper by Iordache, a technique called HySime, itself based on a paper titled 'Hyperspectral
    Subspace Identification' by Bioucas-Dias & Nascimento (2008), is used to determine the optimal number of retained
    eigenvectors. See the respective papers for more info.
    """
    eigvect = eigvect[:, :n_eigenvector]

    # calculate Euclidean distances between spectra and the hyperplane formed by the retained eigenvectors
    p = np.diag(np.ones(image.shape[1])) - np.dot(eigvect, eigvect.T)
    dist = np.sum((np.dot(p, spectra.T) ** 2), axis=0) ** 0.5
    dist /= np.sum(spectra ** 2, axis=1).squeeze() ** 0.5

    return dist


def dice(image, spectra,
         mode='max',
         n_components=None):

    """
    This is an experimental image optimization technique that retains spectra based on their Deviation from Image Center
    in Eigenspace (DICE). It essentially transforms the image to standardized PCs, with unit standard deviation and zero
    mean, and performs the same transformation on the tested spectra. Then, a distance measure is defined that assesses
    how far the tested spectra deviate from the image mean vector in this transformed space. The underlying assumption
    is that the composition of the image determines variations in the image feature space. By extension, if a tested
    spectrum has a large amplitude in normalized PC space, this indicates that the spectrum is located near or passed
    the edges of the point cloud formed by the image, and may thus be less pertinent to describe its spectral
    variability.
    By definition, this approach is sensitive to the size and composition of the image. It is easier to perform
    optimization on smaller images, i.e. to define a DICE threshold that allows separating pertinent from non-pertinent
    spectra. A possible context-sensitive way to define this threshold is to compute DICE values for each pixel in the
    image and take a certain high percentile of these values as a threshold (e.g. 95% or 99%).
    :param image: Either 3D-array of floats with size (rows, cols, bands) or 2D-array of floats with shape (rows * cols,
    bands)
    :param spectra: 2D array of shape (n spectra, b bands)
    :param mode: string, denoting the distance measure used to define DICE, either 'l1', 'l2' or 'max'
    :param n_components: int, b by default, number of PC to use, starting from first when decreasingly ordered by
    variance explained
    :return: d: 1D-array of floats with shape (n spectra,), containing DICE values for each tested spectrum.
    """

    if len(image.shape) > 2:
        rows, cols, bands = image.shape
        image = image.reshape(rows * cols, bands)

    spectra = copy.deepcopy(spectra)
    image = copy.deepcopy(image)

    pca = PCA(n_components=n_components)
    pca.fit(image)
    image = pca.transform(image)

    scaler = StandardScaler()
    scaler.fit(image)

    spectra_pc = pca.transform(spectra)
    d = scaler.transform(spectra_pc)
    d = np.abs(d)

    if mode == 'l1':
        d = d.mean(axis=1)
    elif mode == 'l2':
        d = np.mean(d**2, axis=1)**0.5
    elif mode == 'max':
        d = d.max(axis=1)

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