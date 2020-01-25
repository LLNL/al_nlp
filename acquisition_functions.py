# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 09:46:39
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-10-29 15:32:03
import numpy as np
import scipy.stats as stats


def random(pk):
    """ Random acquisition function. Generate a random
        vector of scalars. The ranking of the corresponding
        samples will be random.

    Args:
        pk (np.array): array of probability. Probability predictions
                       for each sample.

    Returns:
        np.array: array of random values
    """
    return np.random.rand(pk.shape[0])


def entropy(pk):
    """Compute entropy directly from the probability vector.

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of entropies for all samples
    """
    h = np.zeros((pk.shape[0], ))
    for i in range(pk.shape[0]):
        h[i] = stats.entropy(pk[i, :])
    return h


def margin_sampling(pk):
    """Compute margin sampling from the probability vector.

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of margin sampled values for all samples
    """
    rev = np.sort(pk, axis=-1)
    h = rev[:, 0] - rev[:, 1]
    return -h


def least_confidence(pk):
    """Compute least-confidence from the probability vector.

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of least confidence values for all samples
    """
    rev = np.sort(pk, axis=-1)
    h = 1 - rev[:, 0]
    return -h


def abstention(pk):
    """ Extract abstention probability from the probability vector

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of abstention probabilities
    """
    return pk[:, -1].ravel()


def abstention_entropy_amean(pk):
    """ Compute arithmetic mean from abstention and entropy.

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of abstention probabilities
    """
    h = 0.5 * entropy(pk) + 0.5 * pk[:, -1].ravel()
    return h


def abstention_entropy_hmean(pk):
    """ Compute harmonic mean from abstention and entropy.

    Args:
        pk (np.array): array of probabilities. Each row
                       is the vector of probability for
                       each class, for each sample

    Returns:
        np.array: array of abstention probabilities
    """
    h0 = entropy(pk)
    h1 = pk[:, -1].ravel()
    h = (2 * h0 * h1) / (h0 + h1)
    return h

# def compute_entropy_from_prob(pk):
#     """Compute entropy directly from the probability vector.

#     Args:
#         pk (np.array): array of probabilities

#     Returns:
#         np.array: array of entropies for all samples
#     """
#     h = np.zeros((pk.shape[0], ))
#     for i in range(pk.shape[0]):
#         h[i] = stats.entropy(pk[i, :])
#     return h
