# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 09:49:42
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-08-07 11:47:32
import numpy as np


def compute_probabilities(y, y_classes):
    """Compute frequency probabilities from a list of prediction.

    Args:
        y (np.array): array of labels
        y_classes (np.array): Probability distribution support

    Returns:
        TYPE: Description
    """
    pk = np.zeros(y_classes.shape)
    y = y.ravel()
    for i in y_classes:
        pk[i] = np.count_nonzero(y == i)
    pk = pk / y.shape[0]
    return pk
