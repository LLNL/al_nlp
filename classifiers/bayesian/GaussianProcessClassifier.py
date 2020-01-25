# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 17:03:18
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-08-08 17:15:47
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from ..base import BaseEstimator


class GaussianProcess(BaseEstimator):

    """Implement a Gaussian Process Classifier. GP is by definition a
    Bayesian model, so uncertainty on the prediction is easy acquired.

    Attributes:
        model (TYPE): Description

    """

    def __init__(self, name='GP'):

        super().__init__(name)
        self.model = None

    def fit(self, x, y, **kwargs):
        """Train a Gaussian Process Classifier.

        Args:
            x (np.array): design matrix (inputa data)
            y (np.array): labels
            **kwargs: additional parameters
        """
        # Specify Gaussian Processes with fixed and optimized hyperparameters
        self.model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                               multi_class='one_vs_rest')

        self.model.fit(x, y)

    def predict(self, x):
        """Perform prediction using GP.

        Args:
            x (np.array): input data

        Returns:
            np.array: multiple predictions for each sample x_i
        """
        return self.model.predict_proba(x)
