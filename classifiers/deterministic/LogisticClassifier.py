# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 10:51:22
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-20 16:22:29
import numpy as np
from sklearn import linear_model
from ..base import BaseEstimator


class LogisticClassifier(BaseEstimator):

    """Implement a Logistic Regression with bootstrapping to generate
    uncertainty about prediction. This is a frequentist way of generating
    uncertainty on the predictions.

    Attributes:
        model (list): List of sklearn models in the ensemble
        nb_components (int): number of components in the ensemble
        sampling_perc (float): percentage of samples randomdly selected
                              to train a particular component in the
                              ensemble.
    """

    def __init__(self, name='LC'):

        super().__init__(name)
        self.model = None
        self.classes = None

    def fit(self, x, y, **kwargs):
        """Perform boostrapping and train multiple (deterministic) models.

        Args:
            x (np.array): design matrix (inputa data)
            y (np.array): labels
            **kwargs: additional parameters
        """
        self.classes = np.sort(np.unique(y))
        self.model = linear_model.LogisticRegression(solver='lbfgs') #,
                                                     # multi_class='multinomial')
        self.model.fit(x, y)

    def predict(self, x):
        """Perform prediction for all models in the ensemble.

        Args:
            x (np.array): input data

        Returns:
            np.array: multiple predictions for each sample x_i
        """
        return self.model.predict_proba(x)
