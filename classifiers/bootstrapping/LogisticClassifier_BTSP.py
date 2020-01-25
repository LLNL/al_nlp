# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 10:51:22
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-08-08 14:19:52
import numpy as np
from sklearn import linear_model
from ..base import BaseEstimator
from ..helpers import compute_probabilities


class LogisticClassifier_BTSP(BaseEstimator):

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

    def __init__(self, nb_components, sampling_perc, name='LC-BTSP'):

        super().__init__(name)
        assert nb_components > 0
        self.nb_components = nb_components
        assert (sampling_perc > 0) and (sampling_perc < 1)
        self.sampling_perc = sampling_perc
        self.model = list()
        self.classes = None

    def fit(self, x, y, **kwargs):
        """Perform boostrapping and train multiple (deterministic) models.

        Args:
            x (np.array): design matrix (inputa data)
            y (np.array): labels
            **kwargs: additional parameters
        """
        self.classes = np.sort(np.unique(y))
        for i in range(self.nb_components):
            # select samples from the training set
            nb_samples = int(self.sampling_perc * x.shape[0])
            ids = np.random.choice(range(x.shape[0]), size=nb_samples, replace=True)

            x_tr = x[ids, :].copy()
            y_tr = y[ids].copy()
            self.model.append(linear_model.LogisticRegression(solver='lbfgs',
                                                              multi_class='ovr'))
            self.model[i].fit(x_tr, y_tr)

    def predict(self, x):
        """Perform prediction for all models in the ensemble.

        Args:
            x (np.array): input data

        Returns:
            np.array: multiple predictions for each sample x_i
        """
        yhat = np.zeros((x.shape[0], self.nb_components))
        for i in range(self.nb_components):
            yhat[:, i] = self.model[i].predict(x)

        yhat_prob = np.zeros((x.shape[0], self.classes.size))
        for i, yhat_i in enumerate(yhat):
            yhat_prob[i, :] = compute_probabilities(yhat_i, self.classes)

        return yhat_prob
