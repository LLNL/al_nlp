#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:22:58 2019

@author: ray34
"""
import sklearn.datasets
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from sklearn.metrics import accuracy_score
from ..base import BaseEstimator
from ..helpers import compute_probabilities


def softmax(x):
    """Softmax function.

    Args:
        x (np.array/scalar): input x

    Returns:
        float: function value
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_most_frequent(y):
    """Compute the mode of the predictive distribution.

    Args:
        y (np.array): array of predictions

    Returns:
        np.array: mode of the predictive distribution
                  for each sample.
    """
    y = y.astype(np.int16)
    yf = np.zeros((y.shape[0],), dtype=np.int16)
    for i in range(y.shape[0]):
        counts = np.bincount(y[i, :])
        yf[i] = np.argmax(counts)
    return yf.astype(np.int16)


class BayesianLinearClassifier(BaseEstimator):
    """ Implement a Bayesian Linear Classifier """

    def __init__(self, name='Bayesian LC'):
        self.params = {'alpha': None, 'beta': None}
        # set method's name and paradigm
        super().__init__(name)  # , fit_intercept, normalize)
        self.classes = None

    def fit(self, x, y, niters=500, **kwargs):
        """ Train model """

        self.classes = np.sort(np.unique(y))
        F = x.shape[1]
        K = np.unique(y).shape[0]

        with pm.Model() as linear_model:

            # Creating the model
            beta = pm.Normal('beta', mu=0, sd=10, shape=(F, K))
            alpha = pm.Normal('alpha', mu=0, sd=10, shape=K)
            mu = tt.dot(x, beta) + alpha
            p = pm.Deterministic('p', tt.nnet.softmax(mu))
            yl = pm.Categorical('yl', p=p, observed=y)

        with linear_model:
            # trace = pm.sample(niters, njobs=1, chains=1, verbose=False)  # No U-turn sampler
            trace = pm.sample(step=pm.Metropolis(), draws=50000, njobs=1, tune=50)
            self.params['beta'] = trace['beta'][-niters:].copy()  # storing last 500 samples from sampler
            self.params['alpha'] = trace['alpha'][-niters:].copy()

    def predict(self, x):
        """ Makes prediction with the trained model """
        yhat = np.zeros((x.shape[0], self.params['alpha'].shape[0]))
        # for all MCMC models
        for i in range(self.params['beta'].shape[0]):
            alpha = self.params['alpha'][i]
            beta = self.params['beta'][i]
            predict_prob = np.dot(x, beta) + alpha
            for j in range(x.shape[0]):
                pred = np.random.multinomial(1, softmax(predict_prob[j]))
                yhat[j, i] = np.argwhere(pred == 1)[0][0].copy()
        
        yhat_prob = np.zeros((x.shape[0], self.classes.size))
        for i, yhat_i in enumerate(yhat):
            yhat_prob[i, :] = compute_probabilities(yhat_i, self.classes)
        # print(yhat_prob)

        return yhat_prob  # get_most_frequent(y_pred), y_pred.astype(np.int16),


if __name__ == '__main__':

    # Loading the Iris dataset
    X, y = sklearn.datasets.load_iris(True)
    clf = BayesianLinearClassifier(name='BLC')
    print(X.shape)
    print(y.shape)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Accuracy=", accuracy_score(y, np.array(y_pred)))
