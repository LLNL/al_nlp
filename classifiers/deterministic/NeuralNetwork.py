# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 17:17:14
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-04 09:30:34
import os
import numpy as np
from time import time
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow.keras.callbacks as callbacks
from keras.optimizers import Adam

from sklearn.preprocessing import OneHotEncoder
from ..base import BaseEstimator
# from ..helpers import compute_probabilities


class NeuralNet(BaseEstimator):

    """Implement a Bayesian Neural Network using the idea of
    concrete dropout (Gal et.al, 2017)

    Attributes:
        arch (list): NN architecture (list of number of nodes per layer)
        batch_size (int): Batch size for SGD
        epochs (int): Number of training epochs
        model (TF Model): Variable storing TF model
        name (str): string containing a label for the model (use for plotting)
        one_hot_encoder (sklearn OHE): Encode categorical label variable into OHE
    """

    def __init__(self, name='NN', epochs=100,
                 batch_size=64, arch=[20, 20]):

        super().__init__(name)
        assert len(arch) >= 1
        self.arch = arch
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.model_path = ''

    def fit(self, x, y, verbose=False, **kwargs):
        """ Train a Bayesian Neural Network with Concrete Dropout.

        Args:
            x (np.array): input data (design matrix)
            y (np.array): labels
            verbose (bool, optional): Print or not info about the training process.
            **kwargs: Additional parameters.
        """
        if len(y.shape) > 1:
            nb_classes = y.shape[1]
        else:
            nb_classes = np.unique(y).size
            y = self.one_hot_encoder.fit_transform(y[:, np.newaxis])

        # This returns a tensor
        inputs = Input(shape=(x.shape[1],))

        # a layer instance is callable on a tensor, and returns a tensor
        model = Dense(self.arch[0], activation='tanh')(inputs)
        for i in range(1, len(self.arch)):
            model = Dense(self.arch[i], activation='tanh')(model)
        predictions = Dense(nb_classes, activation='softmax')(model)

        # This creates a model that includes
        # the Input layer and three Dense layers
        self.model = Model(inputs=inputs, outputs=predictions)
        # self.model.summary()

        opt = Adam(lr=0.001,
                   beta_1=0.9,
                   beta_2=0.999)
        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # log_dir = os.path.join(self.output_directory,
        # "logs/{}".format(time()))
        # tensorboard = callbacks.TensorBoard(log_dir=log_dir,
        # histogram_freq=0,
        # write_graph=True)
        model_path = os.path.join(self.output_directory, "model.tf")
        model_checkpoint = callbacks.ModelCheckpoint(filepath=model_path,
                                                     save_best_only=True,
                                                     save_weights_only=True)
        # reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                                         patience=10, min_lr=1e-6, verbose=1)

        # callbacks_list = [tensorboard, model_checkpoint]
        callbacks_list = [model_checkpoint]

        self.model.fit(x, y, epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       callbacks=callbacks_list,
                       verbose=verbose)
        self.model.load_weights(model_path)
        # self.model.summary()

    def predict(self, x):
        """Run forward in the model to get prediction.

        Args:
            x (np.array): samples to predict

        Returns:
            np.array: predicted label
        """
        yhat = self.model.predict(x)
        return yhat
