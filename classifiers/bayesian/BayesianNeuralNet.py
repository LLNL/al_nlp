# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-05-17 09:44:12
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-20 16:16:55
import os
import numpy as np
from time import time
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense, Wrapper, InputSpec
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from ..base import BaseEstimator
from ..helpers import compute_probabilities


class BayesianNeuralNet(BaseEstimator):

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

    def __init__(self, name='Bayesian NN', epochs=100,
                 batch_size=64, arch=[20, 20]):

        super().__init__(name)
        assert len(arch) >= 1
        self.arch = arch
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, x, y, verbose=False, **kwargs):
        """ Train a Bayesian Neural Network with Concrete Dropout.

        Args:
            x (np.array): input data (design matrix)
            y (np.array): labels
            verbose (bool, optional): Print or not info about the training process.
            **kwargs: Additional parameters.
        """
        ell = 1e-4
        N = x.shape[0]
        wd = ell**2. / N
        dd = 2. / N

        if len(y.shape) > 1:
            nb_classes = y.shape[1]
        else:
            nb_classes = np.unique(y).size
            y = self.one_hot_encoder.fit_transform(y[:, np.newaxis])

        self.model = Sequential()
        self.model.add(ConcreteDropout(Dense(self.arch[0], activation='tanh'), input_shape=(x.shape[1],)))
        for n in range(1, len(self.arch)):
            self.model.add(ConcreteDropout(Dense(self.arch[n], activation='tanh'),
                                           weight_regularizer=wd, dropout_regularizer=dd))
        self.model.add(ConcreteDropout(Dense(nb_classes, activation='softmax'),
                                       weight_regularizer=wd, dropout_regularizer=dd))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir=os.path.join(self.output_directory, "logs/{}".format(time())),
                                  histogram_freq=0,
                                  write_graph=True)
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[tensorboard], verbose=verbose)
        # self.model.summary()

    def predict(self, x, return_prob=False):
        '''
        '''
        yhat_prob = self.model.predict(x)
        print(yhat_prob.shape)
        # if not return_prob:
        yhat = np.argmax(yhat_prob, axis=1)
        # print(yhat.shape)
        # print('-' * 10)
        return yhat_prob


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                             shape=(1,),
                                             initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                             trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * tf.cast(input_dim, dtype=tf.float32)
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (K.log(self.p + eps) -
                     K.log(1. - self.p + eps) +
                     K.log(unif_noise + eps) -
                     K.log(1. - unif_noise + eps))
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


if __name__ == '__main__':

    # Loading the Iris dataset
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    encoder.fit(y[:, np.newaxis])
    encoded_Y = encoder.transform(y[:, np.newaxis])
    dummy_y = np.array(encoded_Y)

    clf = BayesianNeuralNet()
    clf.fit(X, dummy_y, epochs=200)
    y_pred = clf.predict(X, return_prob=True)
    print(y_pred)
    print("Accuracy=", accuracy_score(y, np.array(y_pred)))
