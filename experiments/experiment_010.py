# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 10:28:49
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-10-30 10:08:45
import sys
sys.path.append('../')
from datasets import CIFAR10
from classifiers.deterministic.LogisticClassifier import LogisticClassifier
from classifiers.deterministic.ConvNeuralNetwork import ConvNeuralNetwork
from design import ActiveLearningLoop


if __name__ == '__main__':

    dataset = CIFAR10()
    dataset.prepare()

    n_epochs = 100

    methods = [{'model': ConvNeuralNetwork(name='CNN-Random', epochs=n_epochs), 'acq_func': 'random'},
               {'model': ConvNeuralNetwork(name='CNN-Entropy', epochs=n_epochs), 'acq_func': 'entropy'},
               {'model': ConvNeuralNetwork(name='CNN-Abstention', epochs=n_epochs), 'acq_func': 'abstention'},
               {'model': ConvNeuralNetwork(name='CNN-Abstention-Entropy-A', epochs=n_epochs), 'acq_func': 'abstention_entropy_amean'},
               {'model': ConvNeuralNetwork(name='CNN-Abstention-Entropy-H', epochs=n_epochs), 'acq_func': 'abstention_entropy_hmean'}, ]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.1,
                     test_perc=0.3,
                     label_block_size=0.05,
                     nb_runs=5)
    exp_loop.generate_report(plot_spread=True)
