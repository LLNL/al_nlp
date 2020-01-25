# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-10-30 10:09:00
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-10-30 10:11:34
import sys
sys.path.append('../')
from datasets import CIFAR10
from classifiers.deterministic.LogisticClassifier import LogisticClassifier
from classifiers.deterministic.DenseNet import DenseNet
from design import ActiveLearningLoop


if __name__ == '__main__':

    dataset = CIFAR10()
    dataset.prepare()

    n_epochs = 2

    methods = [{'model': DenseNet(name='DN-Random', epochs=n_epochs), 'acq_func': 'random'},
               {'model': DenseNet(name='DN-Entropy', epochs=n_epochs), 'acq_func': 'entropy'},
               {'model': DenseNet(name='DN-Abstention', epochs=n_epochs), 'acq_func': 'abstention'},
               {'model': DenseNet(name='DN-Abstention-Entropy-A', epochs=n_epochs), 'acq_func': 'abstention_entropy_amean'},
               {'model': DenseNet(name='DN-Abstention-Entropy-H', epochs=n_epochs), 'acq_func': 'abstention_entropy_hmean'}, ]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.2,
                     test_perc=0.3,
                     label_block_size=0.1,
                     nb_runs=2)
    exp_loop.generate_report(plot_spread=True)
