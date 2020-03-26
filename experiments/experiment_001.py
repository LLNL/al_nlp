# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 10:28:49
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-10-02 16:25:09
import sys
sys.path.append('../')
from datasets import TwentyNewsGroups, PathologyReports
from feature_extraction.feature_extraction import BERT, BOW_DimReduction
# from classifiers.bayesian.BayesianLinearClassifier import BayesianLinearClassifier
# from classifiers.bayesian.BayesianNeuralNet import BayesianNeuralNet
from classifiers.deterministic.LogisticClassifier import LogisticClassifier
# from classifiers.deterministic.NeuralNetwork import NeuralNet
from design import ActiveLearningLoop


if __name__ == '__main__':

    fe = BOW_DimReduction(features_dim=50, projection='PCA', remove_stop_words=True)
    # fe = BOW_TopicModel(nb_topics=30)
    # fe = BERT(sentence_len=20)

    dataset = TwentyNewsGroups(fe)
    dataset.prepare()

    # methods = [{'model': BayesianNeuralNet(name='BDNNr', epochs=2), 'acq_func': 'random'},
    #            {'model': BayesianNeuralNet(name='BDNNe', epochs=2), 'acq_func': 'entropy'}]
    methods = [{'model': LogisticClassifier(name='LogClass-random'), 'acq_func': 'random'},
               {'model': LogisticClassifier(name='LogClass-entropy'), 'acq_func': 'entropy'},
               {'model': LogisticClassifier(name='LogClass-margin_samp'), 'acq_func': 'margin_sampling'},
               {'model': LogisticClassifier(name='LogClass-least_conf'), 'acq_func': 'least_confidence'}, ]

    # {'model': NeuralNet(name='NN-random', epochs=30), 'acq_func': 'random'},
    # {'model': NeuralNet(name='NN-entropy', epochs=30), 'acq_func': 'entropy'}]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.2,
                     test_perc=0.2,
                     label_block_size=0.02,
                     nb_runs=10)
    exp_loop.generate_report(plot_spread=True)
