# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-09-03 16:50:08
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-09 07:38:50
import sys
sys.path.append('../')
from datasets import TwentyNewsGroups, PathologyReports
from feature_extraction.feature_extraction import BERT, BOW_DimReduction
# from classifiers.bayesian.BayesianLinearClassifier import BayesianLinearClassifier
# from classifiers.bayesian.BayesianNeuralNet import BayesianNeuralNet
# from classifiers.deterministic.LogisticClassifier import LogisticClassifier
from classifiers.deterministic.NeuralNetwork import NeuralNet
from design import ActiveLearningLoop


if __name__ == '__main__':

    # dataset = TwentyNewsGroups('BOW_DimReduction')
    fe = BOW_DimReduction(features_dim=100, projection='PCA')
    # fe = BOW_TopicModel(nb_topics=30)
    # fe = BERT(sentence_len=20)

    dataset = PathologyReports('GTKum', fe)
    dataset.prepare()

    methods = [{'model': NeuralNet(name='NN-entropy', epochs=20), 'acq_func': 'entropy'},
               {'model': NeuralNet(name='NN-random', epochs=20), 'acq_func': 'random'},
               {'model': NeuralNet(name='NN-margin_samp', epochs=20), 'acq_func': 'margin_sampling'}, ]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.2,
                     test_perc=0.2,
                     label_block_size=0.02,
                     nb_runs=10,
                     report_only=True)
    exp_loop.generate_report(plot_spread=True)
