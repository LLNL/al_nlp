# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-09-03 16:45:43
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-09 07:40:50
import sys
sys.path.append('../')
from datasets import TwentyNewsGroups, PathologyReports
from feature_extraction.feature_extraction import BERT, BOW_DimReduction
from classifiers.bayesian.NeuralNet_BBB import BayesianNeuralNet_BBB
from design import ActiveLearningLoop


if __name__ == '__main__':

    fe = BOW_DimReduction(features_dim=30, projection='PCA')
    # fe = BOW_TopicModel(nb_topics=30)
    # fe = BERT(sentence_len=20)

    # dataset = TwentyNewsGroups(fe)
    dataset = PathologyReports('GTKum', fe)
    dataset.prepare()

    methods = [{'model': BayesianNeuralNet_BBB(name='NN-BBB-entropy',
                                               batch_size=16, epochs=20,
                                               arch=[10, 10]), 'acq_func': 'entropy'},
               {'model': BayesianNeuralNet_BBB(name='NN-BBB-random',
                                               batch_size=16, epochs=20,
                                               arch=[10, 10]), 'acq_func': 'random'},
               {'model': BayesianNeuralNet_BBB(name='NN-BBB-MarginSamp',
                                               batch_size=16, epochs=20,
                                               arch=[10, 10]), 'acq_func': 'margin_sampling'}]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.2,
                     test_perc=0.2,
                     label_block_size=0.02,
                     nb_runs=5)
                     # report_only=True)
    exp_loop.generate_report(plot_spread=True)
