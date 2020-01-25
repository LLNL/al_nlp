# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 11:32:57
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-11 10:32:43
import sys
sys.path.append('../')
from datasets import TwentyNewsGroups, PathologyReports
from feature_extraction.feature_extraction import BERT, BOW_DimReduction
from classifiers.deterministic.LogisticClassifier import LogisticClassifier
from classifiers.bayesian.GaussianProcessClassifier import GaussianProcess
from design import ActiveLearningLoop


if __name__ == '__main__':

    fe = BOW_DimReduction(features_dim=100, projection='PCA', remove_stop_words=True)
    # fe = BOW_TopicModel(nb_topics=30)
    # fe = BERT(sentence_len=20)
    # dataset = TwentyNewsGroups(fe)
    dataset = PathologyReports('GTKum', fe)
    dataset.prepare()

    methods = [  # {'model': LogisticClassifier(name='LogClass-random'), 'acq_func': 'random'},
        #{'model': LogisticClassifier(name='LogClass-entropy'), 'acq_func': 'entropy'},
        {'model': GaussianProcess(name='GP-random'), 'acq_func': 'random'},
        {'model': GaussianProcess(name='GP-entropy'), 'acq_func': 'entropy'}]

    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.2,
                     test_perc=0.2,
                     label_block_size=0.02,
                     nb_runs=10)
    exp_loop.generate_report(plot_spread=True)
