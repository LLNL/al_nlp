# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 11:32:57
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-08-28 10:50:00
import sys
sys.path.append('../')
from datasets import TwentyNewsGroups
from feature_extraction.feature_extraction import BERT, BOW_DimReduction
from classifiers.bayesian.BayesianLinearClassifier import BayesianLinearClassifier
from design import ActiveLearningLoop


if __name__ == '__main__':

    fe = BOW_DimReduction(features_dim=100, projection='PCA')
    # fe = BOW_TopicModel(nb_topics=30)
    # fe = BERT(sentence_len=20)
    dataset = TwentyNewsGroups(fe)
    dataset.prepare()

    methods = [{'model': BayesianLinearClassifier(name='BLC-entropy'), 'acq_func': 'entropy'},
               {'model': BayesianLinearClassifier(name='BLC-margin_sampling'), 'acq_func': 'margin_sampling'},
               {'model': BayesianLinearClassifier(name='BLC-least_confidence'), 'acq_func': 'least_confidence'},
               {'model': BayesianLinearClassifier(name='BLC-random'), 'acq_func': 'random'}]
               
    exp_folder = __file__.strip('.py')
    exp_loop = ActiveLearningLoop(exp_folder)
    exp_loop.execute(dataset, methods,
                     train_perc=0.1,
                     test_perc=0.3,
                     label_block_size=0.02,
                     nb_runs=1)
    exp_loop.generate_report(plot_spread=True)
