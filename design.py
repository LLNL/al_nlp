# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 09:39:33
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-10-29 17:18:08
import os
import types
import numpy as np
import copy
import shutil
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from datasets import Dataset
# from acquisition_functions import compute_entropy_from_prob
from utils import config
import acquisition_functions


class ActiveLearningLoop(object):
    def __init__(self, name):
        """Class constructor.

        Args:
            methods (list): List of method to be executed.
            dataset (dataset): Dataset instance.
        """
        self.name = name

    def _check_inputs(self, methods, dataset, train_perc,
                      test_perc, label_block_size):

        # it has to be a list, otherwise everything down will break
        assert isinstance(methods, list)
        # list has to have at least one method
        assert len(methods) > 0
        self.methods = methods

        assert isinstance(dataset, Dataset)
        self.dataset = dataset

        assert (train_perc > 0) and (train_perc < 1)
        self.train_perc = train_perc

        assert (test_perc > 0) and (test_perc < 1)
        self.test_perc = test_perc

        assert (label_block_size > 0) and (label_block_size < 1)
        self.label_block_size = label_block_size

    def execute(self, dataset, methods,
                train_perc=0.2, test_perc=0.3,
                label_block_size=0.05,
                nb_runs=1, report_only=False):
        """Run active learning loop.

        Args:
            dataset (Dataset): Dataset object.
            methods (list): List of methods to compare.
            train_perc (float, optional): Description
            test_perc (float, optional): Description
            label_block_size (float, optional): Description
            nb_runs (int, optional): Description
        """
        if report_only:
            return

        self._check_inputs(methods, dataset, train_perc,
                           test_perc, label_block_size)

        # get list of implemented acquisition functions
        acq_funcs = {a: acquisition_functions.__dict__.get(a)
                     for a in dir(acquisition_functions)
                     if isinstance(acquisition_functions.__dict__.get(a),
                                   types.FunctionType)}

        directory = os.path.join(config.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)

        # make a new directory with experiment name
        os.makedirs(directory)

        for run_i in range(nb_runs):

            print('Run: {}'.format(run_i))

            run_directory = os.path.join(directory, 'run_{}'.format(run_i + 1))
            # perform stratified train/test/unlabel split
            ids_train = np.array([], dtype=int)
            ids_test = np.array([], dtype=int)
            ids_unlab = np.array([], dtype=int)
            for c in np.unique(self.dataset.y):
                # get all samples with label c
                ids_class = np.where(self.dataset.y == c)[0]
                idx = np.random.permutation(ids_class.size)

                nb_test = int(self.test_perc * ids_class.size)
                nb_train = int(self.train_perc * ids_class.size)

                ids_train = np.concatenate([ids_train,
                                            ids_class[idx[:nb_train]].copy()])
                ids_test = np.concatenate([ids_test,
                                           ids_class[idx[nb_train:(nb_train + nb_test)]].copy()])
                ids_unlab = np.concatenate([ids_unlab,
                                            ids_class[idx[(nb_train + nb_test):]].copy()])

            # tt = np.concatenate([ids_train, ids_test, ids_unlab])

            # shuffle data from all classes
            np.random.shuffle(ids_train)
            np.random.shuffle(ids_test)
            np.random.shuffle(ids_unlab)

            # nb_test = int(self.test_perc * self.dataset.X.shape[0])
            # nb_train = int(self.train_perc * self.dataset.X.shape[0])
            # ids = np.random.permutation(self.dataset.X.shape[0])

            # ids_train = ids[:nb_train].copy()
            # ids_test = ids[nb_train:(nb_train + nb_test)].copy()
            # ids_unlab = ids[(nb_train + nb_test):].copy()

            # each method will have it's own dataset, as it
            # will be altered during the active learning loop
            for method in self.methods:

                # add one dataset object to each model
                method['dataset'] = copy.deepcopy(self.dataset)
                method['dataset'].data_split(ids_train, ids_unlab, ids_test)

                # set method's output directory
                method_directory = os.path.join(run_directory, method['model'].__str__())
                # create directory to save method's results/logs
                os.makedirs(method_directory)
                method['model'].set_output_directory(method_directory)

            tolabel_block_size = int(self.label_block_size * self.dataset.X.shape[0])

            self.perf_on_test = {}
            for method in self.methods:
                self.perf_on_test[method['model'].__str__()] = list()
            self.train_ss = list()

            while True:

                get_out = False
                for m_id, method in enumerate(self.methods):

                    method['model'].fit(method['dataset'].train['x'],
                                        method['dataset'].train['y'],
                                        path_to_output=directory)

                    # select 'tolab_block_size' samples
                    nb_to_label = np.minimum(tolabel_block_size,
                                             method['dataset'].unlabeled['x'].shape[0])

                    # for efficiency: doesn't need to run forward to get predictions
                    # as it will not be used for random acquisition function
                    if method['acq_func'] != 'random':
                        # make predictions on the unlabeled set
                        y_pred = method['model'].predict(method['dataset'].unlabeled['x'])

                    else:
                        # will not be used by random acquisition function
                        # we just need to get the correct size
                        y_pred = np.zeros(method['dataset'].unlabeled['x'].shape)

                    # compute acquisition function
                    h_pred = acq_funcs[method['acq_func']](y_pred)

                    ids_sorted_samples = (-h_pred).argsort()
                    ids_to_label = ids_sorted_samples[:nb_to_label]

                    method['dataset'].update_data_sets(ids_to_label)

                    # compute performance on the test set
                    y_pred = method['model'].predict(method['dataset'].test['x'])
                    y_true = method['dataset'].test['y']

                    if len(y_pred.shape) > 1:
                        acc = metrics.accuracy_score(y_true,
                                                     np.argmax(y_pred, axis=1))
                    else:
                        acc = metrics.accuracy_score(y_true, y_pred)

                    self.perf_on_test[method['model'].__str__()].append(acc)

                    if m_id == 0:
                        self.train_ss.append(method['dataset'].train['x'].shape[0] - nb_to_label)

                        if (method['dataset'].unlabeled['x'].shape[0] == 0):
                            get_out = True

                if get_out:
                    break

            with open(os.path.join(run_directory, 'results.pkl'), 'wb') as fh:
                pickle.dump([copy.deepcopy(self.train_ss), copy.deepcopy(self.perf_on_test)], fh)

    def generate_report(self, plot_spread=True):
        """Generate a pdf with results.
        """
        experiment_dir = os.path.join(config.path_to_output, self.name)

        # store all results from the pkl files in a dict
        result_contents = {}

        pdf_filename = os.path.join(config.path_to_output, self.name, self.name + '.pdf')
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        fig = plt.figure()

        for run in next(os.walk(experiment_dir))[1]:
            run_dir = os.path.join(experiment_dir, run)

            with open(os.path.join(run_dir, 'results.pkl'), 'rb') as fh:
                # dict with each task result as a key
                # for each key is assigned a dict w/ task specific results
                sample_sizes, perf = pickle.load(fh)
                for method in perf.keys():
                    if method not in result_contents.keys():
                        result_contents[method] = list()
                    result_contents[method].append(perf[method])

        colors = ['red', 'green', 'blue', 'black', 'magenta', 'gray']
        for i, method in enumerate(result_contents.keys()):
            result_contents[method] = np.array(result_contents[method])
            mean = result_contents[method].mean(axis=0)
            std = result_contents[method].std(axis=0)
            plt.plot(sample_sizes, mean, '-', color=colors[i], label=method)
            if plot_spread:
                plt.fill_between(sample_sizes, mean - std, mean + std,
                                 color=colors[i], alpha=0.15)
        plt.xlabel('number of training samples')
        plt.ylabel('Accuracy on test')
        plt.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        pdf.close()
