# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 09:45:23
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-06 09:27:31
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.datasets import fetch_20newsgroups
import torch
import torchvision
import torchvision.transforms as transforms
from feature_extraction.feature_extraction import *
from utils import config


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name

        self.train = {}
        self.unlabeled = {}
        self.test = {}

    @abstractmethod
    def prepare(self):
        pass

    def data_split(self, ids_train, ids_unlab, ids_test):
        """Split data into training, validation, test.
        We have to pass ids_train and ids_test as
        all datasets (one for each method) must
        have the same training and test set at
        the beginning of the active learning loop.
        Test set will remain the same throughout
        the process, but training will be altered,
        as the unlabeleb samples will selected by
        the AL method.

        Args:
            ids_train (np.array): training set samples id
            ids_unlab (np.array): unlabeled set samples id
            ids_test (np.array): test set samples id
        """
        self.train['x'] = self.X[ids_train, :].copy()
        self.train['y'] = self.y[ids_train].copy()

        self.test['x'] = self.X[ids_test, :].copy()
        self.test['y'] = self.y[ids_test].copy()

        self.unlabeled['x'] = self.X[ids_unlab, :].copy()
        self.unlabeled['y'] = self.y[ids_unlab].copy()

    def update_data_sets(self, ids):
        """Update the list of samples in the training and unlabeled set.
        It emulates the process of labeling unlabeled dataset and moving
        it to the training set.

        Args:
            ids (np.array): ids of the unlabeled samples to be unlabeled
        """
        # move samples from the unlabeled set to the training set
        # simulating the process of label a block of unlabeled data
        self.train['x'] = np.vstack((self.train['x'], self.unlabeled['x'][ids]))
        self.train['y'] = np.concatenate((self.train['y'], self.unlabeled['y'][ids]))

        # remove the set of samples that has just been labeled
        # from the unlabeled set
        self.unlabeled['x'] = np.delete(self.unlabeled['x'], ids, axis=0)
        self.unlabeled['y'] = np.delete(self.unlabeled['y'], ids, axis=0)

    def _feature_extraction(self):
        """Perform one of the few feature extraction methods available.

        Returns:
            X, y (np.array): Extracted features and labels.

        Raises:
            ValueError: Unknown feature extraction method.
        """
        # if self.feature_extraction == 'BOW_DimReduction':
        #     fe = BOW_DimReduction(features_dim=100,
        #                           projection='PCA')
        # elif self.feature_extraction == 'BOW_TopicModel':
        #     fe = BOW_TopicModel(nb_topics=30)
        # elif self.feature_extraction == 'BERT':
        #     fe = BERT()
        # else:
        #     raise ValueError("Unknown feature extraction: {}".format(self.feature_extraction))
        self.X, self.y = self.fe.extract_features(self)

    def is_unlabeled_empty(self):
        """Check whether all unlabeled samples have
         been labeled.

        Returns:
            Bool: unlabeled bucket is empty of not
        """
        return (len(self.unlabeled) == 0)

    def get_name(self):
        """Return model's name.

        Returns:
            str: Model's name.
        """
        return self.name


class TwentyNewsGroups(Dataset):

    def __init__(self, feature_extraction):
        super().__init__('20News_Groups')
        self.fe = feature_extraction

        # only 4 classes
        categories = ['alt.atheism',
                      'talk.religion.misc',
                      'comp.graphics',
                      'sci.space']

        remove = ('headers', 'footers', 'quotes')

        # 20 news group dataset
        dataset = fetch_20newsgroups(config.path_to_data,
                                     categories=categories,
                                     shuffle=True,
                                     random_state=42,
                                     remove=remove)
        self.data = dataset.data
        self.target = dataset.target
        self.feature_extraction = feature_extraction

    def prepare(self):
        super()._feature_extraction()


class PathologyReports(Dataset):

    def __init__(self, target_var, feature_extraction):
        super().__init__('PathologyReports')
        self.fe = feature_extraction

        # 20 news group dataset
        fname = os.path.join(config.path_to_processed_path_reports,
                             'labeled_reports.csv')
        df = pd.read_csv(fname)

        # remove all rows where target_var is nan
        df = df.dropna(subset=[target_var])
        df = df.groupby(target_var).filter(lambda x: len(x) >= 5)

        if target_var != 'PR' and target_var != 'ER':
            # map string to numbers
            mapping = dict()
            for i, c in enumerate(df[target_var].unique()):
                mapping[c] = i
            # replace strings by numbers (label encoding)
            df = df.replace({target_var: mapping})

        self.data = df['Report'].to_list()
        self.target = df[target_var].values.astype(int)

        # remove nans
        self.feature_extraction = feature_extraction

    def prepare(self):
        super()._feature_extraction()


class CIFAR10(Dataset):

    def __init__(self):
        super().__init__('CIFAR10')

        transform = None

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

        self.trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                     download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                    download=True, transform=transform)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

    def normalize(self, x):
        """ Apply normalization on the images. """
        x /= 255.
        x.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return x

    def prepare(self):
        """ Convert dataloader type to tensors. """

        x_train = torch.from_numpy(self.trainset.data).type(torch.float32)
        x_test = torch.from_numpy(self.testset.data).type(torch.float32)

        y_train = torch.tensor(self.trainset.targets, dtype=torch.int64)
        y_test = torch.tensor(self.testset.targets, dtype=torch.int64)

        self.X = torch.cat((x_train, x_test), 0)
        self.X = self.X.permute(0, 3, 1, 2)

        # transform by hand
        self.X = self.normalize(self.X)

        self.X = self.X.numpy()
        self.y = torch.cat((y_train, y_test), 0).numpy()

        # self.X = self.X[0:500]
        # self.y = self.y[0:500]

        print('data.shape: {}'.format(self.X.shape))
        print('target.shape: {}'.format(self.y.shape))


class SVHN(Dataset):

    def __init__(self):
        super().__init__('SVHN')

        transform = None

        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])

        self.trainset = torchvision.datasets.SVHN(root='../data', split='train',
                                                  download=True, transform=transform)
        self.testset = torchvision.datasets.SVHN(root='../data', split='test',
                                                 download=True, transform=transform)

    def normalize(self, x):
        """ Apply normalization on the images. """
        x /= 255.
        x.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return x

    def prepare(self):
        """ Convert dataloader type to tensors. """

        x_train = torch.from_numpy(self.trainset.data).type(torch.float32)
        x_test = torch.from_numpy(self.testset.data).type(torch.float32)

        y_train = torch.tensor(self.trainset.labels, dtype=torch.int64)
        y_test = torch.tensor(self.testset.labels, dtype=torch.int64)

        self.X = torch.cat((x_train, x_test), 0)

        # transform by hand
        self.X = self.normalize(self.X)

        self.X = self.X.numpy()
        self.y = torch.cat((y_train, y_test), 0).numpy()

        # self.X = self.X[0:500]
        # self.y = self.y[0:500]

        print('data.shape: {}'.format(self.X.shape))
        print('target.shape: {}'.format(self.y.shape))
