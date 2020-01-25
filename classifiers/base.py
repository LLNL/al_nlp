# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-08 09:42:58
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-08-08 10:33:52
from abc import ABCMeta, abstractmethod


class BaseEstimator(object):
    """ Abstract class representing a generic Method. """
    __metaclass__ = ABCMeta

    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            name (str): name to be used as reference
        """
        self.name = name
        self.output_directory = ''
        # self.logger = Logger()

    def __str__(self):
        return self.name
        # pars = self.get_params()
        # if pars is not None:
        #     pars_list = list()
        #     for k in pars.keys():
        #         value = '{0:.3f}'.format(pars[k])
        #         pars_list.append('{}={}'.format(str(k), value))
        #     # ref = '.'.join(['{}={}'.format(str(k), pars[k]) for k in pars.keys()])
        #     ref = '.'.join(pars_list)
        #     ref = ref.replace('_', '').replace('.', '_')
        # return '{}_{}'.format(self.name, ref)

    @abstractmethod
    def fit(self):
        """
        Train method's parameters.
        Args
            name (np.array()):
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Perform prediction.
        Args
            name (np.array()):
        Return
            name (np.array()):
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Perform prediction.
        Args
            name (np.array()):
        Return
            name (np.array()):
        """
        pass

    @abstractmethod
    def set_params(self):
        """
        Set method's parameters.
        Args
            name (np.array()):
        """
        pass

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        # self.logger.set_path(output_dir)
        # self.logger.setup_logger(self.__str__())
