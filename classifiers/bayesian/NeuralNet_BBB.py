# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-29 13:35:49
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-20 16:19:39
import os
import glob
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from ..base import BaseEstimator
from ..helpers import compute_probabilities

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(self, tensor_x, tensor_y):
        """ Initialization. """
        self.labels = tensor_y
        self.data = tensor_x

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.data.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data. """
        return self.data[index], self.labels[index]


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, device):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample(DEVICE)
            bias = self.bias.sample(DEVICE)
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNeuralNet(nn.Module):

    """Implement a Bayesian Neural Network using 'Bayes by Backprop'.

    Attributes:
        arch (list): NN architecture (list of number of nodes per layer)
        batch_size (int): Batch size for SGD
        input_dim (int): Input tensor dimension.
        nb_classes (int): Number of classes.
        nb_train_samples (int): number of forwards when computing
                                stochastic gradient.

    """

    def __init__(self, batch_size, input_dim,
                 nb_train_samples, num_batches,
                 nb_classes, arch):

        nn.Module.__init__(self)

        assert len(arch) >= 1
        self.arch = arch
        self.nb_classes = nb_classes
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.nb_train_samples = nb_train_samples
        self._num_batches = num_batches

        # define neural net model
        self.lay1 = BayesianLinear(input_dim, self.arch[0])
        self.lay2 = BayesianLinear(self.arch[0], self.arch[1])
        self.lay3 = BayesianLinear(self.arch[1], nb_classes)

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, new_value):
        assert new_value > 0
        self._num_batches = new_value

    def forward(self, x, sample=False):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.lay1(x, sample))
        x = F.relu(self.lay2(x, sample))
        x = F.log_softmax(self.lay3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.lay1.log_prior \
            + self.lay2.log_prior \
            + self.lay3.log_prior

    def log_variational_posterior(self):
        return self.lay1.log_variational_posterior \
            + self.lay2.log_variational_posterior \
            + self.lay3.log_variational_posterior

    def sample_elbo(self, input, target):
        outputs = torch.zeros(self.nb_train_samples, self.batch_size, self.nb_classes).to(DEVICE)
        log_priors = torch.zeros(self.nb_train_samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(self.nb_train_samples).to(DEVICE)
        for i in range(self.nb_train_samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / self.num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


class BayesianNeuralNet_BBB(BaseEstimator):

    def __init__(self, name='BBB', batch_size=64,
                 epochs=100, nb_train_samples=2,
                 nb_test_samples=30,
                 arch=[20, 20], save_every=5):
        """Bayesian Neural Net using the Bayes by Backprop model.

        Args:
            name (str): A string to indentify the model in the plots.
            batch_size (int, optional): Batch size for training.
            epochs (int, optional): Number of training epochs
            nb_test_samples (int, optional): Number of forwards in the nn
                                             to compute probabilities.
            arch (list, optional): Neural net architecture, number of
                                   nodes in each layer.
        """
        super().__init__(name)
        self.arch = arch
        self.batch_size = batch_size
        self.epochs = epochs
        self.nb_train_samples = nb_train_samples
        self.nb_test_samples = nb_test_samples
        self.save_every = save_every
        self.net = None
        self.classes = None

    def fit(self, x, y, **kwargs):

        # get validation set from training set
        ids = np.random.permutation(x.shape[0])
        # ids_tr = ids[0:int(x.shape[0] * 0.8)]
        # ids_val = ids[int(x.shape[0] * 0.8):]

        # model is float32 too. it has to match
        x = x.astype(np.float32)

        # Parameters
        params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'num_workers': 4,
                  'drop_last': True}

        # Generators
        # training_set = Dataset(x[ids_tr], y[ids_tr])
        training_set = Dataset(x[ids], y[ids])
        training_generator = data.DataLoader(training_set, **params)

        # validation_set = Dataset(x[ids_val], y[ids_val])
        # validation_generator = data.DataLoader(validation_set, **params)

        input_dim = x.shape[1]  # input dimension
        self.classes = np.unique(y)  # set of labels
        self.net = BayesianNeuralNet(self.batch_size, input_dim,
                                     self.nb_train_samples,
                                     len(training_set),
                                     self.classes.size,
                                     self.arch).to(DEVICE)

        self._num_batches = len(training_generator)

        optimizer = optim.Adam(self.net.parameters())
        for epoch in range(self.epochs):
            self.net.train()
            for batch_x, batch_y in training_generator:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)  # Transfer to GPU

                optimizer.zero_grad()
                loss, log_prior, log_var_post, neg_loglik = self.net.sample_elbo(batch_x, batch_y)
                loss.backward()
                optimizer.step()

            # checkpoint model periodically
            if epoch % self.save_every == 0:
                snapshot_prefix = os.path.join(self.output_directory, 'snapshot')
                # snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_model.pt'.format(loss.item(), epoch)
                torch.save(self.net, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

                # print('{} | {}'.format(loss, neg_loglik))

            # Validation
            # self.net.eval()
            # correct = 0
            # corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
            # with torch.set_grad_enabled(False):
            #     for batch_x, batch_y in validation_generator:
            #         # Transfer to GPU
            #         batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            #         outputs = torch.zeros(TEST_SAMPLES + 1,
            #                               self.batch_size,
            #                               self.nb_classes).to(DEVICE)
            #         for i in range(TEST_SAMPLES):
            #             outputs[i] = self(batch_x, sample=True)
            #         outputs[TEST_SAMPLES] = self(batch_x, sample=False)
            #         output = outputs.mean(0)
            #         # print(outputs.shape)
            #         preds = outputs.max(2, keepdim=True)[1]
            #         pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            #         corrects += preds.eq(batch_y.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            #         correct += pred.eq(batch_y.view_as(pred)).sum().item()
            # for index, num in enumerate(corrects):
            #     if index < TEST_SAMPLES:
            #         print('Component {} Accuracy: {}/{}'.format(index, num, len(validation_set)))
            #     else:
            #         print('Posterior Mean Accuracy: {}/{}'.format(num, len(validation_set)))
            # print('Ensemble Accuracy: {}/{}'.format(correct, len(validation_set)))

    def predict(self, x):
        """ Make predictions. """
        x = torch.from_numpy(x).float().to(DEVICE)
        self.net.eval()  # switch to evaluation mode
        with torch.no_grad():  # gradients not computed
            outputs = torch.zeros(self.nb_test_samples + 1,
                                  x.shape[0],
                                  self.classes.size).to(DEVICE)
            # run forward multiple times. recalling that this model
            # is not deterministic: every new forward gives a slightly
            # different prediction
            for i in range(self.nb_test_samples):
                outputs[i] = self.net(x, sample=True)
            outputs[self.nb_test_samples] = self.net(x, sample=False)

        # the class with highest probability is said to be the prediction
        _, ind = outputs.max(2)  # return max values and indices
        y_hat = torch.t(ind).numpy()  # transpose and to numpy format

        # compute probabilities from the multiple predictions
        yhat_prob = np.zeros((x.shape[0], self.classes.size))
        for i, yhat_i in enumerate(y_hat):
            yhat_prob[i, :] = compute_probabilities(yhat_i, self.classes)
        print(yhat_prob.shape)
        return yhat_prob


# # evaluate performance on validation set periodically
# if iterations % args.dev_every == 0:

#     # switch model to evaluation mode
#     model.eval(); dev_iter.init_epoch()

#     # calculate accuracy on validation set
#     n_dev_correct, dev_loss = 0, 0
#     with torch.no_grad():
#         for dev_batch_idx, dev_batch in enumerate(dev_iter):
#              answer = model(dev_batch)
#              n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
#              dev_loss = criterion(answer, dev_batch.label)
#     dev_acc = 100. * n_dev_correct / len(dev)

#     print(dev_log_template.format(time.time()-start,
#         epoch, iterations, 1+batch_idx, len(train_iter),
#         100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

#     # update best valiation set accuracy
#     if dev_acc > best_dev_acc:

#         # found a model with better validation set accuracy

#         best_dev_acc = dev_acc
#         snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
#         snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

#         # save model, delete previous 'best_snapshot' files
#         torch.save(model, snapshot_path)
#         for f in glob.glob(snapshot_prefix + '*'):
#             if f != snapshot_path:
#                 os.remove(f)
