# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-10-28 14:12:47
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-06 09:01:59
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..base import BaseEstimator

from ..helpers import dac_loss, shuffle_data

# for numerical stability
epsilon = 1e-7

# this might be changed from inside dac_sandbox.py
# total_epochs = 200
alpha_final = 1.0
alpha_init_factor = 64.

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNeuralNetwork(BaseEstimator):

    def __init__(self, name='Conv-NN', epochs=100, batch_size=64):

        super().__init__(name)
        self.model = None
        self.total_epochs = epochs
        self.learn_epochs = 10
        self.batch_size = batch_size
        self.nb_classes = -1  # to be defined later
        self.model_path = ''

    def fit(self, x, y, verbose=False, **kwargs):
        self.nb_classes = np.unique(y).size + 1
        self.model = Net(self.nb_classes)
        self.model.to(DEVICE)
        criterion = dac_loss(model=self.model,
                             learn_epochs=self.learn_epochs,
                             total_epochs=self.total_epochs,
                             use_cuda=True)

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        n_steps = int(x.shape[0] / self.batch_size)

        self.model.train()

        for epoch in range(self.total_epochs):  # loop over the dataset multiple times

            if (epoch % 10) == 0:
                print('Epoch: {}'.format(epoch))
            batch_x, batch_y = shuffle_data(x, y)

            beg = 0
            end = self.batch_size

            running_loss = 0.0
            for i in range(n_steps):

                # get the inputs; data is a list of [inputs, labels]
                inputs = torch.from_numpy(batch_x[beg:end]).to(DEVICE)
                labels = torch.from_numpy(batch_y[beg:end]).to(DEVICE)

                beg += self.batch_size
                end += self.batch_size

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels, epoch)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def predict(self, x):
        """Run forward in the model to get prediction.

        Args:
            x (np.array): samples to predict

        Returns:
            np.array: predicted label
        """
        self.model.eval()
        pred = self.model(torch.from_numpy(x).to(DEVICE))
        p_out = F.softmax(pred, dim=1)
        return p_out.cpu().detach().numpy()


class Net(nn.Module):
    def __init__(self, nb_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
