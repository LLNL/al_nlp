# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-10-30 10:22:55
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-06 09:01:15
'''DenseNet in PyTorch.'''
import math
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


class DenseNet(BaseEstimator):

    def __init__(self, name='DenseNet', epochs=100, batch_size=64):

        super().__init__(name)
        self.model = None
        self.total_epochs = epochs
        self.learn_epochs = 10
        self.batch_size = batch_size
        self.nb_classes = -1  # to be defined later
        self.model_path = ''

    def fit(self, x, y, verbose=False, **kwargs):
        self.nb_classes = np.unique(y).size + 1
        self.model = DenseNet_Base(Bottleneck, [6, 12, 24, 16],
                                   growth_rate=12,
                                   num_classes=self.nb_classes)
        # self.model.to(DEVICE)
        # self.model.train()
        print(x.shape)

        criterion = dac_loss(model=self.model,
                             learn_epochs=self.learn_epochs,
                             total_epochs=self.total_epochs,
                             use_cuda=True)

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        n_steps = int(x.shape[0] / self.batch_size)

        self.model.to(DEVICE)

        # if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    self.model = nn.DataParallel(self.model)

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


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_Base(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet_Base, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
