# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-11-06 09:01:38
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-11-06 09:01:46
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base import BaseEstimator
from ..helpers import dac_loss, shuffle_data

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ResNet(BaseEstimator):

    def __init__(self, name, model_type, epochs=100, batch_size=64):

        assert model_type in ['ResNet18', 'ResNet34', 'ResNet50',
                              'ResNet101', 'ResNet152']

        super().__init__(name)
        self.model = None
        self.model_type = model_type
        self.total_epochs = epochs
        self.learn_epochs = 10
        self.batch_size = batch_size
        self.nb_classes = -1  # to be defined later
        self.model_path = ''

    def fit(self, x, y, verbose=False, **kwargs):
        self.nb_classes = np.unique(y).size + 1

        if self.model_type == 'ResNet18':
            self.model = ResNet_Base(BasicBlock, [2, 2, 2, 2],
                                     num_classes=self.nb_classes)
        elif self.model_type == 'ResNet34':
            self.model = ResNet_Base(BasicBlock, [3, 4, 6, 3],
                                     num_classes=self.nb_classes)
        elif self.model_type == 'ResNet50':
            self.model = ResNet_Base(Bottleneck, [3, 4, 6, 3],
                                     num_classes=self.nb_classes)
        elif self.model_type == 'ResNet101':
            self.model = ResNet_Base(Bottleneck, [3, 4, 23, 3],
                                     num_classes=self.nb_classes)
        elif self.model_type == 'ResNet152':
            self.model = ResNet_Base(Bottleneck, [3, 8, 36, 3],
                                     num_classes=self.nb_classes)

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

        beg = 0
        end = self.batch_size
        y_hat = np.zeros((x.shape[0], self.nb_classes))

        while beg < x.shape[0]:

            # get the inputs
            inputs = torch.from_numpy(x[beg:end]).to(DEVICE)

            pred = self.model(inputs)
            p_out = F.softmax(pred, dim=1)
            y_hat[beg:end] = p_out.cpu().detach().numpy()

            beg += self.batch_size
            end += self.batch_size

        return y_hat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Base(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Base, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
