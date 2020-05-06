# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-09-16 16:24:31
"""

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def conv_init(m):
    """
    Introduction of function
    ------------------------
    This function inits parameters in a layer.

    Parameters
    ----------
    m: torch.nn.Module
        a layer containing parameters to be inited

    Returns
    -------
    NONE
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain = np.sqrt(2))
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements basic block in mobile network.

    Variables
    ---------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block 
    stride: int
        stride used in convolutional layers in basic block

    Attributes
    ----------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block 
    stride: int
        stride used in convolutional layers in basic block
    conv1: torch.nn.Conv2d
        first convolutional layer in basic block
    bn1: torch.nn.BatchNorm2d
        first batch normalization layer in basic block
    conv2: torch.nn.Conv2d
        second convolutional layer in basic block
    bn2: torch.nn.BatchNorm2d
        second convolutional layer in basic block

    Methods
    -------
    forward([x]): torch.Tensor
        forward process of basic block
    """

    def __init__(self, in_channels_of_basic_block, out_channels_of_basic_block, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = in_channels_of_basic_block,
            kernel_size = (3, 3), stride = (stride, stride), padding = (1, 1), groups = in_channels_of_basic_block, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels_of_basic_block)
        self.conv2 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels_of_basic_block)

    def forward(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of basic block in mobile network.

        Parameters
        ----------
        x: torch.Tensor
            input of basic block

        Returns
        -------
        y: torch.Tensor
            output of basic block
        """

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace = True)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y, inplace = True)

        return y


class MobileNet(nn.Module):
    def __init__(self, number_of_classes, ca):
        super(MobileNet, self).__init__()
        self.number_of_classes = number_of_classes
        self.ca = ca
        self.cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        for i in range(0, len(self.cfg)):
            if isinstance(self.cfg[i], int):
                self.cfg[i] = int(self.cfg[i] * ca)
            else:
                self.cfg[i] = (int(self.cfg[i][0] * ca), self.cfg[i][1])
        
        self.frist_channel = int(32 * ca)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.frist_channel, kernel_size = (3, 3), stride = (2, 2),
            padding = (1, 1), bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = self.frist_channel)
        self.layers = self.generate_layers(in_planes = self.frist_channel)
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc = nn.Linear(in_features = self.cfg[12], out_features = number_of_classes)

    def generate_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(BasicBlock(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x, flag = 0):
        if flag == 0:
            y = self.conv1(x)
            y = self.bn1(y)
            y = F.relu(y, inplace = True)
            y = self.layers(y)
            y = self.pool(y)
            y = y.view(y.size()[0], -1)
            y = self.fc(y)

            return y
        elif flag == 1:
            y = self.conv1(x)
            y = self.bn1(y)
            y = F.relu(y, inplace = True)
            y = self.layers(y)
            y = self.pool(y)
            y = y.view(y.size()[0], -1)

            return y