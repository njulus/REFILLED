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
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain = np.sqrt(2))
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



class BasicBlock(nn.Module):
    def __init__(self, in_channels_of_basic_block, out_channels_of_basic_block, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = in_channels_of_basic_block,
            kernel_size = (3, 3), stride = (stride, stride), padding = (1, 1), groups = in_channels_of_basic_block, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels_of_basic_block)
        self.conv2 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels_of_basic_block)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace = True)
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y, inplace = True)

        return y



class MyNetwork(nn.Module):
    def __init__(self, args):
        super(MyNetwork, self).__init__()
        self.args = args
        ca = args.ca
        number_of_classes = args.number_of_classes
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
    
    def forward(self, x, flag_embedding=False, flag_both=False):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace = True)
        y = self.layers(y)
        y = self.pool(y)
        y = y.view(y.size()[0], -1)
        if flag_embedding:
            return y
        else:
            l = self.fc(y)
            if flag_both:
                return l, y
            else:
                return l
    
    def get_network_params(self):
        modules = [self.conv1, self.bn1, self.layers, self.pool]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j

    def get_classifier_params(self):
        modules = [self.fc]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j