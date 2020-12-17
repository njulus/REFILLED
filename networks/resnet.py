# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-07-15 15:21:44
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



class LambdaLayer(nn.Module):
    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f
    
    def forward(self, x):
        y = self.f(x)
        return y



class BasicBlock(nn.Module):
    def __init__(self, in_channels_of_basic_block, out_channels_of_basic_block, stride):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels_of_basic_block
        self.out_channels = out_channels_of_basic_block
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (3, 3), stride = (stride, stride), padding = (1, 1), bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels_of_basic_block)
        self.conv2 = nn.Conv2d(in_channels = out_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels_of_basic_block)
        
        self.shortcut = nn.Sequential()
        # size of feature map changes or number of channels changes
        if stride != 1 or in_channels_of_basic_block != out_channels_of_basic_block:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels_of_basic_block // 4,
                    out_channels_of_basic_block // 4), 'constant', 0))
            
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.shortcut(x)
        y = F.relu(y)

        return y



class MyNetwork(nn.Module):
    def __init__(self, args):
        super(MyNetwork, self).__init__()
        self.args = args
        depth = args.depth
        number_of_classes = args.number_of_classes

        # depth must be of form (6n + 2)
        # number of convolutional layers in a basic block = 2
        # number of layers in a wide residual network = 3
        # number of blocks in each layer = n
        # number of other simple layers = 2
        assert((depth - 2) % 6 == 0)
        # calculate number of blocks in each layer
        number_of_blocks_in_each_layer = int((depth - 2) / 6)
        # define number of channels after each block
        number_of_channels_after_each_layer = [16, 16, 32, 64]

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = number_of_channels_after_each_layer[0],
            kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = number_of_channels_after_each_layer[0])
        # generate 3 layers
        self.layer1 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[0],
            out_channels_of_layer = number_of_channels_after_each_layer[1], number_of_blocks = number_of_blocks_in_each_layer,
            stride_of_first_block = 1)
        self.layer2 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[1],
            out_channels_of_layer = number_of_channels_after_each_layer[2], number_of_blocks = number_of_blocks_in_each_layer,
            stride_of_first_block = 2)
        self.layer3 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[2],
            out_channels_of_layer = number_of_channels_after_each_layer[3], number_of_blocks = number_of_blocks_in_each_layer,
            stride_of_first_block = 2)
        # generate linear layer
        self.fc = nn.Linear(in_features = number_of_channels_after_each_layer[3], out_features = number_of_classes)
    
    def generate_layer(self, in_channels_of_layer, out_channels_of_layer, number_of_blocks,
                       stride_of_first_block):
        strides_of_each_block = [stride_of_first_block] + [1] * (number_of_blocks - 1)
        blocks = []
        # generate a layer with number_of_blocks blocks
        for i in range(0, number_of_blocks):
            # generate the first basic block in this layer
            if i == 0:
                blocks.append(BasicBlock(in_channels_of_basic_block = in_channels_of_layer, out_channels_of_basic_block = out_channels_of_layer,
                    stride = strides_of_each_block[i]))
            # generate other basic blocks
            else:
                blocks.append(BasicBlock(in_channels_of_basic_block = out_channels_of_layer, out_channels_of_basic_block = out_channels_of_layer,
                    stride = strides_of_each_block[i]))
        # generate the whole layer using blocks     
        layer = nn.Sequential(*blocks)
        return layer
    
    def forward(self, x, flag_embedding=False, flag_both=False):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = F.avg_pool2d(y, y.size()[3])
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
        modules = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j

    def get_classifier_params(self):
        modules = [self.fc]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j