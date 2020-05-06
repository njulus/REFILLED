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


class LambdaLayer(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements lambda layer which completes a specified computing
    process according to a given function.

    Variables
    ---------
    f: function
        a function defining how to calculate output

    Attributes
    ----------
    f: function
        a function defining how to calculate output

    Methods
    -------
    forward([x]): torch.Tensor
        forward process of LambdaLayer
    """

    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f
    
    def forward(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of LambdaLayer.

        Parameters
        ----------
        x: torch.Tensor
            input of LambdaLayer
        
        Returns
        -------
        y: torch.Tensor
            output of LambdaLayer
        """

        y = self.f(x)
        return y


class BasicBlock(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements basic block in residual network.

    Variables
    ---------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block
    stride: int
        stride used by convolutional layer of basic block
    
    Attributes
    ----------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block
    stride: int
        stride used by convolutional layer of basic block
    conv1: torch.nn.Conv2d
        first convolutional layer
    bn1: torch.nn.BatchNorm2d
        first batch normalization layer
    conv2: torch.nn.Conv2d
        second convolutional layer
    bn2: torch.nn.BatchNorm2d
        second batch normalization layer
    shortcut: torch.nn.Sequential
        shortcut in basic block
    
    Methods
    -------
    forward([x]): torch.autograd.Variable
        forward process of basic block
    """

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
        """
        Introduction of method
        ----------------------
        This method implements forward process of BasicBlock.

        Parameters
        ----------
        x: torch.Tensor
            input of BasicBlock
        
        Returns
        -------
        y: torch.Tensor
            output of BasicBlock
        """

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.shortcut(x)
        y = F.relu(y)

        return y


class ResNet(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements residual network.

    Variables
    ---------
    depth: int
        total number of simple layers in wide residual network
    number_of_classes: int
        number of classes in a classification task
    
    Attributes
    ----------
    depth: int
        total number of simple layers in wide residual network
    number_of_classes: int
        number of classes in a classification task
    conv1: torch.nn.Conv2d
        first convolutional layers in wide residual network
    bn1: torch.nn.BatchNorm2d
        batch normalization layer
    layer1: torch.nn.Sequential
        first layer composed of several basic blocks
    layer2: torch.nn.Sequential
        second layer composed of several basic blocks
    layer3: torch.nn.Sequential
        third layer composed of several basic blocks
    fc: torch.nn.Linear
        full connected(linear) layer
    
    Methods
    -------
    generate_layer([in_channels_of_layer, out_channels_of_layer,
        number_of_blocks, stride_of_first_block]): torch.nn.Sequential
        generate a whole layer composed of several basic blocks and some parameters defining
        this layer and basic blocks are given to the method
    forward([x]): torch.Tensor
        forward process of residual network
    forward_embedding([x]): torch.Tensor
        forward process of residual network in embedding
    """

    def __init__(self, depth, number_of_classes):
        super(ResNet, self).__init__()
        self.depth = depth
        self.number_of_classes = number_of_classes

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
        """ 
        Introduction of method
        ----------------------
        This method generates a whole layer using basic blocks.

        Parameters
        ----------
        in_channels_of_layer: int
            number of input channels of layer
        out_channels_of_layer: int
            number of output channels of layer
        number_of_blocks: int
            number of basic blocks in a single layer
        stride_of_first_block: int
            stride used by first basic block in this layer, stride of other basic blocks is 1
        
        Returns
        -------
        layer: torch.nn.Sequential
            a whole layer generated using basic blocks
        """

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
    
    def forward(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of residual network.

        Parameters
        ----------
        x: torch.Tensor
            input of residual network
        
        Returns
        -------
        y: torch.Tensor
            output of residual network
        """

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = F.avg_pool2d(y, y.size()[3])
        y = y.view(y.size()[0], -1)
        y = self.fc(y)

        return y
    
    def forward_embedding(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of residual network used in embedding.

        Parameters
        ----------
        x: torch.Tensor
            input of residual network
        
        Returns
        -------
        y: torch.Tensor
            output of residual network in embedding
        """

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = F.avg_pool2d(y, y.size()[3])
        y = y.view(y.size()[0], -1)
        
        return y