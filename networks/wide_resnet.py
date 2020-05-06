# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-07-15 13:57:46
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
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements basic block in wide residual network.

    Variables
    ---------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block
    dropout_rate: float
        dropout rate used by dropout layer of basic block
    stride: int
        stride used by convolutional layer of basic block
    
    Attributes
    ----------
    in_channels_of_basic_block: int
        number of input channels of basic block
    out_channels_of_basic_block: int
        number of output channels of basic block
    dropout_rate: float
        dropout rate used by dropout layer of basic block
    stride: int
        stride used by convolutional layer of basic block
    bn1: torch.nn.BatchNorm2d
        first batch normalization layer
    conv1: torch.nn.Conv2d
        first convolutional layer
    dropout: torch.nn.Dropout
        dropout layer
    bn2: torch.nn.BatchNorm2d
        second batch normalization layer
    conv2: torch.nn.Conv2d
        second convolutional layer
    shortcut: torch.nn.Sequential
        shortcut in basic block
    
    Methods
    -------
    forward([x]): torch.autograd.Variable
        forward process of basic block
    """

    def __init__(self, in_channels_of_basic_block, out_channels_of_basic_block, dropout_rate, stride):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels_of_basic_block
        self.out_channels = out_channels_of_basic_block
        self.dropout_rate = dropout_rate
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(num_features = in_channels_of_basic_block)
        self.conv1 = nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = True)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels_of_basic_block)
        self.conv2 = nn.Conv2d(in_channels = out_channels_of_basic_block, out_channels = out_channels_of_basic_block,
            kernel_size = (3, 3), stride = (stride, stride), padding = (1, 1), bias = True)
        self.shortcut = nn.Sequential()
        # size of feature map changes or number of channels changes
        if stride != 1 or in_channels_of_basic_block != out_channels_of_basic_block:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels_of_basic_block, out_channels = out_channels_of_basic_block,
                    kernel_size = (1, 1), stride = (stride, stride), bias = True)
            )

    def forward(self, x):
        """  
        Introduction of method
        ----------------------
        This method implements forward process of basic block in wide residual resnet.

        Parameters
        ----------
        x: torch.autograd.Variable
            input of the basic block
        
        Returns
        -------
        y: torch.autograd.Variable
            output of the basic block
        """

        y = self.bn1(x)
        y = F.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y += self.shortcut(x)

        return y
    

class WideResNet(nn.Module):
    """
    Introduction of class
    ---------------------
    This class implements wide residual network.

    Variables
    ---------
    depth: int
        total number of simple layers in wide residual network
    width: int
        multiple of number of channels after each layer
    number_of_classes: int
        number of classes in a classification task
    dropout_rate: float
        dropout_rate used by dropout layers
    
    Attributes
    ----------
    depth: int
        total number of simple layers in wide residual network
    width: int
        multiple of number of channels after each layer
    number_of_classes: int
        number of classes in a classification task
    dropout_rate: float
        dropout_rate used by dropout layers
    conv1: torch.nn.Conv2d
        first convolutional layers in wide residual network
    layer1: torch.nn.Sequential
        first layer composed of several basic blocks
    layer2: torch.nn.Sequential
        second layer composed of several basic blocks
    layer3: torch.nn.Sequential
        third layer composed of several basic blocks
    bn: torch.nn.BatchNorm2d
        batch normalization layer
    pool: torch.nn.AdapativeAvgPool2d
        adaptive average pooling layer
    fc: torch.nn.Linear
        full connected(linear) layer
    
    Methods
    -------
    generate_layer([in_channels_of_layer, out_channels_of_layer,
        number_of_blocks, dropout_rate, stride_of_first_block]): torch.nn.Sequential
        generate a whole layer composed of several basic blocks and some parameters defining
        this layer and basic blocks are given to the method
    forward([x]): torch.autograd.Variabel
        forward process of wide residual network
    forward_embedding([x]): torch.autograd.Variable
        forward process of wide residual network in embedding
    """

    def __init__(self, depth, width, number_of_classes, dropout_rate):
        super(WideResNet, self).__init__()
        self.depth = depth
        self.width = width
        self.number_of_classes = number_of_classes
        self.dropout_rate = dropout_rate

        # depth must be of form (6n + 4)
        # number of convolutional layers in a basic block = 2
        # number of layers in a wide residual network = 3
        # number of blocks in each layer = n
        # number of other simple layers = 4
        assert((depth - 4) % 6 == 0)
        # calculate number of blocks in each layer
        number_of_blocks_in_each_layer = int((depth - 4) / 6)
        # define number of channels after each block
        number_of_channels_after_each_layer = [16, 16 * width, 32 * width, 64 * width]

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3, 3),
            stride = (1, 1), padding = (1, 1), bias = True)
        # generate 3 layers
        self.layer1 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[0],
            out_channels_of_layer = number_of_channels_after_each_layer[1], number_of_blocks = number_of_blocks_in_each_layer,
            dropout_rate = dropout_rate, stride_of_first_block = 1)
        self.layer2 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[1],
            out_channels_of_layer = number_of_channels_after_each_layer[2], number_of_blocks = number_of_blocks_in_each_layer,
            dropout_rate = dropout_rate, stride_of_first_block = 2)
        self.layer3 = self.generate_layer(in_channels_of_layer = number_of_channels_after_each_layer[2],
            out_channels_of_layer = number_of_channels_after_each_layer[3], number_of_blocks = number_of_blocks_in_each_layer,
            dropout_rate = dropout_rate, stride_of_first_block = 2)
        # generate batch normalization layer
        self.bn = nn.BatchNorm2d(number_of_channels_after_each_layer[3], momentum = 0.9)
        # generate pooling layer
        self.pool = nn.AdaptiveAvgPool2d(output_size = 1)
        # generate linear layer
        self.fc = nn.Linear(in_features = number_of_channels_after_each_layer[3], out_features = number_of_classes)
        
    def generate_layer(self, in_channels_of_layer, out_channels_of_layer, number_of_blocks,
                       dropout_rate, stride_of_first_block):
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
        dropout_rate: float
            dropout rate used by basic blocks in this layer
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
                    dropout_rate = dropout_rate, stride = strides_of_each_block[i]))
            # generate other basic blocks
            else:
                blocks.append(BasicBlock(in_channels_of_basic_block = out_channels_of_layer, out_channels_of_basic_block = out_channels_of_layer,
                    dropout_rate = dropout_rate, stride = strides_of_each_block[i]))
        # generate the whole layer using blocks     
        layer = nn.Sequential(*blocks)
        return layer

    def forward(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of wide residual network.

        Parameters
        ----------
        x: torch.autograd.Variable
            input of wide residual network
        
        Returns
        -------
        y: torch.autograd.Variable
            output of wide residual network
        """

        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.bn(y)
        y = F.relu(y)
        y = self.pool(y)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)

        return y

    def forward_embedding(self, x):
        """
        Introduction of method
        ----------------------
        This method implements forward process of wide residual network used in embedding.

        Parameters
        ----------
        x: torch.autograd.Variable
            input of wide residual network
        
        Returns
        -------
        y: torch.autograd.Variable
            output of wide residual network in embedding
        """

        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.bn(y)
        y = F.relu(y)
        y = F.avg_pool2d(y, 8)
        y = y.view(y.size()[0], -1)
        
        return y