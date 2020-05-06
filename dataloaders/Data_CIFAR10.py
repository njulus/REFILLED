# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-07-15 16:06:46
"""

import pickle
from PIL import Image

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

class MyDataset(Dataset):
    """
    Introduction of class
    ---------------------
    This class is a subclass of torch.utils.data.Dataset. This class is 
    responsible for reading data from source files and passes data to dataloader.

    Variables
    ---------
    data_file_path: str
        a string indicating the file containing data
    transform: torchvision.transforms.transforms
        transform function used to do data preprocess
    
    Attributes
    ----------
    data_file_path: str
        a string indicating the file containing data
    transform: torchvision.transforms.transforms
        transform function used to do data preprocess
    features: numpy.ndarray of uint8
        features of data
    labels: list of int
        labels of data
    
    Methods
    -------
    __len__(): int
        calculate the length of data
    __getitem__([index]): [torch.Tensor, int, int]
        return a single instance indexed by index
    read_data_from_file([data_file_path]): [numpy.ndarray of uint8, list of int, list of int]
        read data from file indicated by data_file_path
    """

    def __init__(self, data_file_path, transform):
        super(MyDataset, self).__init__()
        self.data_file_path = data_file_path
        self.features, self.labels = self.read_data_from_file(data_file_path)
        self.transform = transform

    def __len__(self):
        """
        Introduction of method
        ----------------------
        This method calculates the length of data.

        Parameters
        ----------
        NONE

        Returns
        -------
        length: int
            length of data
        """

        length = len(self.labels)
        return length
    
    def __getitem__(self, index):
        """
        Introduction of method
        ----------------------
        This method returns a single instance in data indexed by index.

        Parameters
        ----------
        index: int
            index of the wanted instance
        
        Returns
        -------
        image: torch.Tensor
            feature of the wanted instance
        label: int
            label of the wanted instance
        """

        feature = self.features[index, :]
        # reshape feature to the shape of (height, width, depth)
        feature_r = feature[:1024].reshape(32, 32)
        feature_g = feature[1024:2048].reshape(32, 32)
        feature_b = feature[2048:].reshape(32, 32)
        feature = np.dstack((feature_r, feature_g, feature_b))
        image = Image.fromarray(feature)
        # data preprocess
        image = self.transform(image)
        label = self.labels[index]
        return image, label

    def read_data_from_file(self, data_file_path):
        """
        Introduction of method
        ----------------------
        This method reads data from file indicated by file_path.

        Parameters
        ----------
        data_file_path: str
            a string indicating the file containing data
        
        Returns
        -------
        features: numpy.ndarray of uint8
            features of data
        labels: list of int
            labels of data
        """

        with open(data_file_path, 'rb') as fp:
            data = pickle.load(fp, encoding = 'bytes')
        features = data[b'data']
        labels = data[b'labels']
        return features, labels


# debug test
if __name__ == "__main__":
    data_file_path = '../datasets/CIFAR10/train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    batch_size = 64
    my_dataset = MyDataset(data_file_path = data_file_path, transform = transform)
    dataset_loader = DataLoader(dataset = my_dataset, batch_size = batch_size)
    print(my_dataset.__len__())
    print(len(my_dataset))
    print(len(dataset_loader))