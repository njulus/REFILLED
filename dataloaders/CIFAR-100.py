# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 15:42:03
"""

import pickle
from PIL import Image

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path, flag_mode, flag_tuning):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.flag_mode = flag_mode
        self.flag_tuning = flag_tuning

        self.features, self.labels = self.read_data()

        self.transform_augment = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        self.transform_simple = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    
    def read_data(self):
        if self.flag_tuning:
            if self.flag_mode == 'train':
                data_file_path = self.data_path + 'train'
            elif self.flag_mode == 'val':
                data_file_path = self.data_path + 'val'
            elif self.flag_mode == 'test':
                data_file_path = self.data_path + 'test'
        else:
            if self.flag_mode == 'train':
                data_file_path = self.data_path + 'train_and_val'
            elif self.flag_mode == 'val':
                data_file_path = self.data_path + 'test'
            elif self.flag_mode == 'test':
                data_file_path = self.data_path + 'test'
        with open(data_file_path, 'rb') as fp:
            data = pickle.load(fp, encoding='bytes')
        features = data[b'data']
        labels = data[b'fine_labels']

        return features, labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        feature = self.features[index, :]
        # reshape feature to the shape of (height, width, depth)
        feature_r = feature[:1024].reshape(32, 32)
        feature_g = feature[1024:2048].reshape(32, 32)
        feature_b = feature[2048:].reshape(32, 32)
        feature = np.dstack((feature_r, feature_g, feature_b))
        image = Image.fromarray(feature)
        # data preprocess
        if self.flag_mode == 'train':
            image = self.transform_augment(image)
        else:
            image = self.transform_simple(image)
        label = self.labels[index]
        return image, label

    def get_n_classes(self):
        return max(self.labels) + 1



def generate_data_loader(data_path, flag_mode, flag_tuning, batch_size, n_workers):
    my_dataset = MyDataset(data_path, flag_mode, flag_tuning)
    my_data_loader = DataLoader(my_dataset, batch_size, shuffle=True, num_workers=n_workers)
    return my_data_loader



# debug test
if __name__ == '__main__':
    data_path = '../datasets/CIFAR-100/'
    flag_mode = 'train'
    batch_size = 2
    n_workers = 0

    my_data_loader = generate_data_loader(data_path, flag_mode, batch_size, n_workers)
    for batch_index, batch in enumerate(my_data_loader):
        image, label = batch
        print(image.size())
        print(label.size())
        break

    print(my_data_loader.dataset.get_n_classes())