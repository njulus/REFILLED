# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-08-13 14:42:53
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import torch
from torch.utils.data import DataLoader


def do_cluster(dataset, batch_size, model, flag_gpu, number_of_classes, devices):
    """
    Introduction of function
    ------------------------
    This function evaluates the quality of a set of embeddings X by k-means
    clustering. Normalized Mutual Information(NMI) is used as evaluation
    metric of clustering.

    Parameters
    ----------
    dataset: torch.utils.Dataset
        dataset to do test on
    batch_size: int
        batch size used in dataset_loader
    model: torch.nn.Module
        model to test
    flag_gpu: bool
        indicate whether to use gpu or not
    number_of_classes: int
        number of different classes in this dataset
    devices: list of int
        ids of devices
    
    Returns
    -------
    NMI: float
        Normalized Mutual Information of clustering result
    """

    # generate dataset_loader
    dataset_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False, drop_last = False)

    # change model to evaluate mode
    model.eval()

    # init number of instances
    number_of_instances = dataset.__len__()
    
    # loop for each mini batch in dataset_loader
    for mini_batch_index, (feature, fine_label) in enumerate(dataset_loader):
        feature = feature.float().cuda(devices[0]) if flag_gpu else feature.float()
        fine_label = fine_label.long().cuda(devices[0]) if flag_gpu else fine_label.long()

        # model forward process
        X = model.forward(feature, 1)
        if mini_batch_index == 0:
            embeddings = X.cpu().data.numpy()
            labels = fine_label.cpu().data.numpy()
        else:
            embeddings = np.concatenate((embeddings, X.cpu().data.numpy()), axis = 0)
            labels = np.concatenate((labels, fine_label.cpu().data.numpy()), axis = 0)

    solver = KMeans(n_clusters = number_of_classes)
    predicted_labels = solver.fit_predict(embeddings)
    NMI = normalized_mutual_info_score(labels, predicted_labels)

    return NMI


def calculate_inter_class_mean(dataset, model, flag_gpu, number_of_classes):
    """
    Introduction of function
    ------------------------
    This function finds inter-class means in a given dataset.

    Parameters
    ----------
    dataset: torch.utils.Dataset
        dataset to do calculate inter-class means in
    model: torch.nn.Module
        model to test
    flag_gpu: bool
        indicate whether to use gpu or not
    number_of_classes: int
        number of different classes in this dataset

    Returns
    -------
    inter_class_meanns: torch.autograd.Variable
        inter-class means in the given dataset
    """

    # generate dataset_loader
    dataset_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False, drop_last = False)

    # change model to evaluate mode
    model.eval()

    # init inter_class_means
    inter_class_means = torch.zeros((number_of_classes, 64))
    # inter_class_means = torch.zeros((number_of_classes, 128))
    if flag_gpu:
        inter_class_means = inter_class_means.cuda()
    # init label_counts
    label_counts = torch.zeros((number_of_classes))
    if flag_gpu:
        label_counts = label_counts.cuda()

    # loop for each mini batch in dataset_loader
    for mini_batch_index, (feature, fine_label) in enumerate(dataset_loader):
        feature = feature.float().cuda() if flag_gpu else feature.float()
        fine_label = fine_label.long().cuda() if flag_gpu else fine_label.long()

        # model forward process
        embedding = model.forward_embedding(feature).squeeze(0)
        label = fine_label

        # update inter_class_means
        inter_class_means[fine_label, :] += embedding
        label_counts[fine_label] += 1
        
    # calculate final inter-class means
    inter_class_means = (inter_class_means.t() / label_counts).t()

    return inter_class_means