# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-07-15 14:54:19
"""

import torch
from torch.utils.data import DataLoader

def do_test_process(dataset, batch_size, model, flag_gpu, devices):
    """
    Introduction of function
    ------------------------
    This function tests model on dataset.

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
    devices: list of int
        ids of devices
    
    Returns
    -------
    accuracy: float
        classification accuracy on model on dataset
    """

    # generate dataset_loader
    dataset_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = False, drop_last = False)

    # change model to evaluate mode
    model.eval()

    # init number of instances
    number_of_instances = dataset.__len__()
    # init accuracy
    accuracy = 0
    # loop for each mini batch in dataset_loader
    for mini_batch_index, (feature, fine_label) in enumerate(dataset_loader):
        feature = feature.float().cuda(devices[0]) if flag_gpu else feature.float()
        fine_label = fine_label.long().cuda(devices[0]) if flag_gpu else fine_label.long()

        # model forward process
        model_output = model(feature)

        # update accuracy
        predicted_label = torch.max(model_output, dim = 1)[1]
        accuracy += torch.sum(predicted_label == fine_label).data.cpu().item()
    
    accuracy /= number_of_instances
    return accuracy