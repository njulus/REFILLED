# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-09 20:03:32
"""

import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import global_variable as GV

def test(args, data_loader, network, description='testing'):
    # init accuracy
    accuracy = 0

    network.eval()
    for batch_index, batch in enumerate(data_loader):
        images, labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])
        
        with torch.no_grad():
            logits = network.forward(images)
        prediction = torch.argmax(logits, dim=1)
        accuracy += torch.sum((prediction == labels).float()).cpu().item()

    accuracy /= data_loader.dataset.__len__()
    return accuracy



def test_ncm(args, data_loader, network, description='testing'):
    # init class center and class count
    n_classes = data_loader.dataset.get_n_classes()
    n_dimension = network.fc.in_features
    class_center = torch.zeros((n_classes, n_dimension)).cuda(args.devices[0])
    class_count = torch.zeros(n_classes).cuda(args.devices[0])

    network.eval()
    for batch_index, batch in enumerate(data_loader):
        images, labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])
        
        with torch.no_grad():
            embedding = network.forward(images, flag_embedding=True)
            for i in range(0, n_classes):
                index_of_class_i = (labels == i)
                class_center[i] += torch.sum(embedding[index_of_class_i], dim=0)
                class_count[i] += index_of_class_i.size()[0]
    # get class center
    class_count = class_count.unsqueeze(1)
    class_center = class_center / class_count
    class_center = F.normalize(class_center, p=2, dim=1)

    # init accuracy
    accuracy = 0

    network.eval()
    for batch_index, batch in enumerate(data_loader):
        images, labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])

        with torch.no_grad():
            embedding = network.forward(images, flag_embedding=True)
            logits = torch.mm(embedding, class_center.t())
        prediction = torch.argmax(logits, dim=1)
        accuracy += torch.sum((prediction == labels).float()).cpu().item()

    accuracy /= data_loader.dataset.__len__()
    return accuracy