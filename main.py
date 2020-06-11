# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-08-14 15:33:09
"""

import argparse
import pickle
import importlib
import random
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch

from torchvision import transforms

from networks import wide_resnet
from networks import resnet
from networks import mobile_net
import Train
import Test
from utils import evaluate_embedding


def lr_lambda_in_stage1(epoch):
    """
    Introduction of function
    ------------------------
    This function acts as a parameter passed to Train.do_train_process().
    This function controls the change of learning rate in stage1.

    Parameters
    ----------
    epoch: int
        current epoch
    
    Returns
    -------
    efficient: float
        efficient controlling the change of learning rate
    """

    efficient = 1
    return efficient


def lr_lambda_in_stage2(epoch):
    """
    Introduction of function
    ------------------------
    This function acts as a parameter passed to Train.do_train_process().
    This function controls the change of learning rate in stage2.

    Parameters
    ----------
    epoch: int
        current epoch
    
    Returns
    -------
    efficient: float
        efficient controlling the change of learning rate
    """

    if epoch >= 0 and epoch < 100:
        efficient = 1
    elif epoch >= 100 and epoch < 140:
        efficient = 0.1
    else:
        efficient = 0.01
    return efficient



if __name__ == "__main__":    
    # set gpu
    devices = [0, 1]

    # create a parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default = 'CIFAR100')
    parser.add_argument('--model_name', default = 'WideResNet')
    parser.add_argument('--depth', default = 16)
    parser.add_argument('--width', default = 1)
    parser.add_argument('--ca', default = 0.25)
    parser.add_argument('--number_of_classes', default = 100)
    parser.add_argument('--dropout_rate', default = 0.3)
    parser.add_argument('--train_batch_size', default = 512)
    parser.add_argument('--validate_batch_size', default = 128)
    parser.add_argument('--test_batch_size', default = 128)
    parser.add_argument('--learning_rate_in_stage1', default = 0.1)
    parser.add_argument('--learning_rate_in_stage2', default = 0.1)
    parser.add_argument('--momentum', default = 0.9)
    parser.add_argument('--weight_decay', default = 0.0005)
    parser.add_argument('--nesterov', default = True)
    parser.add_argument('--number_of_epochs_in_stage1', default = 2)
    parser.add_argument('--number_of_epochs_in_stage2', default = 2)
    parser.add_argument('--flag_gpu', default = True)
    parser.add_argument('--model_path_stage1', default = 'saves/trained_models/ReFilled_stage1/')
    parser.add_argument('--model_path_stage2', default = 'saves/trained_models/ReFilled_stage2/')
    parser.add_argument('--result_path_stage1', default = 'saves/results/ReFilled_stage1/')
    parser.add_argument('--result_path_stage2', default = 'saves/results/ReFilled_stage2/')
    parser.add_argument('--teacher_model_file_path',
        default = 'saves/trained_models/single_depth=40_width=2_dropout=0.3_batch=128_lr=0.1_mmt=0.9_wd=0.0005')
    parser.add_argument('--beta', default = 1000)
    parser.add_argument('--tau1', default = 4)
    parser.add_argument('--tau2', default = 1)
    parser.add_argument('--tau3', default = 2)
    parser.add_argument('--debug', default = False)
    parser.add_argument('--visualize', default = False)
    
    # get hyperparameters
    args = parser.parse_args()
    data_name = args.data_name
    model_name = args.model_name
    data_path = 'datasets/' + data_name + '/'
    depth = int(args.depth)
    width = int(args.width)
    ca = float(args.ca)
    number_of_classes = int(args.number_of_classes)
    dropout_rate = float(args.dropout_rate)
    train_batch_size = int(args.train_batch_size)
    validate_batch_size = int(args.validate_batch_size)
    test_batch_size = int(args.test_batch_size)
    learning_rate_in_stage1 = float(args.learning_rate_in_stage1)
    learning_rate_in_stage2 = float(args.learning_rate_in_stage2)
    momentum = float(args.momentum)
    weight_decay = float(args.weight_decay)
    nesterov = bool(args.nesterov)
    number_of_epochs_in_stage1 = int(args.number_of_epochs_in_stage1)
    number_of_epochs_in_stage2 = int(args.number_of_epochs_in_stage2)
    flag_gpu = bool(args.flag_gpu)
    model_path_stage1 = args.model_path_stage1
    model_path_stage2 = args.model_path_stage2
    result_path_stage1 = args.result_path_stage1
    result_path_stage2 = args.result_path_stage2
    teacher_model_file_path = args.teacher_model_file_path
    beta = float(args.beta)
    tau1 = float(args.tau1)
    tau2 = float(args.tau2)
    tau3 = float(args.tau3)
    debug = bool(args.debug)
    visualize = bool(args.visualize)
    if data_name == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        validate_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    elif data_name == 'CUB200':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        validate_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    lr_lambda_in_stage1 = lr_lambda_in_stage1
    lr_lambda_in_stage2 = lr_lambda_in_stage2

    # print some information
    print('data_name = %s' % (data_name))
    print('model_name = %s' % (model_name))
    print('depth = %d' % (depth))
    print('width = %d' % (width))
    print('number_of_classes = %d' % (number_of_classes))
    print('dropout_rate = %f' % (dropout_rate))
    print('learning_rate_in_stage1 = %f' % (learning_rate_in_stage1))
    print('learning_rate_in_stage2 = %f' % (learning_rate_in_stage2))
    print('momentum = %f' % (momentum))
    print('weight_decay = %f' % (weight_decay))
    print('number_of_epochs_in_stage1 = %d' % (number_of_epochs_in_stage1))
    print('number_of_epochs_in_stage2 = %d' % (number_of_epochs_in_stage2))
    print('teacher_model_file_path = %s' % (teacher_model_file_path))
    print('beta = %f' % (beta))
    print('tau1 = %f' % (tau1))
    print('tau2 = %f' % (tau2))
    print('tau3 = %f' % (tau3))
    print('flag_gpu = %r' % (flag_gpu))
    print('debug = %r' % (debug))
    print('visualize = %r' % (visualize))

    # generate train_dataset, validate_dataset and test_dataset
    train_data_file_path = data_path + 'train_all'
    validate_data_file_path = data_path + 'validate'
    test_data_file_path = data_path + 'test'

    Data = importlib.import_module('dataloaders.Data_' + data_name)
    train_dataset = Data.MyDataset(data_file_path = train_data_file_path, transform = train_transform)
    validate_dataset = Data.MyDataset(data_file_path = validate_data_file_path, transform = validate_transform)
    test_dataset = Data.MyDataset(data_file_path = test_data_file_path, transform = test_transform)
    
    # generate teacher
    if model_name == 'WideResNet':
        teacher = wide_resnet.WideResNet(depth = 40, width = 2, number_of_classes = number_of_classes,
            dropout_rate = 0.3)
        if flag_gpu:
            if len(devices) != 1:
                teacher = torch.nn.DataParallel(teacher, device_ids = devices)
            teacher.load_state_dict(torch.load(teacher_model_file_path))
            teacher = teacher.cuda(devices[0])
        else:
            teacher.load_state_dict(torch.load(teacher_model_file_path, map_location = 'cpu'))
    elif model_name == 'ResNet':
        teacher = resnet.ResNet(depth = 110, number_of_classes = number_of_classes)
        if flag_gpu:
            if len(devices) != 1:
                teacher = torch.nn.DataParallel(teacher, device_ids = devices)
            teacher.load_state_dict(torch.load(teacher_model_file_path))
            teacher = teacher.cuda(devices[0])
        else:
            teacher.load_state_dict(torch.load(teacher_model_file_path, map_location = 'cpu'))
    elif model_name == 'MobileNet':
        # teacher = resnet.ResNet(depth = 110, number_of_classes = number_of_classes)
        teacher = mobile_net.MobileNet(number_of_classes = number_of_classes, ca = 1)
        if flag_gpu:
            if len(devices) != 1:
                teacher = torch.nn.DataParallel(teacher, device_ids = devices)
            teacher.load_state_dict(torch.load(teacher_model_file_path))
            teacher = teacher.cuda(devices[0])
        else:
            teacher.load_state_dict(torch.load(teacher_model_file_path, map_location = 'cpu'))
    # set teacher to evaluate mode
    teacher.eval()

    # generate model
    if model_name == 'WideResNet':
        model = wide_resnet.WideResNet(depth = depth, width = width, number_of_classes = number_of_classes,
            dropout_rate = dropout_rate)
        model.apply(wide_resnet.conv_init)
        if flag_gpu:
            if len(devices) != 1:
                model = torch.nn.DataParallel(model, device_ids = devices)
            model = model.cuda(devices[0])
    elif model_name == 'ResNet':
        if depth == 18:
            model = models.resnet18(num_classes = number_of_classes)
        elif depth == 34:
            model = models.resnet34(num_classes = number_of_classes)
        elif depth == 50:
            model = models.resnet50(num_classes = number_of_classes)
        elif depth == 101:
            model = models.resnet101(num_classes = number_of_classes)
        else:
            model = resnet.ResNet(depth = depth, number_of_classes = number_of_classes)
        if flag_gpu:
            if len(devices) != 1:
                model = torch.nn.DataParallel(model, device_ids = devices)
            model = model.cuda(devices[0])
    elif model_name == 'MobileNet':
        model = mobile_net.MobileNet(number_of_classes = number_of_classes, ca = ca)
        model.apply(mobile_net.conv_init)
        if flag_gpu:
            if len(devices) != 1:
                model = torch.nn.DataParallel(model, device_ids = devices)
            model = model.cuda(devices[0])

    # check teacher NMI
    teacher_nmi = evaluate_embedding.do_cluster(dataset = validate_dataset,
        batch_size = validate_batch_size, model = teacher, flag_gpu = flag_gpu, number_of_classes = number_of_classes, devices = devices)
    print('teacher nmi: %f' % (teacher_nmi))

    # train process
    if model_name == 'WideResNet':
        model_file_path_in_stage1 = model_path_stage1 + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_width=' + str(width) + \
            '_dropout=' + str(dropout_rate) + \
            '_batch=' + str(train_batch_size) + \
            '_lr1=' + str(learning_rate_in_stage1) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_tau1=' + str(tau1)
        model_file_path_in_stage2 = model_path_stage2 + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_width=' + str(width) + \
            '_dropout=' + str(dropout_rate) + \
            '_batch=' + str(train_batch_size) + \
            '_lr2=' + str(learning_rate_in_stage2) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_beta=' + str(beta) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
        log_dir = 'saves/runs/' + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_width=' + str(width) + \
            '_dropout=' + str(dropout_rate) + \
            '_beta=' + str(beta) + \
            '_tau1=' + str(tau1) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
    elif model_name == 'ResNet':
        model_file_path_in_stage1 = model_path_stage1 + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_batch=' + str(train_batch_size) + \
            '_lr1=' + str(learning_rate_in_stage1) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_tau1=' + str(tau1)
        model_file_path_in_stage2 = model_path_stage2 + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_batch=' + str(train_batch_size) + \
            '_lr2=' + str(learning_rate_in_stage2) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_beta=' + str(beta) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
        log_dir = 'saves/runs/' + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_beta=' + str(beta) + \
            '_tau1=' + str(tau1) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
    elif model_name == 'MobileNet':
        model_file_path_in_stage1 = model_path_stage1 + 'ReFilled' + \
            '_ca=' + str(ca) + \
            '_batch=' + str(train_batch_size) + \
            '_lr1=' + str(learning_rate_in_stage1) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_tau1=' + str(tau1)
        model_file_path_in_stage2 = model_path_stage2 + 'ReFilled' + \
            '_ca=' + str(ca) + \
            '_batch=' + str(train_batch_size) + \
            '_lr2=' + str(learning_rate_in_stage2) + \
            '_mmt=' + str(momentum) + \
            '_wd=' + str(weight_decay) + \
            '_beta=' + str(beta) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
        log_dir = 'saves/runs/' + 'ReFilled' + \
            '_depth=' + str(depth) + \
            '_beta=' + str(beta) + \
            '_tau1=' + str(tau1) + \
            '_tau2=' + str(tau2) + \
            '_tau3=' + str(tau3)
    guiding_loss_after_each_epoch, guiding_nmi_after_each_epoch, \
        training_loss_after_each_epoch, teaching_loss_after_each_epoch, \
            training_accuracy_after_each_epoch, validating_accuracy_after_each_epoch = \
                Train.do_train_process(train_dataset = train_dataset, validate_dataset = validate_dataset,
                    train_batch_size = train_batch_size, validate_batch_size = validate_batch_size, model = model,
                    learning_rate_in_stage1 = learning_rate_in_stage1, learning_rate_in_stage2 = learning_rate_in_stage2,
                    momentum = momentum, weight_decay = weight_decay, nesterov = nesterov, lr_lambda_in_stage1 = lr_lambda_in_stage1,
                    lr_lambda_in_stage2 = lr_lambda_in_stage2, number_of_epochs_in_stage1 = number_of_epochs_in_stage1,
                    number_of_epochs_in_stage2 = number_of_epochs_in_stage2, flag_gpu = flag_gpu,
                    model_file_path_in_stage1 = model_file_path_in_stage1, model_file_path_in_stage2 = model_file_path_in_stage2,
                    teacher = teacher, beta = beta, tau1 = tau1, tau2 = tau2, tau3 = tau3, debug = debug,
                    visualize = visualize, log_dir = log_dir, devices = devices, number_of_classes = number_of_classes)
    
    # save results if not under debug mode for stage1
    if not debug:
        if model_name == 'WideResNet':
            result_dir_in_stage1 = result_path_stage1 + 'ReFilled' + \
                '_depth=' + str(depth) + \
                '_width=' + str(width) + \
                '_dropout=' + str(dropout_rate) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage1) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_tau1=' + str(tau1)
        elif model_name == 'ResNet':
            result_dir_in_stage1 = result_path_stage1 + 'ReFilled' + \
                '_depth=' + str(depth) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage1) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_tau1=' + str(tau1)
        elif model_name == 'MobileNet':
            result_dir_in_stage1 = result_path_stage1 + 'ReFilled' + \
                '_ca=' + str(ca) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage1) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_tau1=' + str(tau1)
        if not os.path.exists(result_dir_in_stage1):
            os.mkdir(result_dir_in_stage1)
        guiding_loss_after_each_epoch = np.array(guiding_loss_after_each_epoch)
        guiding_nmi_after_each_epoch = np.array(guiding_nmi_after_each_epoch)
        np.save(result_dir_in_stage1 + '/guiding_loss_after_each_epoch.npy', guiding_loss_after_each_epoch)
        np.save(result_dir_in_stage1 + '/guiding_nmi_after_each_epoch.npy', guiding_nmi_after_each_epoch)

    # save results if not under debug mode for stage2
    if not debug:
        if model_name == 'WideResNet':
            result_dir_in_stage2 = result_path_stage2 + 'ReFilled' + \
                '_depth=' + str(depth) + \
                '_width=' + str(width) + \
                '_dropout=' + str(dropout_rate) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage2) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_beta=' + str(beta) + \
                '_tau2=' + str(tau2) + \
                '_tau3=' + str(tau3)
        elif model_name == 'ResNet':
            result_dir_in_stage2 = result_path_stage2 + 'ReFilled' + \
                '_depth=' + str(depth) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage2) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_beta=' + str(beta) + \
                '_tau2=' + str(tau2) + \
                '_tau3=' + str(tau3)
        elif model_name == 'MobileNet':
            result_dir_in_stage2 = result_path_stage2 + 'ReFilled' + \
                '_ca=' + str(ca) + \
                '_batch=' + str(train_batch_size) + \
                '_lr=' + str(learning_rate_in_stage2) + \
                '_mmt=' + str(momentum) + \
                '_wd=' + str(weight_decay) + \
                '_beta=' + str(beta) + \
                '_tau2=' + str(tau2) + \
                '_tau3=' + str(tau3)
        if not os.path.exists(result_dir_in_stage2):
            os.mkdir(result_dir_in_stage2)
        training_loss_after_each_epoch = np.array(training_loss_after_each_epoch)
        teaching_loss_after_each_epoch = np.array(teaching_loss_after_each_epoch)
        training_accuracy_after_each_epoch = np.array(training_accuracy_after_each_epoch)
        validating_accuracy_after_each_epoch = np.array(validating_accuracy_after_each_epoch)
        np.save(result_dir_in_stage2 + '/training_loss_after_each_epoch.npy', training_loss_after_each_epoch)
        np.save(result_dir_in_stage2 + '/teaching_loss_after_each_epoch.npy', teaching_loss_after_each_epoch)
        np.save(result_dir_in_stage2 + '/training_accuracy_after_each_epoch.npy', training_accuracy_after_each_epoch)
        np.save(result_dir_in_stage2 + '/validating_accuracy_after_each_epoch.npy', validating_accuracy_after_each_epoch)

    # do test process if not under debug mode
    if not debug:
        # load best model found
        model.load_state_dict(torch.load(model_file_path_in_stage2))

        # test process
        testing_accuracy = Test.do_test_process(dataset = test_dataset, batch_size = test_batch_size, model = model,
            flag_gpu = flag_gpu, devices = devices)
        print('testing_accuracy = %f' % (testing_accuracy))