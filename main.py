# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-09 20:42:47
"""

import argparse
import random
import importlib
import platform
import copy

import numpy as np

import torch
from torch import nn
from torchvision import models

from networks import resnet, wide_resnet, mobile_net

from Train import train_stage1
from Train import train_stage2
from Test import test
from utils import global_variable as GV
import os

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('teacher_network_name = %s' % (args.teacher_network_name))
    print('student_network_name = %s' % (args.student_network_name))
    print('===== experiment environment arguments =====')
    print('devices = %s' % (str(args.devices)))
    print('flag_debug = %r' % (args.flag_debug))
    print('flag_no_bar = %r' % (args.flag_no_bar))
    print('n_workers = %d' % (args.n_workers))
    print('flag_tuning = %r' % (args.flag_tuning))
    print('===== optimizer arguments =====')
    print('lr1 = %f' % (args.lr1))
    print('lr2 = %f' % (args.lr2))
    print('point = %s' % str((args.point)))
    print('gamma = %f' % (args.gamma))
    print('weight_decay = %f' % (args.wd))
    print('momentum = %f' % (args.mo))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('width = %d' % (args.width))
    print('ca = %f' % (args.ca))
    print('dropout_rate = %d' % (args.dropout_rate))
    print('===== training procedure arguments =====')
    print('n_training_epochs1 = %d' % (args.n_training_epochs1))
    print('n_training_epochs2 = %d' % (args.n_training_epochs2))
    print('batch_size = %d' % (args.batch_size))
    print('flag_merge = %r' % (args.flag_merge))
    print('tau1 = %f' % (args.tau1))
    print('tau2 = %f' % (args.tau2))
    print('lambd = %f' % (args.lambd))



if __name__ == '__main__':
    # set random seed
    random.seed(960402)
    np.random.seed(960402)
    torch.manual_seed(960402)
    torch.cuda.manual_seed(960402)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()
    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100', 'CUB-200'])
    parser.add_argument('--teacher_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    parser.add_argument('--student_network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--flag_no_bar', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    parser.add_argument('--flag_tuning', action='store_true', default=False)
    # optimizer arguments
    parser.add_argument('--lr1', type=float, default=0.1)
    parser.add_argument('--lr2', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=(100,140,180))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--ca', type=float, default=0.25)  # channel
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--n_training_epochs1', type=int, default=200)
    parser.add_argument('--n_training_epochs2', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)  # training batch size
    parser.add_argument('--flag_merge', action='store_true', default=False)
    parser.add_argument('--tau1', type=float, default=4) # temperature for stochastic triplet embedding in stage 1
    parser.add_argument('--tau2', type=float, default=2) # temperature for local distillation in stage 2
    parser.add_argument('--lambd', type=float, default=100) # weight of teaching loss in stage 2

    args = parser.parse_args()

    display_args(args)

    data_path = 'datasets/' + args.data_name + '/'

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.student_network_name)

    # generate data_loader
    train_data_loader = Data.generate_data_loader(data_path, 'train', args.flag_tuning, args.batch_size, args.n_workers)
    args.number_of_classes = train_data_loader.dataset.get_n_classes()
    print('===== train data loader ready. =====')
    validate_data_loader = Data.generate_data_loader(data_path, 'val', args.flag_tuning, args.batch_size, args.n_workers)
    print('===== validate data loader ready. =====')
    test_data_loader = Data.generate_data_loader(data_path, 'test', args.flag_tuning, args.batch_size, args.n_workers)
    print('===== test data loader ready. =====')

    # generate teacher network
    if args.teacher_network_name == 'resnet':
        teacher_args = copy.copy(args)
        teacher_args.depth = 110
        teacher = resnet.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_resnet_teacher.model'
    elif args.teacher_network_name == 'wide_resnet':
        teacher_args = copy.copy(args)
        teacher_args.depth, teacher_args.width = 40, 2
        teacher = wide_resnet.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_wide_resnet_teacher.model'
    elif args.teacher_network_name == 'mobile_net':
        teacher_args = copy.copy(args)
        teacher_args.ca = 1.0
        teacher = mobile_net.MyNetwork(teacher_args)
        pretrained_teacher_save_path = 'saves/pretrained_teachers/' + args.data_name + '_mobile_net_teacher.model'
    record = torch.load(pretrained_teacher_save_path, map_location='cpu')
    teacher.load_state_dict(record['state_dict'])
    teacher = teacher.cuda(args.devices[0])
    if len(args.devices) > 1:
        teacher = torch.nn.DataParallel(teacher, device_ids=args.devices)
    # set teacher to evaluation mode
    teacher.eval()
    print('===== teacher ready. =====')

    # generate student network
    student = Network.MyNetwork(args)
    student = student.cuda(args.devices[0])
    if len(args.devices) > 1:
        student = torch.nn.DataParallel(student, device_ids=args.devices)
    print('===== student ready. =====')

    # model save path and statistics save path for stage 1
    model_save_path1 = 'saves/trained_students/' + \
                        args.data_name + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                        '_lr1=' + str(args.lr1) + \
                        '_wd=' + str(args.wd) + \
                        '_mo=' + str(args.mo) + \
                        '_depth=' + str(args.depth) + \
                        '_width=' + str(args.width) + \
                        '_ca=' + str(args.ca) + \
                        '_dropout=' + str(args.dropout_rate) + \
                        '_batch=' + str(args.batch_size) + \
                        '_tau1=' + str(args.tau1) + \
                        '.model'
    statistics_save_path1 = 'saves/student_statistics/' + \
                            args.data_name + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                            '_lr1=' + str(args.lr1) + \
                            '_wd=' + str(args.wd) + \
                            '_mo=' + str(args.mo) + \
                            '_depth=' + str(args.depth) + \
                            '_width=' + str(args.width) + \
                            '_ca=' + str(args.ca) + \
                            '_dropout=' + str(args.dropout_rate) + \
                            '_batch=' + str(args.batch_size) + \
                            '_tau1=' + str(args.tau1) + \
                            '.stat'

    # create model directories
    dirs = os.path.dirname(model_save_path1)
    os.makedirs(dirs, exist_ok=True)

    # model training stage 1
    training_loss_list1, validating_accuracy_list1 = \
        train_stage1(args, train_data_loader, validate_data_loader, teacher, student, model_save_path1)
    record = {
        'training_loss1': training_loss_list1,
        'validating_accuracy1': validating_accuracy_list1
    }

    # create stats directories
    dirs = os.path.dirname(statistics_save_path1)
    os.makedirs(dirs, exist_ok=True)
    if args.n_training_epochs1 > 0 and (not args.flag_debug):
        torch.save(record, statistics_save_path1)
    print('===== training stage 1 finish. =====')

    # load best model found in stage 1
    if not args.flag_debug:
        record = torch.load(model_save_path1)
        best_validating_accuracy = record['validating_accuracy']
        student.load_state_dict(record['state_dict'])
        print('===== best model in stage 1 loaded, validating acc = %f. =====' % (record['validating_accuracy']))

    # model save path and statistics save path for stage 2
    model_save_path2 = 'saves/trained_students/' + \
                        args.data_name + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                        '_lr2=' + str(args.lr2) + \
                        '_point=' + str(args.point) + \
                        '_gamma=' + str(args.gamma) + \
                        '_wd=' + str(args.wd) + \
                        '_mo=' + str(args.mo) + \
                        '_depth=' + str(args.depth) + \
                        '_width=' + str(args.width) + \
                        '_ca=' + str(args.ca) + \
                        '_dropout=' + str(args.dropout_rate) + \
                        '_batch=' + str(args.batch_size) + \
                        '_tau2=' + str(args.tau2) + \
                        '_lambd=' + str(args.lambd) + \
                        '.model'
    statistics_save_path2 = 'saves/student_statistics/' + \
                            args.data_name + '_' + args.student_network_name + '_' + args.teacher_network_name + \
                            '_lr2=' + str(args.lr2) + \
                            '_point=' + str(args.point) + \
                            '_gamma=' + str(args.gamma) + \
                            '_wd=' + str(args.wd) + \
                            '_mo=' + str(args.mo) + \
                            '_depth=' + str(args.depth) + \
                            '_width=' + str(args.width) + \
                            '_ca=' + str(args.ca) + \
                            '_dropout=' + str(args.dropout_rate) + \
                            '_batch=' + str(args.batch_size) + \
                            '_tau2=' + str(args.tau2) + \
                            '_lambd=' + str(args.lambd) + \
                            '.stat'

    # model training stage 2
    training_loss_list2, teaching_loss_list2, training_accuracy_list2, validating_accuracy_list2 = \
        train_stage2(args, train_data_loader, validate_data_loader, teacher, student, model_save_path2)
    record = {
        'training_loss2': training_loss_list2,
        'teaching_loss2': teaching_loss_list2,
        'training_accuracy2': training_accuracy_list2,
        'validating_accuracy2': validating_accuracy_list2
    }

    # create stats directories
    dirs = os.path.dirname(statistics_save_path2)
    os.makedirs(dirs, exist_ok=True)
    if args.n_training_epochs2 > 0 and (not args.flag_debug):
        torch.save(record, statistics_save_path2)
    print('===== training stage 2 finish. =====')

    # load best model found in stage 2
    if not args.flag_debug:
        record = torch.load(model_save_path2)
        best_validating_accuracy = record['validating_accuracy']
        student.load_state_dict(record['state_dict'])
        print('===== best model in stage 2 loaded, validating acc = %f. =====' % (record['validating_accuracy']))

    # model testing
    testing_accuracy = test(args, test_data_loader, student, description='testing')
    print('===== testing finished, testing acc = %f. =====' % (testing_accuracy))