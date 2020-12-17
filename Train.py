# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 20:59:35
"""

import os
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.nn import functional as F

from pytorch_metric_learning.miners import TripletMarginMiner

from Test import test, test_ncm

def pretrain(args, train_data_loader, validate_data_loader, network, model_save_path):
    # build a loss function
    loss_function = nn.CrossEntropyLoss()
    # build an optimizer
    optimizer = SGD(params=network.parameters(), lr=args.lr, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)
    # build a scheduler
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

    training_loss_list = []
    training_accuracy_list = []
    validating_accuracy_list = []
    best_validating_accuracy = 0

    for epoch in range(1, args.n_training_epochs + 1):
        # init training loss and training accuracy in this epoch
        training_loss = 0
        training_accuracy = 0
        # build a bar
        if not args.flag_no_bar:
            total = train_data_loader.__len__()
            bar = tqdm(total=total, desc='epoch %d' % (epoch), unit='batch')

        network.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            logits = network.forward(images)
            loss_value = loss_function(logits, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)
            training_loss += loss_value.cpu().item() * images.size()[0]
            training_accuracy += torch.sum((prediction == labels).float()).cpu().item()

            if not args.flag_no_bar:
                bar.update(1)

        # get average training loss and average training accuracy
        training_loss /= train_data_loader.dataset.__len__()
        training_loss_list.append(training_loss)
        training_accuracy /= train_data_loader.dataset.__len__()
        training_accuracy_list.append(training_accuracy)
        # get validating accuracy
        validating_accuracy = test(args, validate_data_loader, network, description='validating')
        validating_accuracy_list.append(validating_accuracy)

        if not args.flag_no_bar:
            bar.close()
        # output after each epoch
        print('epoch %d finish: training_loss = %f, training_accuracy = %f, validating_accuracy = %f' % (
            epoch, training_loss, training_accuracy, validating_accuracy
        ))

        # if we find a better model
        if not args.flag_debug:
            if validating_accuracy > best_validating_accuracy:
                best_validating_accuracy = validating_accuracy
                record = {
                    'state_dict': network.state_dict(),
                    'validating_accuracy': validating_accuracy,
                    'epoch': epoch
                }
                torch.save(record, model_save_path)

        # adjust learning rate
        scheduler.step()

    return training_loss_list, training_accuracy_list, validating_accuracy_list



def train_stage1(args, train_data_loader, validate_data_loader, teacher, student, model_save_path1):
    print('===== training stage 1 =====')
    # build a loss function
    loss_function = nn.KLDivLoss(reduction='batchmean')
    # build an optimizer
    optimizer1 = SGD(params=student.parameters(), lr=args.lr1, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)
    # build a scheduler
    scheduler1 = CosineAnnealingLR(optimizer1, args.n_training_epochs1, 0.1 * args.lr1)
    # generate a semi-hard triplet miner
    miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')

    training_loss_list1 = []
    validating_accuracy_list1 = []
    best_validating_accuracy = 0

    for epoch in range(1, args.n_training_epochs1 + 1):
        # init training loss and n_triplets in this epoch
        training_loss = 0
        n_triplets = 0
        # build a bar
        if not args.flag_no_bar:
            total = train_data_loader.__len__()
            bar = tqdm(total=total, desc='stage1: epoch %d' % (epoch), unit='batch')

        student.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            # teacher embedding
            with torch.no_grad():
                teacher_embedding = teacher.forward(images, flag_embedding=True)
                teacher_embedding = F.normalize(teacher_embedding, p=2, dim=1)
            
            # student embedding
            student_embedding = student.forward(images, flag_embedding=True)
            student_embedding = F.normalize(student_embedding, p=2, dim=1)

            # generate triplets
            with torch.no_grad():
                anchor_id, positive_id, negative_id = miner(student_embedding, labels)

            # get teacher embedding in triplets
            teacher_anchor = teacher_embedding[anchor_id]
            teacher_positive = teacher_embedding[positive_id]
            teacher_negative = teacher_embedding[negative_id]

            # get student embedding in triplets
            student_anchor = student_embedding[anchor_id]
            student_positive = student_embedding[positive_id]
            student_negative = student_embedding[negative_id]

            # get a-p dist and a-n dist in teacher embedding
            teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
            teacher_an_dist = torch.norm(teacher_anchor - teacher_negative, p=2, dim=1)

            # get a-p dist and a-n dist in student embedding
            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor - student_negative, p=2, dim=1)

            # get probability of triplets in teacher embedding
            teacher_prob = torch.sigmoid((teacher_an_dist - teacher_ap_dist) / args.tau1)
            teacher_prob_aug = torch.cat([teacher_prob.unsqueeze(1), 1 - teacher_prob.unsqueeze(1)])
            
            # get probability of triplets in student embedding
            student_prob = torch.sigmoid((student_an_dist - student_ap_dist) / args.tau1)
            student_prob_aug = torch.cat([student_prob.unsqueeze(1), 1 - student_prob.unsqueeze(1)])

            loss_value = 1000 * loss_function(torch.log(student_prob_aug), teacher_prob_aug)

            optimizer1.zero_grad()
            loss_value.backward()
            optimizer1.step()

            training_loss += loss_value.cpu().item() * student_prob.size()[0]
            n_triplets += student_prob.size()[0]

            if not args.flag_no_bar:
                bar.update(1)
    
        # get average training loss
        training_loss /= n_triplets
        training_loss_list1.append(training_loss)

        if not args.flag_no_bar:
            bar.close()
        
        if epoch % 10 == 0:
            # get validating accuracy
            validating_accuracy = test_ncm(args, validate_data_loader, student, description='validating')
            validating_accuracy_list1.append(validating_accuracy)
            # output after each epoch
            print('epoch %d finish: training_loss = %f, validating_accuracy = %f' % (
                epoch, training_loss, validating_accuracy
            ))

            # if we find a better model
            if not args.flag_debug:
                if validating_accuracy > best_validating_accuracy:
                    best_validating_accuracy = validating_accuracy
                    record = {
                        'state_dict': student.state_dict(),
                        'validating_accuracy': validating_accuracy,
                        'epoch': epoch
                    }
                    torch.save(record, model_save_path1)
        else:
            # output after each epoch
            print('epoch %d finish: training_loss = %f' % (epoch, training_loss))

        # adjust learning rate
        scheduler1.step()

    return training_loss_list1, validating_accuracy_list1



def train_stage2(args, train_data_loader, validate_data_loader, teacher, student, model_save_path2):
    print('===== training stage 2 =====')
    # build a loss function
    training_loss_function = nn.CrossEntropyLoss()
    teaching_loss_function = nn.KLDivLoss(reduction='batchmean')
    # build an optimizer
    optimizer2 = SGD([
        {'params':student.get_network_params(), 'lr': 0.1 * args.lr2},
        {'params':student.get_classifier_params(), 'lr':args.lr2}
    ], weight_decay=args.wd, momentum=args.mo, nesterov=True)
    # build a scheduler
    scheduler2 = MultiStepLR(optimizer2, args.point, args.gamma)

    # get number of classes and number of embedding dimensions
    n_classes = train_data_loader.dataset.get_n_classes()
    n_teacher_dimension = teacher.fc.in_features
    n_student_dimension = student.fc.in_features

    # get global class centers with teacher model
    global_class_center_file_path = 'saves/class_centers/' + \
        '_data=' + str(args.data_name) + \
        '_teacher=' + str(args.teacher_network_name) + \
        '.center'
    if os.path.exists(global_class_center_file_path):
        class_center = torch.load(global_class_center_file_path)
        class_center = class_center.cuda(args.devices[0])
    else:
        class_center = torch.zeros((n_classes, n_teacher_dimension)).cuda(args.devices[0])
        class_count = torch.zeros(n_classes).cuda(args.devices[0])
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])
            
            with torch.no_grad():
                embedding = teacher.forward(images, flag_embedding=True)
                for i in range(0, n_classes):
                    index_of_class_i = (labels == i)
                    class_center[i] += torch.sum(embedding[index_of_class_i], dim=0)
                    class_count[i] += index_of_class_i.size()[0]
        class_count = class_count.unsqueeze(1)
        class_center = class_center / class_count
        class_center = F.normalize(class_center, p=2, dim=1)
        torch.save(class_center, global_class_center_file_path)
    print('===== gloabl class centers ready. =====')

    training_loss_list2 = []
    teaching_loss_list2 = []
    training_accuracy_list2 = []
    validating_accuracy_list2 = []
    best_validating_accuracy = 0

    for epoch in range(1, args.n_training_epochs2 + 1):
        # init training loss, teaching loss, and training accuracy in this epoch
        training_loss = 0
        teaching_loss = 0
        training_accuracy = 0
        # build a bar
        if not args.flag_no_bar:
            total = train_data_loader.__len__()
            bar = tqdm(total=total, desc='stage2: epoch %d' % (epoch), unit='batch')

        student.train()
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])

            # compute student logits and training loss
            student_logits, student_embedding = student.forward(images, flag_both=True)
            training_loss_value = training_loss_function(student_logits, labels)

            # get local classes and their class centers
            label_table = torch.arange(n_classes).long().unsqueeze(1).cuda(args.devices[0])
            class_in_batch = (labels == label_table).any(dim=1)
            class_center_in_batch = class_center[class_in_batch]

            # compute teacher logits and teaching loss
            with torch.no_grad():
                teacher_logits = torch.mm(student_embedding, class_center_in_batch.t())
            
            teaching_loss_value = args.lambd * teaching_loss_function(
                F.log_softmax(student_logits[:, class_in_batch] / args.tau2),
                F.softmax(teacher_logits / args.tau2, dim=1)
            )

            loss_value = training_loss_value + teaching_loss_value

            optimizer2.zero_grad()
            loss_value.backward()
            optimizer2.step()

            prediction = torch.argmax(student_logits, dim=1)
            training_loss += training_loss_value.cpu().item() * images.size()[0]
            teaching_loss += teaching_loss_value.cpu().item() * images.size()[0]
            training_accuracy += torch.sum((prediction == labels).float()).cpu().item()

            if not args.flag_no_bar:
                bar.update(1)
    
        # get average training loss, average teaching loss, and average training accuracy
        training_loss /= train_data_loader.dataset.__len__()
        training_loss_list2.append(training_loss)
        teaching_loss /= train_data_loader.dataset.__len__()
        teaching_loss_list2.append(teaching_loss)
        training_accuracy /= train_data_loader.dataset.__len__()
        training_accuracy_list2.append(training_accuracy)
        # get validating accuracy
        validating_accuracy = test(args, validate_data_loader, student, description='validating')
        validating_accuracy_list2.append(validating_accuracy)

        if not args.flag_no_bar:
            bar.close()
        # output after each epoch
        print('epoch %d finish: training_loss = %f, teaching_loss = %f, training_accuracy = %f, validating_accuracy = %f' % (
            epoch, training_loss, teaching_loss, training_accuracy, validating_accuracy
        ))

        # if we find a better model
        if not args.flag_debug:
            if validating_accuracy > best_validating_accuracy:
                best_validating_accuracy = validating_accuracy
                record = {
                    'state_dict': student.state_dict(),
                    'validating_accuracy': validating_accuracy,
                    'epoch': epoch
                }
                torch.save(record, model_save_path2)

        # adjust learning rate
        scheduler2.step()
    
    return training_loss_list2, teaching_loss_list2, training_accuracy_list2, validating_accuracy_list2