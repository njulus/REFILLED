# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-08-14 15:33:19
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from torchvision import transforms

from tensorboardX import SummaryWriter

import Test
from utils import evaluate_embedding
from utils import Triplet

def do_train_process(train_dataset, validate_dataset, train_batch_size, validate_batch_size, model,
                     learning_rate_in_stage1, learning_rate_in_stage2, momentum, weight_decay, nesterov,
                     lr_lambda_in_stage1, lr_lambda_in_stage2, number_of_epochs_in_stage1, number_of_epochs_in_stage2,
                     flag_gpu, model_file_path_in_stage1, model_file_path_in_stage2, teacher, beta, tau1, tau2, tau3,
                     debug, visualize, log_dir, devices, number_of_classes):
    """
    Introduction of function
    ------------------------
    This function trains model on train_dataset and validates model on validate_dataset.
    Some hyperparameters are given to the function.

    Parameters
    ----------
    train_dataset: torch.utils.Dataset
        training dataset
    validate_dataset: torch.utils.Dataset
        validating dataset
    train_batch_size: int
        batch size used in train_dataset_loader
    validate_batch_size: int
        batch size used in validate_dataset_loader
    model: torch.nn.Module
        model to train
    learning_rate_in_stage1: float
        initial learning rate used by SGD optimizer in stage1
    learning_rate_in_stage2: float
        initial learning rate used by SGD optimizer in stage2
    momentum: float
        momentum uses by SGD optimizer
    weight_decay: float
        weight_decay used by momentum optimizer
    nesterov: bool
        indicate whether to use nesterov in SDG optimizer or not
    lr_lambda_in_stage1: function
        function controlling the change of learning rate used in stage1
    lr_lambda_in_stage2: function
        function conrtolling the change of learning rate used in stage2
    number_of_epochs_in_stage1: int
        maximum number of epochs in training phase in stage1
    number_of_epochs_in_stage2: int
        maximum number of epochs in training phase in stage2
    flag_gpu: bool
        inidicate whether to use gpu or not
    model_file_path_in_stage1: str
        indicate where to save the best model found in stage1
    model_file_path_in_stage2: str
        indicate where to save the best model found in stage2
    teacher: torch.nn.Module
        well-trained teacher model
    beta: float
        effiecient balancing training loss and teaching loss in stage2
    tau1: float
        temperature of stochastic triplet embedding
    tau2: float
        temperature of cross entropy calibration
    tau3: float
        temperature of softmax function in teaching loss
    debug: bool
        indicate whether under debug mode
    visualize: bool
        indicate whether to visualize results
    log_dir: bool
        a string indicating the tensorboard file folder
    devices: list of int
        ids of devices
    number_of_classes: int
        number of different classes in dataset
    
    Returns
    -------
    guiding_loss_after_each_epoch: list of float
        guiding loss after each epoch
    guiding_nmi_after_each_epoch: list of float
        guiding nmi after each epoch
    training_loss_after_each_epoch: list of float
        training loss after each epoch
    training_accuracy_after_each_epoch: list of float
        training accuracy after each epoch
    validating_accuracy_after_each_epoch: list of float
        validating_accuracy_after_each epoch
    """

    if visualize:
        # generate writer for this task
        writer = SummaryWriter(logdir = log_dir, flush_secs = 45)

    # generate train_dataset_loader
    train_dataset_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size,
        shuffle = True, drop_last = False)
    # calculate number of training instances
    number_of_training_instances = train_dataset.__len__()

    # generate optimizer used in stage1
    optimizer = SGD(params = model.parameters(), lr = learning_rate_in_stage1, momentum = momentum,
        weight_decay = weight_decay, nesterov = nesterov)
    # generate scheduler used in stage1
    scheduler = LambdaLR(optimizer = optimizer, lr_lambda = lr_lambda_in_stage1)

    # init guiding_loss_after_each_epoch
    guiding_loss_after_each_epoch = []
    # init guiding_nmi_after_each_epoch
    guiding_nmi_after_each_epoch = []
    # init best_guiding_nmi
    best_guiding_nmi = 0

    print('===== training stage1 start... =====')
    for epoch in range(1, number_of_epochs_in_stage1 + 1):
        print('stage1: epoch %d start...' % (epoch))

        # change model to train mode
        model.train()

        # init guiding_loss_after_this_epoch
        guiding_loss_after_this_epoch = 0
        # init guiding_nmi_after_this_epoch
        guiding_nmi_after_this_epoch = 0
        # init number_of_guiding_triplets
        number_of_guiding_triplets = 0
        for mini_batch_index, (feature, fine_label) in enumerate(train_dataset_loader):
            feature = feature.float().cuda(devices[0]) if flag_gpu else feature.float()
            fine_label = fine_label.long().cuda(devices[0]) if flag_gpu else fine_label.long()

            with torch.no_grad():
                # teacher model forward process
                teacher_embedding = teacher.forward(feature, 1)
                # normalize teacher_embedding
                teacher_embedding = F.normalize(teacher_embedding, p = 2, dim = 1)

            # student model forward process
            student_embedding = model.forward(feature, 1)
            # normalize student_embedding
            student_embedding = F.normalize(student_embedding, p = 2, dim = 1)
            # generate triplets according to student_embedding
            with torch.no_grad():
                anchors_id, positives_id, negatives_id = Triplet.generate_semi_hard_triplets(
                    student_embedding, fine_label, flag_gpu)

            # calculate guiding_loss_value
            # get anchors, positives and negatives under teacher model
            teacher_anchors = teacher_embedding[anchors_id]
            teacher_positives = teacher_embedding[positives_id]
            teacher_negatives = teacher_embedding[negatives_id]
            # get anchors, positives and negatives under student model
            student_anchors = student_embedding[anchors_id]
            student_positives = student_embedding[positives_id]
            student_negatives = student_embedding[negatives_id]
            # calculate ap_dist and an_dist under teacher model
            teacher_ap_dist = torch.norm(teacher_anchors - teacher_positives, p = 2, dim = 1)
            teacher_an_dist = torch.norm(teacher_anchors - teacher_negatives, p = 2, dim = 1)
            # calculate ap_dist and an_dist under student model
            student_ap_dist = torch.norm(student_anchors - student_positives, p = 2, dim = 1)
            student_an_dist = torch.norm(student_anchors - student_negatives, p = 2, dim = 1)

            # calculate teacher_output
            teacher_output = torch.sigmoid((teacher_an_dist - teacher_ap_dist) / tau1)
            # calculate student_output
            student_output = torch.sigmoid((student_an_dist - student_ap_dist) / tau1)

            # calculate teacher_output_augmented
            teacher_output_augmented = torch.cat([teacher_output.unsqueeze(1), 1 - teacher_output.unsqueeze(1)], dim = 1)
            # calculate student_output_augmented
            student_output_augmented = torch.cat([student_output.unsqueeze(1), 1 - student_output.unsqueeze(1)], dim = 1)

            # calculate guiding loss
            guiding_loss_value = 1000 * nn.KLDivLoss()(
                torch.log(student_output_augmented), teacher_output_augmented
            )

            # update guiding_loss_after_this_epoch
            guiding_loss_after_this_epoch += guiding_loss_value.data.cpu().item() * anchors_id.size()[0]

            number_of_guiding_triplets += anchors_id.size()[0]

            # model backward process
            optimizer.zero_grad()
            guiding_loss_value.backward()
            optimizer.step()
        
        # update guiding_loss_after_each_epoch
        guiding_loss_after_this_epoch /= number_of_guiding_triplets
        guiding_loss_after_each_epoch.append(guiding_loss_after_this_epoch)
        # update guiding_nmi_after_each_epoch
        guiding_nmi_after_this_epoch = evaluate_embedding.do_cluster(dataset = validate_dataset,
            batch_size = validate_batch_size, model = model, flag_gpu = flag_gpu, number_of_classes = number_of_classes, devices = devices)
        guiding_nmi_after_each_epoch.append(guiding_nmi_after_this_epoch)

        # save best model in stage1
        if guiding_nmi_after_this_epoch > best_guiding_nmi:
            best_guiding_nmi = guiding_nmi_after_this_epoch
            if not debug:
                torch.save(model.state_dict(), model_file_path_in_stage1)
        
        # print some information after each epoch
        print('stage1: epoch %d finish, guiding_loss = %f, guiding_nmi = %f' % (
            epoch, guiding_loss_after_this_epoch, guiding_nmi_after_this_epoch
        ))

        if visualize:
            # write data to the writer before scheduler.step()
            writer.add_scalars('stage1', {
                'guiding loss':guiding_loss_after_this_epoch,
                'guiding nmi':guiding_nmi_after_this_epoch,
                'learning rate':optimizer.param_groups[0]['lr']
            }, epoch)

        scheduler.step()
    
    print('===== training stage1 finish =====')

    # init training_loss_after_each_epoch
    training_loss_after_each_epoch = []
    # init teaching_loss_after_each_epoch
    teaching_loss_after_each_epoch = []
    # init training_accuracy_after_each_epoch
    training_accuracy_after_each_epoch = []
    # init validating_accuracy_after_each_epoch
    validating_accuracy_after_each_epoch = []
    # init best_validating_accuracy
    best_validating_accuracy = 0.0

    # load best model found in stage1
    model.load_state_dict(torch.load(model_file_path_in_stage1))
    print('===== best model in stage1 loaded =====')

    # check best_guiding_nmi
    best_guiding_nmi = evaluate_embedding.do_cluster(dataset = validate_dataset,
        batch_size = validate_batch_size, model = model, flag_gpu = flag_gpu, number_of_classes = number_of_classes, devices = devices)
    print('===== best guiding nmi = %f' % (best_guiding_nmi))

    optimizer = SGD(params = model.parameters(), lr = learning_rate_in_stage2, momentum = momentum,
        weight_decay = weight_decay, nesterov = nesterov)
    # generate scheduler used in stage2
    scheduler = LambdaLR(optimizer = optimizer, lr_lambda = lr_lambda_in_stage2)

    print('===== training stage2 start... =====')
    for epoch in range(1, number_of_epochs_in_stage2 + 1):
        print('epoch %d start...' % (epoch))

        # change model to train mode
        model.train()

        # init training_loss_after_this_epoch
        training_loss_after_this_epoch = 0
        # init teaching_loss_after_this_epoch
        teaching_loss_after_this_epoch = 0
        # init training_accuracy_after_this_epoch
        training_accuracy_after_this_epoch = 0
        # loop for each mini batch in train_dataset_loader
        for mini_batch_index, (feature, fine_label) in enumerate(train_dataset_loader):
            feature = feature.float().cuda(devices[0]) if flag_gpu else feature.float()
            fine_label = fine_label.long().cuda(devices[0]) if flag_gpu else fine_label.long()

            # teacher model embedding forward process
            with torch.no_grad():
                teacher_embedding = teacher.forward(feature, 1)
                inter_class_means = torch.zeros((number_of_classes, 128))
                if flag_gpu:
                    inter_class_means = inter_class_means.cuda(devices[0])
                label_counts = torch.zeros((number_of_classes))
                if flag_gpu:
                    label_counts = label_counts.cuda(devices[0])
                
                for i in range(0, teacher_embedding.size()[0]):
                    label_counts[fine_label[i]] += 1
                    inter_class_means[fine_label[i], :] += teacher_embedding[i, :]
                
                inter_class_means = (inter_class_means.t() / label_counts).t()

                norms = torch.norm(inter_class_means, dim = 1)
                inter_class_means = (inter_class_means.t() / norms).t()

                label_exists = []
                for i in range(0, label_counts.size()[0]):
                    if label_counts[i] != 0:
                        label_exists.append(i)
                label_exists = torch.Tensor(label_exists).long().cuda(devices[0])
                inter_class_means = inter_class_means[label_exists, :]

            # model forward process
            model_output = model(feature)

            teacher_output = torch.mm(teacher_embedding, inter_class_means.t())

            training_loss_value = nn.CrossEntropyLoss()(model_output, fine_label)
            local_model_output = model_output[:, label_exists]
            teaching_loss_value = beta * nn.KLDivLoss()(
                F.log_softmax(local_model_output / tau3, dim = 1),
                F.softmax(teacher_output / tau3, dim = 1)
            )
            # calculate total loss value
            loss_value = training_loss_value + teaching_loss_value
            
            # update training_loss_after_this_epoch
            training_loss_after_this_epoch += training_loss_value.data.cpu().item() * fine_label.size()[0]
            # update teaching_loss_after_this_epoch
            teaching_loss_after_this_epoch += teaching_loss_value.data.cpu().item() * fine_label.size()[0]
            # update training_accuracy_after_this_epoch
            predicted_label = torch.max(model_output, dim = 1)[1]
            training_accuracy_after_this_epoch += torch.sum(predicted_label == fine_label).data.cpu().item()

            # model backward process
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # update training_loss_after_each_epoch
        training_loss_after_this_epoch /= number_of_training_instances
        training_loss_after_each_epoch.append(training_loss_after_this_epoch)
        # update teaching_loss_after_each_epoch
        teaching_loss_after_this_epoch /= number_of_training_instances
        teaching_loss_after_each_epoch.append(teaching_loss_after_this_epoch)
        # update training_accuracy_after_each_epoch
        training_accuracy_after_this_epoch /= number_of_training_instances
        training_accuracy_after_each_epoch.append(training_accuracy_after_this_epoch)
        # update validating_accuracy_after_each_epoch
        validating_accuracy_after_this_epoch = Test.do_test_process(dataset = validate_dataset,
            batch_size = validate_batch_size, model = model, flag_gpu = flag_gpu, devices = devices)
        validating_accuracy_after_each_epoch.append(validating_accuracy_after_this_epoch)

        # save best model
        if validating_accuracy_after_this_epoch > best_validating_accuracy:
            best_validating_accuracy = validating_accuracy_after_this_epoch
            if not debug:
                torch.save(model.state_dict(), model_file_path_in_stage2)

        # print some information after each epoch
        print('stage2: epoch %d finish, training_loss = %f, teaching_loss = %f, training_accuracy = %f, validating_accuracy = %f' % (
            epoch, training_loss_after_this_epoch, teaching_loss_after_this_epoch,
            training_accuracy_after_this_epoch, validating_accuracy_after_this_epoch))
        
        if visualize:
            writer.add_scalars('stage2', {
                'training loss':training_loss_after_this_epoch,
                'teaching loss':teaching_loss_after_this_epoch,
                'training accuracy':training_accuracy_after_this_epoch,
                'validating accuracy':validating_accuracy_after_this_epoch,
                'learning rate':optimizer.param_groups[0]['lr']
            }, epoch)
        
        scheduler.step()

        beta = beta * math.exp(-0.05)

    print('===== training stage2 finish =====')
    print('best_validating_accuracy = %f' % (best_validating_accuracy))

    if visualize:
        # close writer
        writer.close()

    return guiding_loss_after_each_epoch, guiding_nmi_after_each_epoch, \
        training_loss_after_each_epoch, teaching_loss_after_each_epoch, \
            training_accuracy_after_each_epoch, validating_accuracy_after_each_epoch


def calculate_inter_class_mean(embedding, label, number_of_classes, flag_gpu):
    """
    Introduction of function
    ------------------------
    This function finds inter-class means in a given mini-batch. For classes that are
    in the mini-batch, exact inter-class means are calculated. For classes that are
    not in the mini-batch, we simply ignore them.

    Parameters
    ----------
    embedding: torch.autograd.Variable
        embedding of samples in a mini-batch generated by teacher model
    label: torch.autograd.Variable
        label of samples in a mini-batch
    number_of_classes: int
        number of different classes in this task
    flag:gpu: bool
        indicate whether to use a gpu

    Returns
    -------
    inter_class_mean: torch.autograd.Variable
        inter-class means in a given mini-batch
    """
    
    # calculate some useful variables
    batch_size = embedding.size()[0]
    number_of_embedding_dimensions = embedding.size()[1]

    # init inter_class_mean
    inter_class_mean = torch.zeros((number_of_classes, number_of_embedding_dimensions))
    if flag_gpu:
        inter_class_mean = inter_class_mean.cuda()
    
    # generate a tensor containing all labels
    all_label = torch.Tensor(range(0, number_of_classes)).long()
    if flag_gpu:
        all_label = all_label.cuda()
    all_label_stacked = torch.stack([all_label] * batch_size, dim = 1)

    label_stacked = torch.stack([label] * number_of_classes, dim = 0)

    # find which classes are in this mini-batch
    exist = torch.sum(torch.eq(all_label_stacked, label_stacked), dim = 1) > 0
    
    # for classes that are in the mini-batch
    embedding_stacked = torch.stack([embedding] * number_of_classes, dim = 0)
    all_label_stacked = all_label_stacked.view(-1)
    label_stacked = label_stacked.view(-1)
    mask = torch.stack([torch.eq(all_label_stacked, label_stacked)] * number_of_embedding_dimensions, dim = 1) \
        .view(number_of_classes, batch_size, number_of_embedding_dimensions)
    inter_class_mean = torch.mean(embedding_stacked * mask.float(), dim = 1).squeeze(1)
    
    inter_class_mean = inter_class_mean[exist.nonzero().squeeze(1)]

    # normalize inter-class means
    norms = torch.norm(inter_class_mean, dim = 1)
    inter_class_mean = (inter_class_mean.t() / norms).t()
    
    return inter_class_mean