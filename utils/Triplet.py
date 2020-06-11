# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2019-08-15 12:22:55
"""

import torch

def calculate_pairwise_distances(embeddings, flag_gpu):
    """
    Introduction of function
    ------------------------
    This function calculates pairwise distance matrix of given embeddings.

    Parameters
    ----------
    embeddings: torch.autograd.Variable
        embeddings to calculate distances
    flag_gpu: bool
        indicate whether to use a gpu
    
    Returns
    -------
    pairwise_distances: torch.autograd.Variable
        pairwise distances between embeddings
    """

    # calculate some useful variables
    batch_size = embeddings.size()[0]

    # calculate squared pairwise distances
    pairwise_distances = torch.add(
        torch.sum(embeddings ** 2, dim = 1, keepdim=True),
        torch.sum(embeddings.t() ** 2, dim = 0, keepdim=True)) - \
            2.0 * torch.mm(embeddings, embeddings.t())
    
    # deal with numerical inaccuracies
    zero_matrix = torch.zeros_like(pairwise_distances)
    if flag_gpu:
        zero_matrix = zero_matrix.cuda()
    pairwise_distances = torch.max(pairwise_distances, zero_matrix)

    # clear diagonal values
    one_matrix = torch.ones_like(pairwise_distances)
    if flag_gpu:
        one_matrix = one_matrix.cuda()
    eye_matrix = torch.eye(batch_size)
    if flag_gpu:
        eye_matrix = eye_matrix.cuda()
    mask_offdiagonal = one_matrix - eye_matrix
    pairwise_distances = pairwise_distances * mask_offdiagonal

    return pairwise_distances


def masked_minimum(data, mask):
    """
    Introduction of function
    ------------------------
    This function calculates the minimum values of each row under a given mask

    Parameters
    ----------
    data: torch.autograd.Variable
        a data matrix of shape (n, d)
    mask: torch.autograd.Variable
        a mask of shape (n, d)

    Returns
    -------
    masked_minimum: torch.autograd.Variable
        masked minimum values of each row
    masked_minimum_id: torch.autograd.Variable
        ids of masked minimum values of each row
    """

    flag = (torch.sum(mask, dim = 1, keepdim = True) == 1).float()
    only_nonzero = torch.max(mask, dim = 1, keepdim = True)[1]

    maximum = torch.max(data, dim = 1, keepdim = True)[0]
    masked_minimum, masked_minimum_id = torch.min((data - maximum) * mask, dim = 1, keepdim = True)
    masked_minimum += maximum

    masked_minimum_id = (1 - flag) * masked_minimum_id.float() + flag * only_nonzero.float()
    return masked_minimum, masked_minimum_id.long()


def masked_maximum(data, mask):
    """
    Introduction of function
    ------------------------
    This function calculates the maximum values of each row under a given mask

    Parameters
    ----------
    data: torch.autograd.Variable
        a data matrix of shape (n, d)
    mask: torch.autograd.Variable
        a mask of shape (n, d)

    Returns
    -------
    masked_maximum: torch.autograd.Variable
        masked maximum values of each row
    masked_maximum_id: torch.autograd.Variable
        ids of masked maximum values of each row
    """

    flag = (torch.sum(mask, dim = 1, keepdim = True) == 1).float()
    only_nonzero = torch.max(mask, dim = 1, keepdim = True)[1]

    minimum = torch.min(data, dim = 1, keepdim = True)[0]
    masked_maximum, masked_maximum_id = torch.max((data - minimum) * mask, dim = 1, keepdim = True)
    masked_maximum += minimum

    masked_maximum_id = (1 - flag) * masked_maximum_id.float() + flag * only_nonzero.float()
    return masked_maximum, masked_maximum_id.long()


def generate_semi_hard_triplets(embeddings, labels, flag_gpu):
    """
    Introduction of function
    ------------------------
    This function generates semi-hard triplets according to embeddings.

    Parameters
    ----------
    embeddings: torch.autograd.Variable
        a set of embedding for the generation of semi-hard triplets
    labels: torch.autograd.Variable
        labels of embeddings
    flag_gpu: bool
        indicate whether to use a gpu or not

    Returns
    -------
    anchors_id: torch.autograd.Variable
        ids of anchors
    positives_id: torch.autograd.Variable
        ids of positives
    negatives_id: torch.autograd.Variable
        ids of negatives
    """

    # calculate some useful variables
    batch_size = embeddings.size()[0]

    # calculate pairwise distances
    pairwise_distances = calculate_pairwise_distances(embeddings, flag_gpu)
    # calculate pairwise adjacency
    pairwise_adjacency = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    # calculate pairwise not-adjacency
    pairwise_not_adjacency = pairwise_adjacency ^ 1

    # we copy pairwise_distances b times and each copy corresponds to a different p(positive)
    tiled_pairwise_distances = pairwise_distances.repeat([batch_size, 1])

    # calculate mask for calculating negatives outside: smallest dist[a, n] where dist[a, n] > dist[a, p]
    # mask is of shape (batch_size * batch_size, batch_size), and mask[p * b + a, n] = 1 means (a, p, n) forms a triplet
    # (a, p, n) forms a triplet means pairwise_not_adjacency[a, n] = 1 and dist[a, n] > dist[a, p]
    mask = pairwise_not_adjacency.repeat([batch_size, 1]) & \
        (tiled_pairwise_distances > (pairwise_distances.t().contiguous().view(-1, 1)))
    
    # calculate mask_final for calculating semi-hard negatives
    # mask_final is of shape (batch_size * batch_size, 1), and mask_final[p * b + a, 0] > 0 means
    # there exists a legal triplet (a, p, n) when fixing anchor and positive
    # legal triplet means we should use negatives outside
    mask_final = (torch.sum(mask.float(), dim = 1, keepdim = True) > 0.0)
    # now we reshape mask_final to shape (batch_size, batch_size)
    # note that rows correspond to positives and columns correspond to anchors
    mask_final = mask_final.view(batch_size, batch_size)
    # now rows correspond to anchors and columns correspond to positives
    mask_final = mask_final.t()

    pairwise_not_adjacency = pairwise_not_adjacency.float()
    mask = mask.float()

    # calculate negatives outside: smallest dist[a, n] where dist[a, n] > dist[a, p]
    negatives_outside, negatives_outside_id = masked_minimum(tiled_pairwise_distances, mask)
    # now negatives_outside is of shape (batch_size, batch_size) and negatives_outside[a, p] is when fixing
    # a as anchor and p as positive, the distance between a and its negative outside
    negatives_outside = negatives_outside.view(batch_size, batch_size).t()
    # now negatives_outside_id is of shape (batch_size, batch_size) and negatives_outside_id[a, p] is when fixing
    # a as anchor and p as positive, the id of its negative outside
    negatives_outside_id = negatives_outside_id.view(batch_size, batch_size).t()

    # calculate negatives inside: largest dist[a, n]
    negatives_inside, negatives_inside_id = masked_maximum(pairwise_distances, pairwise_not_adjacency)
    # now negatives_inside is of shape (batch_size, batch_size) and negatives_inside[a, :] is when fixing
    # a as anchor and any positive as positive, the distance between a and its negative inside
    negatives_inside = negatives_inside.repeat([1, batch_size])
    # now negatives_inside_id is of shape (batch_size, batch_size) and negatives_inside_id[a, :] is when fixing
    # a as anchor and any positive as positive, the id of its negative inside
    negatives_inside_id = negatives_inside_id.repeat([1, batch_size])

    # calculate semi-hard negatives
    # mask_final[a, p] = 1 means there exists a legal negative outside when fixing a as anchor and p as positive
    # otherwise we use negative inside
    # now semi_hard_negatives is of shape (batch_size, batch_size) and semi_hard_negatives[a, p] is when fixing
    # a as anchor and p as positive, the distance between a and its semi-hard negative
    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)
    # now semi_hard_negatives_id is of shape (batch_size, batch_size) and semi_hard_negatives[a, p] is when fixing
    # a as anchor and p as positive, the id of its semi-hard negative 
    semi_hard_negatives_id = torch.where(mask_final, negatives_outside_id, negatives_inside_id)

    # calculate mask_positives for calculating positives
    eye_matrix = torch.eye(batch_size)
    if flag_gpu:
        eye_matrix = eye_matrix.cuda()
    mask_positives = pairwise_adjacency.float() - eye_matrix
    
    # we reshape semi_hard_negatives_id to shape (batch_size, batch_size) firstlt, and then
    # create anchors_id and positives_id. Finally we choose legal triplets according to mask_positives.
    negatives_id = semi_hard_negatives_id.view(-1, 1).squeeze(1)
    anchors_id = torch.Tensor(range(0, batch_size)).repeat([batch_size, 1]).t().contiguous().view(-1, 1).squeeze(1)
    if flag_gpu:
        anchors_id = anchors_id.cuda()
    positives_id = torch.Tensor(range(0, batch_size)).repeat([batch_size, 1]).view(-1, 1).squeeze(1)
    if flag_gpu:
        positives_id = positives_id.cuda()
    needed = torch.nonzero(mask_positives.view(-1, 1))[:, 0]
    anchors_id = anchors_id[needed].long()
    positives_id = positives_id[needed].long()
    negatives_id = negatives_id[needed].long()
    
    if check_triplets(anchors_id, positives_id, negatives_id, labels):
        # print_triplets(anchors_id, positives_id, negatives_id, labels)
        return anchors_id, positives_id, negatives_id
    else:
        print('Illegal Triplets!')
    # return anchors_id, positives_id, negatives_id


def check_triplets(anchors_id, positives_id, negatives_id, labels):
    """
    Introduction of function
    ------------------------
    This function checks whether all generated triplets are legal triplets. In detail,
    we want to make sure that for all (a, p, n) in anchors_id * positives_id * negatives_id,
    labels[a] == labels[p] and labels[a] != labels[n].

    Parameters
    ----------
    anchors_id: torch.autograd.Variable
        ids of anchors
    positives_id: torch.autograd.Variable
        ids of positives
    negatives_id: torch.autograd.Variable
        ids of negatives

    Returns
    -------
    flag_legal: bool
        indicate whether all the triplets are legal
    """

    # calculate labels of anchors, positives and negatives
    anchors_label = labels[anchors_id]
    positives_label = labels[positives_id]
    negatives_label = labels[negatives_id]

    # check three constraints
    constraint1 = torch.eq(anchors_label, positives_label)
    constraint2 = torch.eq(anchors_id, positives_id) ^ 1
    constraint3 = torch.eq(anchors_label, negatives_label) ^ 1

    if torch.all(constraint1) and torch.all(constraint2) and torch.all(constraint3):
        flag_legal = True
    else:
        flag_legal = False
        print(torch.all(constraint1))
        print(torch.all(constraint2))
        print(torch.all(constraint3))
        for i in range(0, len(anchors_id)):
            print('labels: a = %d, p = %d, n = %d' % (
                anchors_label[i], positives_label[i], negatives_label[i]
            ))
            print('ids: a = %d, p = %d, n = %d' % (
                anchors_id[i], positives_id[i], negatives_id[i]
            ))
    
    return flag_legal


def print_triplets(anchors_id, positives_id, negatives_id, labels):
    """
    Introduction of function
    ------------------------
    This function prints some information about constructed triplets.

    Parameters
    ----------
    anchors_id: torch.autograd.Variable
        ids of anchors
    positives_id: torch.autograd.Variable
        ids of positives
    negatives_id: torch.autograd.Variable
        ids of negatives
    
    Returns
    -------
    NONE
    """

    number_of_triplets = len(anchors_id)
    anchors_label = labels[anchors_id]
    positives_label = labels[positives_id]
    negatives_label = labels[negatives_id]

    print('number of triplets: %d' % (number_of_triplets))
    for i in range(0, 100):
        print('ids of triplet: a = %d, p = %d, n = %d' % (
            anchors_id[i], positives_id[i], negatives_id[i]
        ))
        print('labels of triplet: a = %d, p = %d, n = %d' % (
            anchors_label[i], positives_label[i], negatives_label[i]
        ))