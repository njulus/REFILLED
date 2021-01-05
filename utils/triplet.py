# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-29 12:18:25
"""

import torch

def merge(args, anchor_id, positive_id, negative_id):
    merged_anchor_id = []
    merged_positive_id = []
    last_a = last_p = -1
    for i in range(0, anchor_id.size()[0]):
        if anchor_id[i] != last_a or positive_id[i] != last_p:
            merged_anchor_id.append(anchor_id[i])
            merged_positive_id.append(positive_id[i])
            last_a, last_p = anchor_id[i], positive_id[i]

    assert(len(merged_anchor_id) == len(merged_positive_id))
    n_tuples = len(merged_anchor_id)
    merged_negative_id = []
    for i in range(0, n_tuples):
        x = anchor_id == merged_anchor_id[i]
        y = positive_id == merged_positive_id[i]
        merged_negative_id.append(negative_id[x & y])
    
    merged_anchor_id = torch.Tensor(merged_anchor_id).long()
    merged_positive_id = torch.Tensor(merged_positive_id).long()
    
    max_len = 0
    for i in range(0, n_tuples):
        max_len = max(max_len, merged_negative_id[i].size()[0])

    mask = []
    for i in range(0, n_tuples):
        raw_len = len(merged_negative_id[i])
        if raw_len < max_len:
            merged_negative_id[i] = torch.cat([merged_negative_id[i],
                torch.zeros(max_len - raw_len).long().cuda(args.devices[0])])
        merged_negative_id[i] = merged_negative_id[i].unsqueeze(0)
        mask.append(torch.cat([
            torch.ones(raw_len).long().cuda(args.devices[0]),
            torch.zeros(max_len - raw_len).long().cuda(args.devices[0])
        ]).unsqueeze(0))
    merged_negative_id = torch.cat(merged_negative_id, dim=0)
    mask = torch.cat(mask, dim=0)

    return merged_anchor_id, merged_positive_id, merged_negative_id, mask