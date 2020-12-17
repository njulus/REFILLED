# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 19:52:05
"""

import os
import socket
import platform

# determine the operating system and the GPUs available currently

if platform.platform().startswith('Windows'):
    PLATFORM = 'CLIENT'
    WORKERS = 0
    n_gpus = 1

    with os.popen('nvidia-smi') as fp:
        lines = fp.readlines()
    lines = [line.replace('\n', '') for line in lines]
    index = [8, 11]
    lines = [lines[i] for i in index]
    memory = [int(line[35:40]) for line in lines]
    utility = [int(line[60:63]) for line in lines]
    id = [i for i in range(0, 16)]
    device_list = list(zip(memory, utility, id))
    device_list = sorted(device_list, key = lambda x:(x[0], x[1]))
    DEVICES = [device_list[i][2] for i in range(0, n_gpus)]

elif platform.platform().startswith('Linux'):
    hostname = socket.gethostname()
    if hostname == 'LAMDA1-GPU2':
        PLATFORM = 'SERVER'
        WORKERS = 8
        n_gpus = 1
        
        with os.popen('nvidia-smi') as fp:
            lines = fp.readlines()
        lines = [line.replace('\n', '') for line in lines]
        index = [8, 11, 14, 17, 20, 23, 26, 29,
                32, 35, 38, 41, 44, 47, 50, 53]
        lines = [lines[i] for i in index]
        memory = [int(line[35:40]) for line in lines]
        utility = [int(line[60:63]) for line in lines]
        id = [i for i in range(0, 16)]
        device_list = list(zip(memory, utility, id))
        device_list = sorted(device_list, key = lambda x:(x[0], x[1]))
        DEVICES = [device_list[i][2] for i in range(0, n_gpus)]
    else:
        PLATFORM = 'CLIENT'
        WORKERS = 8
        n_gpus = 1

        with os.popen('nvidia-smi') as fp:
            lines = fp.readlines()
        lines = [line.replace('\n', '') for line in lines]
        index = [8, 11]
        lines = [lines[i] for i in index]
        memory = [int(line[35:40]) for line in lines]
        utility = [int(line[60:63]) for line in lines]
        id = [i for i in range(0, 16)]
        device_list = list(zip(memory, utility, id))
        device_list = sorted(device_list, key = lambda x:(x[0], x[1]))
        DEVICES = [device_list[i][2] for i in range(0, n_gpus)]
        DEVICES = [0]
