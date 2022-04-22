# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: distributed_utils.py
@time: 2022/4/21 16:36
@description: 
"""

import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    '''
    设置GPU和随机种子
    :param args:
    :return:
    '''
    torch.manual_seed(args.random_seed)  # 为CPU设置随机数
    torch.cuda.manual_seed_all(args.random_seed)  # 为GPU设置随机数
    # 设置rank
    # 多gpu
    # 启动命令 CUDA_VISIBLE_DEVICES=x python -m torch.distributed.langth --nproc_per_node 4  main.py
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        args.logger.info('Not use distributed training')
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    args.logger.info('diatributed init (rank{}): {}'.format(args.rank, args.dist_url))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()  # 等待所有GPU运行到此处


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value