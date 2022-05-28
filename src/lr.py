# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: lr.py
@time: 2022/4/23 20:52
@description: 
"""
from bisect import bisect_right
from math import cos, pi

from torch.optim import lr_scheduler
import numpy as np


class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min=1e-9, lr_max=1e-3 , warmup_iters=0, T_max=10, warmup_factor=0.1,warmup_method="linear",last_epoch=-1):

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_iters = warmup_iters
        self.T_max = T_max
        self.warmup_factor = warmup_factor
        self.cur = last_epoch + 1  # current epoch or iteration
        self.warmup_method = warmup_method
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted",
                " but got {}".format(warmup_method)
            )

    # def get_lr(self):
    #     if (self.warm_up == 0) & (self.cur == 0):
    #         lr = self.lr_max
    #     elif (self.warm_up != 0) & (self.cur < self.warm_up):
    #         if self.cur == 0:
    #             lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
    #         else:
    #             lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
    #         # if self.cur == 0:
    #         #     lr = self.lr_min
    #         # else:
    #         #     lr = self.lr_max / self.warm_up * self.cur
    #             # print(f'{self.cur} -> {lr}')
    #     else:
    #         # this works fine
    #         lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
    #              (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)
    #
    #     self.cur += 1
    #
    #     return [lr for base_lr in self.base_lrs]
    def get_lr(self):

        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:  # 0<10
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor  # 1/3
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters  # self.last_epoch是一直变动的[0,1,2,3,,,50]/10
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # self.warmup_factor=1/3
                # list = {"last_epoch": self.last_epoch, "warmup_iters": self.warmup_iters, "alpha": alpha,
                #         'warmup_factor': warmup_factor}
            lr = self.lr_max * warmup_factor
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                              (np.cos((self.cur - self.warmup_iters) / (self.T_max - self.warmup_iters) * np.pi) + 1)

        self.cur += 1
        return [lr for base_lr in self.base_lrs]  # self.base_lrs,optimizer初始学习率weight_lr=0.0003，bias_lr=0.0006


class WarmupMultiStepLR(lr_scheduler._LRScheduler):
    def __init__(
            self,optimizer,milestones, gamma=0.1, warmup_factor=0.01,warmup_iters=10, warmup_method="linear",last_epoch=-1,):
        if not list(milestones) == sorted(milestones):  # 保证输入的list是按前后顺序放的
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted",
                " but got {}".format(warmup_method)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    '''
    self.last_epoch是一直变动的[0,1,2,3,,,50]
    self.warmup_iters=10固定（表示线性warm up提升10个epoch）

    '''

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:  # 0<10
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor  # 1/3
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters  # self.last_epoch是一直变动的[0,1,2,3,,,50]/10
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # self.warmup_factor=1/3
                # list = {"last_epoch": self.last_epoch, "warmup_iters": self.warmup_iters, "alpha": alpha,
                #         'warmup_factor': warmup_factor}

        # print(base_lr  for base_lr in    self.base_lrs)
        # print(base_lr* warmup_factor* self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs)

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in
                self.base_lrs]  # self.base_lrs,optimizer初始学习率weight_lr=0.0003，bias_lr=0.0006

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch-max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr