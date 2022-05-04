# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: test.py
@time: 2022/4/23 20:52
@description: 
"""

# class
from torch.optim import optimizer, Adam
import matplotlib.pyplot as plt
from src.lr import WarmupCosineLR,WarmupMultiStepLR
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc = nn.Linear(1, 10)
    def forward(self,x):
        return self.fc(x)

def lr_test():
    epochs = 10
    begin_epoch = 1
    warm_up = 5
    LR = 0.001
    model = Net()
    optimizer = Adam([{'params': model.parameters(), 'initial_lr': LR}] ,lr = LR)
    cosine_lr = WarmupCosineLR(optimizer, 1e-6, 1e-3, warm_up, epochs, 0.01,last_epoch=begin_epoch-1)
    # cosine_lr = WarmupMultiStepLR(optimizer, [7,9], warmup_iters=warm_up, last_epoch=begin_epoch-1)

    lrs = []
    for epoch in range(begin_epoch,epochs+1):
        optimizer.step()
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        cosine_lr.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, color='r')

    for i in range(0,len(lrs)):
        if i % 1 == 0:
            plt.text(i, lrs[i], str(lrs[i]))
    plt.show()

lr_test()