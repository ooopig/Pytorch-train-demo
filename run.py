# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: run.py
@time: 2022/5/19 15:16
@description: 仅支持单GPU训练
"""

from collections import OrderedDict

from parse_args import args
import os
import shutil
import time
from math import sqrt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader,distributed,BatchSampler,Sampler
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import transforms

from src.lr import WarmupCosineLR, WarmupMultiStepLR
from src.models.mnist_net import Net

from utils import logger
from utils.yaml_utils import match
from utils.distributed_utils import init_distributed_mode, reduce_value, cleanup


def init_path(args):
    result_path = os.path.join(args.save_model_path,args.dataset)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    args.logger.info('init path done.')
def print_all_parameters(args):
    params = args.__dict__
    args.logger.info('------------Model Parameters--------------')
    for k in params:
        args.logger.info('{}: {}'.format(k, params[k]))
    args.logger.info('------------end--------------')
def save_checkpoint(args, model, optimizer, save_path='', epoch=0, acc=0, loss=0, is_best=False, dataParalle = None):
    '''
    保存检查点
    :param args:
    :param model: 模型
    :param optimizer: 优化器
    :param save_path: 保存路径
    :param epoch: 当=0时，为了让所有GPU上的额模型保持相同初始化参数
    :param acc:
    :param loss:
    :param is_best:
    :param dataParalle:
    :return:
    '''
    checkpoint_dict = dict()
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['epoch_id'] = epoch
    checkpoint_dict['acc'] = acc
    checkpoint_dict['loss'] = loss
    checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
    out_tar = os.path.join(save_path, '{}-epoch-{}.pkl'.format(args.model_type, epoch))
    torch.save(checkpoint_dict, out_tar)
    args.logger.info('saving checkpoint to \'{}\''.format(out_tar))
    if is_best:
        best_dict_path = os.path.join(save_path, '{}-model_best_dict.pkl'.format(args.model_type))
        shutil.copyfile(out_tar, best_dict_path)
        args.logger.info('best model updated \'{}\''.format(best_dict_path))
        if args.save_whole_modle:
            best_model_path = os.path.join(save_path, '{}-mode_best.pkl'.format(args.model_type))
            torch.save(model,best_model_path)
            args.logger.info('best model updated \'{}\''.format(best_model_path))
def load_checkpoint(args, input_file):
    """
    Load model checkpoint.
    :param input_file: Checkpoint file path.
    """
    if os.path.exists(input_file):

        args.logger.info('load checkpoint from {}'.format(input_file))
        checkpoint = torch.load(input_file, map_location="cuda:{}".format(args.gpu))

        # 单GPU加载多GPU模型问题 去掉权重字典键名中的"module"，以保证模型的统一性
        if 'module' in next(iter(checkpoint)):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                    name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
                    new_state_dict[name] = v  # 新字典的key值对应的value一一对应
            checkpoint['model_state_dict'] = new_state_dict
        return checkpoint
    else:
        args.logger.info('no checkpoint found at \'{}\''.format(input_file))
        return None
def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
    '''
    当前mini_batch size不满足batch_size时进行补全
    :param self:
    :param mini_batch:
    :param batch_size:
    :param multi_answers:
    :return:
    '''
def split_train_dataset(epoch,batch_size: int, ratio: float) -> (DataLoader, DataLoader):
    '''
    将测试集划分为train/val，并返回dataloader
    可以使用K flod进行数据集划分
    tips：在一个batch中，多标签数据的数目应该相同
    :param batch_size:
    :param ratio:
    :return:
    '''

    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(), download=True)
    train_size = int(len(train_dataset) * ratio)
    val_size = int(len(train_dataset) - train_size)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True,
                              shuffle=False,pin_memory=args.pin_memory)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                              drop_last=True, shuffle=False, pin_memory=args.pin_memory)
    return train_loader, valid_loader
def split_test_dataset( batch_size: int) -> ( DataLoader):
    test_dataset = datasets.MNIST(root='data/', train=False,
                              transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,drop_last=False,shuffle=False)
    return test_loader
def train_step(args, model, device, train_loader, optimizer, epoch):
    model.train()
    '''
    训练时，把数据送入GPU
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output,dim=1), target)
        loss.backward()
        loss = reduce_value(loss, average = True)
        optimizer.step()
        if (batch_idx+1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) * args.world_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train(args, model, optimizer, lr_scheduler=None):
    args.logger.info('******train******')
    best_acc = get_best_acc(args)
    acc_deacy = 0

    # 加载checkpoint
    if args.begin_epoch > 1:
        latest_model_dict_path = os.path.join(args.checkpoint_path, args.dataset,
                                              '{}-epoch-{}.pkl'.format(args.model_type, args.begin_epoch - 1))
        if os.path.exists(latest_model_dict_path):
            checkpoint = load_checkpoint(args, latest_model_dict_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 训练
    for epoch in range(args.begin_epoch, args.epochs + 1):
        begin_time = time.time()
        train_loader, val_loader = split_train_dataset(epoch, args.batch_size, args.ratio)
        train_step(args, model, DEVICE, train_loader, optimizer, epoch)
        acc, loss = inference(args, model, DEVICE, val_loader)

        save_path = os.path.join(args.checkpoint_path, args.dataset)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if best_acc < acc:
            acc_deacy = 0
            save_checkpoint(args, model, optimizer, save_path, epoch, acc, loss, is_best=True)
        else:
            acc_deacy += 1
            # early stoping
            if acc_deacy == args.early_stop_num:
                args.logger.info('early stopping')
                exit(1)
            save_checkpoint(args, model, optimizer, save_path, epoch, acc, loss, is_best=False)
        end_time = time.time()
        args.logger.info('Epoch:{} run time:{}s lr:{} acc:{} loss:{}'.format(epoch, int(end_time - begin_time),
                                                                             optimizer.param_groups[0]['lr']
                                                                             , acc, loss))
        # 动态改变学习率
        if args.learning_scheduler:
            lr_scheduler.step()

def inference(args, model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    acc = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output,dim=1), target, reduction='sum') # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    args.logger.info('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
        test_loss, int(correct), len(test_loader.dataset),acc))
    return acc, test_loss
def get_best_acc(args):
    acc = 0.0
    # if args.begin_epoch > 1:
    #     best_model_dict_path  = os.path.join(args.checkpoint_path,args.dataset, '{}-model_best_dict.pkl'.format(args.model_type))
    #     checkpoint = load_checkpoint(args, best_model_dict_path)
    #     acc = checkpoint["acc"]
    best_model_dict_path = os.path.join(args.checkpoint_path, args.dataset,
                                        '{}-model_best_dict.pkl'.format(args.model_type))
    if os.path.exists(best_model_dict_path):
        checkpoint = load_checkpoint(args, best_model_dict_path)
        acc = checkpoint["acc"]

    args.logger.info('load best acc: {}'.format(acc))
    return acc
if __name__ == '__main__':
    # nohup cmd(cmd是被拷贝的命令) &
    args = match(args)
    args.logger = logger.log_creater(args.log_path, args.log_to_file)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find Gpu device for training")

    init_path(args)
    DEVICE = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # Model
    model = Net(args)
    model = model.to(DEVICE)
    args.logger.info('model load to-{}'.format(DEVICE))

    # 打印参数
    print_all_parameters(args)

    if args.train:
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.learning_rate}],
                               lr=args.learning_rate)

        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr_min=1e-6, lr_max=args.learning_rate,
                                      warmup_iters=args.warmup_iters,
                                      T_max=args.epochs, warmup_factor=args.warmup_factor,
                                      last_epoch=args.begin_epoch - 1)
        # lr_scheduler = WarmupMultiStepLR(optimizer=optimizer,milestones=args.milestones,gamma=args.decay_ratio,last_epoch=args.begin_epoch-1,
        #                                 warmup_factor=args.warmup_factor, warmup_iters=args.warmup_iters)

        train(args, model, optimizer, lr_scheduler)
    elif args.inference:
        best_model_dict_path = os.path.join(args.checkpoint_path, args.dataset,
                                            '{}-model_best_dict.pkl'.format(args.model_type))
        best_model_path = os.path.join(args.checkpoint_path, args.dataset, '{}-model_best.pkl'.format(args.model_type))
        if args.save_whole_modle:
            model = torch.load(best_model_path)
        else:
            checkpoint = load_checkpoint(args, best_model_dict_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        test_loader = split_test_dataset(args.batch_size)
        acc, loss = inference(args, model, DEVICE, test_loader)
    elif args.test:
        '''
        功能测试所用，可删除
        '''
        args.learning_rate *= sqrt(args.world_size)  # 学习率根据并行GPU数目倍增,n为GPU数目，lr变为n的平方根倍
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.learning_rate}],
                               lr=args.learning_rate)

        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr_min=1e-6, lr_max=args.learning_rate,
                                      warmup_iters=args.warmup_iters,
                                      T_max=args.epochs, warmup_factor=args.warmup_factor,
                                      last_epoch=args.begin_epoch - 1)
        test_model_dict_path = os.path.join(args.checkpoint_path, args.dataset,
                                              '{}-epoch-{}.pkl'.format(args.model_type, 10))
        if os.path.exists(test_model_dict_path):
            checkpoint = load_checkpoint(args, test_model_dict_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:

        args.logger.info('请选择正确的模式(train/inference)')
        exit(1)

