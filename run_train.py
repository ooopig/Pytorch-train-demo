# -*- coding: utf-8 -*-
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader,distributed,BatchSampler
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import transforms
from src.models.mnist_net import Net
from parse_args import args
from utils import logger
from utils.yaml_utils import match
from utils.distributed_utils import init_distributed_mode, reduce_value, cleanup
"""
@author: 樊元新
@software: PyCharm
@file: run_train.py
@time: 2022/4/19 9:27
@description: 
"""
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
        if args.rank == 0:
            args.logger.info('load checkpoint from {}'.format(input_file))
        checkpoint = torch.load(input_file, map_location="cuda:{}".format(args.local_rank))
        return checkpoint
    else:
        if args.rank == 0:
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
    # train_data = TensorDataset(torch.LongTensor(input_ids_train),
    #                            torch.LongTensor(input_types_train),
    #                            torch.LongTensor(input_masks_train),
    #                            torch.LongTensor(y_train))
    # train_sampler = RandomSampler(train_data)
    # train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(), download=True)
    train_size = int(len(train_dataset) * ratio)
    val_size = int(len(train_dataset) - train_size)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = distributed.DistributedSampler(train_dataset)
    train_sampler.set_epoch(epoch)
    val_sampler = distributed.DistributedSampler(val_dataset,shuffle=False)
    train_batch_sampler = BatchSampler(train_sampler,batch_size=batch_size,drop_last=True)
    nw = min([os.cpu_count(),batch_size if batch_size > 1 else 0, 8]) # number of workers

    train_loader = DataLoader(dataset=train_dataset,batch_sampler=train_batch_sampler, num_workers=nw
                              , pin_memory=args.pin_memory)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                              drop_last=True, sampler=val_sampler, num_workers=nw
                              ,shuffle=False, pin_memory=args.pin_memory)
    return train_loader, valid_loader
def split_test_dataset( batch_size: int) -> ( DataLoader):
    test_dataset = datasets.MNIST(root='data/', train=False,
                                  transform=transforms.ToTensor(), download=True)
    test_sampler = distributed.DistributedSampler(test_dataset,shuffle=False,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_sampler)

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
        if (batch_idx+1) % 10 == 0 and args.rank == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) * args.world_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
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

    # 等待所有进程计算完成
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    test_loss =  reduce_value(test_loss,average=False).item()
    correct =  reduce_value(correct,average=False).item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    if args.rank == 0:
        args.logger.info('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
            test_loss, int(correct), len(test_loader.dataset),acc))
    return acc, test_loss
def get_best_acc(args):
    acc = 0.0
    if args.begin_epoch > 1:
        best_model_dict_path  = os.path.join(args.checkpoint_path,args.dataset, '{}-model_best_dict.pkl'.format(args.model_type))
        checkpoint = load_checkpoint(args, best_model_dict_path)
        acc = checkpoint["acc"]
    if args.rank == 0:
        args.logger.info('load best acc: {}'.format(acc))
    return acc
if __name__ == '__main__':
    # nohup cmd(cmd是被拷贝的命令) &
    args = match(args)
    args.logger = logger.log_creater(args.log_path, args.log_to_file)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find Gpu device for training")
    init_distributed_mode(args)
    rank = args.rank
    if rank == 0:
        init_path(args)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    model = Net(args)
    model = model.to(DEVICE)
    args.logger.info('model load to-{}'.format(DEVICE))
    args.learning_rate *= args.world_size # 学习率根据并行GPU数目倍增
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(DEVICE)
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # 在第一个进程中打印参数
    if rank == 0:
        print_all_parameters(args)

    if args.train:
        # 1. 获取数据：get_train_data，get_test_data
        # 2. 创建data_loader
        # 3. 初始化模型
        # 4. 训练
        # 5. 验证
        if rank == 0: args.logger.info('******train******')
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
        else:
            # 保存第一块GPU参数，其他GPU载入相同参数，必须保证所有参与训练的GPU参数是完全一致的
            init_model_dict_path = os.path.join(args.checkpoint_path, args.dataset)
            if rank == 0:
                save_checkpoint(args,model,optimizer,init_model_dict_path)
            dist.barrier() # 用dist.barrier()来同步不同进程间的快慢
            init_checkpoint = load_checkpoint(args, os.path.join(init_model_dict_path, '{}-epoch-0.pkl'.format(args.model_type)))
            model.load_state_dict(init_checkpoint['model_state_dict'])


        # 训练
        for epoch in range(args.begin_epoch, args.epochs+1):
            if rank == 0:
                begin_time = time.time()
            train_loader, val_loader = split_train_dataset(epoch,args.batch_size, args.ratio)
            train_step(args,model, DEVICE, train_loader, optimizer, epoch)
            acc, loss = inference(args,model, DEVICE, val_loader)

            if rank == 0:
                save_path = os.path.join(args.checkpoint_path,args.dataset)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if best_acc < acc:
                    acc_deacy = 0
                    save_checkpoint(args, model, optimizer, save_path, epoch, acc, loss, is_best=True)
                else:
                    acc+=1
                    # early stoping
                    if acc == args.early_stop_num:
                        args.logger.info('early stopping')
                        break
                    save_checkpoint(args, model, optimizer, save_path, epoch, acc, loss, is_best=False)
                end_time = time.time()
                args.logger.info('Epoch:{},run time:{}s'.format(epoch, int(end_time-begin_time)))

        cleanup()
    elif args.inference:
        best_model_dict_path = os.path.join(args.checkpoint_path,args.dataset, '{}-model_best_dict.pkl'.format(args.model_type))
        best_model_path = os.path.join(args.checkpoint_path,args.dataset, '{}-model_best.pkl'.format(args.model_type))
        if args.save_whole_modle:
            model = torch.load(best_model_path)
        else:
            checkpoint = load_checkpoint(args, best_model_dict_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        test_loader = split_test_dataset(args.batch_size)
        acc, loss = inference(args,model, DEVICE, test_loader)
    else:
        if rank == 0:
            args.logger.info('请选择正确的模式(train/inference)')
            exit(1)

