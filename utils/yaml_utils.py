# -*- coding: utf-8 -*-
import os
import sys
"""
@author: 樊元新
@software: PyCharm
@file: yaml_utils.py
@time: 2022/4/19 9:33
@description: 读取yaml文件配置的参数
"""

import yaml

def get_configs(file_name):
    stream = open(file_name, 'r', encoding='utf-8')
    confs = yaml.safe_load(stream)
    stream.close()
    return confs

def match(args):
    '''
    匹配yaml文件和参数名，优先级：yaml > 命令行参数
    :param args:
    :return:
    '''

    if os.path.isfile(args.config_file):
        conf = get_configs(args.config_file)
        for key in conf:
            args.__setattr__(key, conf[key])
    else:
        raise ('无法读取配置文件')
        sys.exit(1)

    return args