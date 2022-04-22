# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: parse_args.py
@time: 2022/4/19 9:28
@description: 
"""

import argparse
import os

parser = argparse.ArgumentParser(description='All parameters that the project needs')

# config file
parser.add_argument('--config_file',type=str,default='configs/config_test.yaml',required=True)

# logger
parser.add_argument('--log_to_file',type=bool, default=False,help='是否将日志写入到文件中')
parser.add_argument('--log_path',type=str, default=os.path.join(os.path.dirname(__file__), 'logs'),
                    help='日志输出路径，默认为../logs')

# Experiment control
parser.add_argument('--process_data', action='store_true', # 触发为真
                    help='process data (default: False)')
parser.add_argument('--train', action='store_true',
                    help='train src (default: False)')
parser.add_argument('--inference', action='store_true',
                    help=' inference src (default: False)')
parser.add_argument('--grid_search', action='store_true',
                    help='run grid_search')
parser.add_argument('--search_random_seed', action='store_true',
                    help='run search_random_seed')
parser.add_argument('--run_ablation_studies', action='store_true',
                    help='run ablation studies')
parser.add_argument('--run_analysis', action='store_true',
                    help='run algorithm analysis and print intermediate results (default: False)')

# 设备，多GPU
# 系统会自动分配
parser.add_argument('--device', type=str, default='cuda',help='gpu device (default: 0) or cpu')
# 开启的进程数，不用设置，会根据nproc_per_node自动设置
# parser.add_argument('--world-size', type=int, default=4,help='分布式进程数目')
parser.add_argument('--dist-url', type=str, default='env://',help='url to set up distributed training')

args = parser.parse_args()
