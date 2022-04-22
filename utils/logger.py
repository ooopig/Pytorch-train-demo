# -*- coding: utf-8 -*-

"""
@author: 樊元新
@software: PyCharm
@file: logger.py
@time: 2022/4/19 9:09
@description: 
"""

import logging
import time
import os

def log_creater( output_dir,write_to_file):
  if write_to_file:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)
    # creat a log
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    # formatter = logging.Formatter(
    #   '[%(asctime)s][%(filename)s][%(funcName)s] ==> %(message)s')
    formatter = logging.Formatter(
      '%(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)
    print('log_to_file')
    log.info('creating {}'.format(final_log_file))
    return log
  else:
    # creat a log
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    # formatter = logging.Formatter(
    #   '[%(asctime)s][%(filename)s][%(funcName)s] ==> %(message)s')
    formatter = logging.Formatter(
      '%(message)s')

    # setFormatter
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(stream)

    return log