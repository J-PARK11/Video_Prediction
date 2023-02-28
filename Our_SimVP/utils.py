"""
Collect & Save useful custom methods 
"""
# library call
import os
import time
import datetime
import numpy as np
import pandas as pd
import random
import torch
#import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import logging

# Model Time Consumption Checker
class TimeHistory():
    def __init__(self, name):
        self.start_time = 0
        self.end_time = 0
        self.name = name
    
    def begin(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
    
    def print(self):
        if ((self.start_time > 0) and (self.end_time > 0)):
            sec = self.end_time - self.start_time
            result = datetime.timedelta(seconds=sec)
            result = str(result).split('.')[0]
            print(f'{self.name} : {result}')

# Tensorboard directory path definition
def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

# check dir & mkdir dir
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# definite random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cudnn.deterministic = True

# Ploting Learning Curve
def plot_learning_curve(df_hist):
    plt.figure(figsize=(8,6))
    plt.title('Loss Learning Curve')
    plt.plot(df_hist.loss, label='loss', color='black', linewidth=2.0)
    plt.plot(df_hist.val_loss, label='val_loss', color='green', linewidth=2.0)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=14)
    plt.savefig('figure/learning_curve.png')
    plt.show()

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

# Pytorch CUDA Connection
def torch_cuda():
    print('Available :',torch.cuda.is_available())
    print('Current device :',torch.cuda.current_device())
    print('Usuable device :',torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    cuda = torch.device('cuda')