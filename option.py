# -*- coding: utf-8 -*-
import os
import json
import argparse
import random
import shutil
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',   default='',                 help='experiment name')
parser.add_argument('--method',           default='BP',             help='method')
parser.add_argument('--data',             default='CIFAR10',    type=str,   help='type of dataset', choices=['mnist-biased','MNIST','CIFAR10','CIFAR100','SVHN','STL10','fMNIST','kMNIST'])
parser.add_argument('--batch_size',       default=512,        type=int,   help='mini-batch size')
parser.add_argument('--epoch',            default=10,        type=int,   help='epoch of each task')

parser.add_argument('--log_step',         default=50,     type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=1,      type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',         default='./',               help='data directory')
parser.add_argument('--save_dir',         default='./exps',           help='save directory for checkpoint')

parser.add_argument('--seed',             default=777,    type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
parser.add_argument('--cpu',              action='store_true',        help='enables cuda')
parser.add_argument('--gpu',              default=0, type=int,        help='which number of gpu used')
parser.add_argument('--dropout',          default=0.1,      type=float,   help='embedding dropout')
parser.add_argument('--start_dim',        default=64,    type=int,   help='start dim for FL')
parser.add_argument('--max_dim',          default=768,    type=int,  help='max dim for FL')
parser.add_argument('--lr',               default=0.001,      type=float,   help='learning rate')
parser.add_argument('--optimizer', type=str, default="AdamW",help='optimizer type')
parser.add_argument('--step_size', type=int, default = 0, help='step size for StepLR')
parser.add_argument('--arch', default='conv',    type=str,   help='type of architecture')
parser.add_argument('--resume',  default=None,    type=str,   help='checkpoint path')
parser.add_argument('--MS_gamma', default=0, type=float,   help='multistep scheduler decay rate')
parser.add_argument('--milestones', nargs='+', type=int, default=[300,450,950,1300],help='epoch milestones to decay learning rate at')
parser.add_argument('--linear',        action='store_true',       help='use final linear layers')
parser.add_argument('--perturbation',        action='store_true',       help='add perturbation to embeddings: legacy feature')
parser.add_argument('--num_layers', type=int, default = 3, help='num hidden layers for fc or depth for vit')
parser.add_argument('--patch', type=int, default = 4, help='patch size for VIT')
parser.add_argument('--patch_fc', type=int, default = 0, help='patch size for fc')
parser.add_argument('--cpb', type=int, default = 0, help='num classes per batch')
parser.add_argument('--no_logging',        action='store_true',       help='ddp')
parser.add_argument('--print_memory',        action='store_true',       help='')
parser.add_argument('--cuda_setting',        action='store_true',       help='')
parser.add_argument('--temp', type=float, default = 0.07, help='temperature for cosine simiarity')
parser.add_argument('--pre_config',        action='store_true',       help='')
parser.add_argument('--alpha', type=float, default = 1.0, help='coefficient for the feature loss')

#final linear


class Config():
    def __init__(self, opt) -> None:
        self.exp_name: str = opt.exp_name
        self.method: str = opt.method.upper()
        self.data:str = opt.data
        self.batch_size: int = opt.batch_size
        self.epoch: int = opt.epoch

        self.log_step: int = opt.log_step
        self.save_step: int = opt.save_step
        self.data_dir: str = opt.data_dir
        self.save_dir: str = opt.save_dir

        self.seed: int = opt.seed
        self.num_workers: int = opt.num_workers

        self.cuda: bool = not opt.cpu
        self.gpu: str = opt.gpu

        self.dropout: float = opt.dropout
        self.start_dim: int = opt.start_dim
        self.max_dim: int = opt.max_dim
        self.lr: float = opt.lr
        self.optimizer: str = opt.optimizer
        self.step_size: int = opt.step_size
        self.arch: str = opt.arch
        self.resume: str = opt.resume
        self.MS_gamma: float = opt.MS_gamma
        self.milestones: list = opt.milestones
        self.linear: bool = opt.linear
        self.perturbation: bool = opt.perturbation
        self.num_layers: int = opt.num_layers
        self.patch:int = opt.patch
        self.patch_fc:int = opt.patch_fc
        self.cpb:int = opt.cpb
        self.no_logging:bool = opt.no_logging
        self.print_memory:bool = opt.print_memory
        self.temp:float = opt.temp
        self.pre_config:bool = opt.pre_config
        self.alpha:float = opt.alpha
        self.cuda_setting:bool = opt.cuda_setting
        #assert len(self.__dict__) == len(opt.__dict__), "Check argparse"

        if self.pre_config:
            # 100 200 300 400 450 cifar10 conv
            #50 150 200 350 cifar10 FC 
            #50 75 100 125 mnist conv
            #50 100 125 mnist fc
            #50 100 150 cifar100 fc
            #50 75 100 125 150 cifar100 conv512
            ##50 100 150 200 250 300 \
            #100 200 250 300 350 simple cifar100 conv256
            #200 300 350 375 vgg cifar stl10
            #50 75 89 94 vgg svhn, kMNIST
            #100 150 175 188 vgg fMNIST
            if 'simple' in self.arch:
                self.lr = 0.0075
                self.dropout = 0
                self.batch_size = 512
                self.MS_gamma = 0.5
                self.optimizer = 'AdamW'
                if self.data == 'CIFAR10':
                    self.milestones = [100,200,300,400,450]
                    self.epoch = 500
                elif 'CIFAR100' in self.data:
                    self.milestones = [100,200,250,300,350]
                    self.epoch = 400
                elif 'MNIST' == self.data:
                    self.epoch = 150
                    self.milestones = [50, 75, 100, 125]
                else:
                    raise AssertionError(f'no predefined config. arch: {self.arch}  data: {self.data}')           

            elif 'fc' in self.arch:
                self.optimizer = 'AdamW'
                self.num_layers=3
                self.batch_size = 512
                self.MS_gamma = 0.5

                if self.data == 'CIFAR10':
                    self.milestones = [50, 150, 200, 350]
                    self.epoch = 400
                    self.lr = 0.0002
                    #self.patch_fc = 24
                    self.max_dim = 3072
                    self.dropout=0
                elif 'CIFAR100' in self.data:
                    self.epoch = 200
                    self.lr = 0.0001
                    self.milestones = [50, 100, 150]
                    #self.patch_fc = 12
                    self.max_dim = 3072
                    self.dropout= 0.3
                elif 'MNIST' == self.data:
                    self.epoch = 150
                    self.lr = 0.0005
                    #self.patch_fc = 8
                    self.max_dim = 1024
                    self.milestones = [50, 100, 125]
                    self.dropout=0               
                else:
                    raise AssertionError(f'no predefined config. arch: {self.arch}  data: {self.data}')           
            elif 'vgg8b' in self.arch:
                self.batch_size = 128
                self.MS_gamma = 0.25
                self.lr = 0.0005
                #self.optimizer='Adam'

                if self.data == 'CIFAR10' or 'CIFAR100' in self.data:
                    self.milestones = [200, 300, 350, 375]
                    self.epoch = 400
                    self.dropout = 0.05
                elif 'MNIST' == self.data:
                    self.epoch = 100
                    self.milestones = [50, 75, 89, 94]
                    self.dropout=0.1
                elif 'STL10' == self.data:
                    self.milestones = [200, 300, 350, 375]
                    self.epoch = 400
                    self.dropout = 0.1
                elif 'SVHN' == self.data:
                    self.epoch = 100
                    self.milestones = [50, 75, 89, 94]
                    self.dropout=0.05
                    self.lr = 0.0003
                elif 'fMNIST' == self.data:
                    self.epoch = 200
                    self.milestones = [100, 150, 175, 188]
                    self.dropout=0.1
                else:
                    raise AssertionError(f'no predefined config. arch: {self.arch}  data: {self.data}')           

        if 'simple' in self.arch:
            self.hyper_param = {
            'data': '', 
            'method': '',
            'lr': '',
            'arch': '',
            'epoch': 'Ep',
            'batch_size': 'B',
            'alpha':'a',
            'seed': 'SEED',
            }            
       
        elif 'fc' in self.arch:
            self.hyper_param = {
            'data': '', 
            'method': '',
            'lr': '',
            'arch': '',
            'patch_fc': 'p',
            'num_layers': 'nl',
            'max_dim':'m',
            'epoch': 'Ep',
            'batch_size': 'B',
            'alpha':'a',
            'seed': 'SEED',
            }
       
        elif 'vgg' in self.arch:
            self.hyper_param = {
            'data': '', 
            'method':'',
            'lr': '',
            'arch': '',
            'epoch': 'Ep',
            'batch_size': 'B',
            'alpha':'a',
            'seed': 'SEED',
            }                
        if 'cos' in self.method or 'COS' in self.method:
            self.hyper_param['temp'] = 'tmp'
        if self.dropout > 0:
            self.hyper_param['dropout'] = 'dp'
        if self.cpb > 0:
            self.hyper_param['cpb'] = 'cpb'
        if self.MS_gamma > 0:
            self.hyper_param['MS_gamma'] = 'MS'
        if self.linear:
            self.hyper_param['linear'] = 'linear'

        self.hyper_param.update({
            
        })

        self._build()

    def _build(self):
        # Set exp name
        for k, v in self.hyper_param.items():
            self.exp_name += f"_{v}{self.__getattribute__(k)}"

        if self.exp_name[0] == '_': self.exp_name = self.exp_name[1:]

        print(self.exp_name)
        self._save()

    def _backend_setting(self,local_rank=0):
        #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        #os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        #os.environ['CUDA_LAUNCH_BLOCKING'] = 1

        if self.seed is None:
            self.seed = random.randint(1, 10000)
        
        if torch.cuda.is_available() and not self.cuda:
            print('[WARNING] GPU is available, but not use it')

        random.seed(self.seed+local_rank)
        np.random.seed(self.seed+local_rank)
        torch.manual_seed(self.seed+local_rank)

        if self.cuda_setting:
            torch.cuda.manual_seed(self.seed+local_rank)
            torch.cuda.manual_seed_all(self.seed+local_rank)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True#False
            torch.backends.cudnn.enabled = False

    def _save(self):
        log_dir = os.path.join(self.save_dir, self.exp_name)
        if os.path.exists(log_dir):
            if 'debug' in self.exp_name: 
                isdelete = "y"
            elif self.resume is not None:
                isdelete = input("resume: delete existing exp dir (y/n): ")
            else:
                isdelete = input("delete exist exp dir (y/n): ")

            if isdelete == "y":
                shutil.rmtree(log_dir)
            elif isdelete == "n":
                if self.resume is not None:
                    pass
                else:    
                    raise FileExistsError
            else:
                raise FileExistsError

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        option_path = os.path.join(self.save_dir, self.exp_name, "options.json")

        with open(option_path, 'w') as fp:
            json.dump(self.__dict__, fp, indent=4, sort_keys=True)


def get_option() -> Config:
    option, unknown_args = parser.parse_known_args()
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"
    return Config(option)
