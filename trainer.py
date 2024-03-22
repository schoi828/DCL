from typing import Dict
import os
from tqdm import tqdm

import torch
from torch import nn, optim, Tensor
from torch.autograd import Variable
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch.nn as nn

from model.modules import SelfLocalLayer, LocalClfLayer, SimpleNet, FCNet,VGG
from option import Config
from utils import logger_setting, get_str_labels
import tempfile


def get_accuracy(pred, label):
    return sum(pred==label)/pred.shape[0]
def get_correct(pred, label):
    return sum(pred==label)

class Trainer(object):
    def __init__(self, option: Config, rank: int):
        super().__init__()
        if not option.no_logging:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer: SummaryWriter

        self.global_step=0
        self.resume_epoch = 0
        self.best_acc = 0
        self.best_acc_layer = 0
        self.device = torch.device('cuda', rank)
        self.option = option

        self.logger = logger_setting(option.exp_name, option.save_dir)
        self._build_model()       
        self.save_path = os.path.join(option.save_dir, option.exp_name)
        if option.resume is not None:
            self._load_checkpoint(option.resume,override=False,load_optimizers=True)
        self._set_cuda()

        self._print_net()
        self.prev_path = None
        self.str_classes = get_str_labels(self.option.data)
        self.best_zeroshot = 0

    def _build_optim(self,lr=None):
        if lr is None:
            lr = self.option.lr

        optim_type=self.option.optimizer
        optimizer = getattr(optim,optim_type)(self.net.parameters(), lr=lr)#*gamma)
        setattr(self,'optimizer',optimizer)

    def optim_step(self, loss):
        if self.net.training:
            loss.backward()
            self.optimizer.step()
            if hasattr(self,'local_scheduler'): self.local_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

    def _build_model(self):
        self.logger.info("Building model...")
        self.logger.info(f"{self.option.method} {self.option.arch} for {self.option.data}")
        in_ch = self.option.img_shape[0]
        img_size = self.option.img_shape[1]
        
        local_optim = self._build_optim() if 'BP' != self.option.method else None
        
        if self.option.data =='CIFAR20':
            classes = 20
        else:
            classes = self.option.classes
        self.classes = self.option.classes
        self.cf_matrix = MulticlassConfusionMatrix(num_classes=self.classes)
        
        if 'simple' in self.option.arch:
            out_dim = int(self.option.arch.split('simple')[-1])
            self.net = SimpleNet(in_ch,out_dim,self.classes,local_optim,self.option.linear,self.option.dropout,
                                 temp=self.option.temp,method=self.option.method)
        
        elif 'fc' in self.option.arch:
            in_dim = 28*28 if self.option.data == 'MNIST' else 32*32*3
            prenorm = False if 'post' in self.option.arch else True
            bn = True if 'bn' in self.option.arch else False
            self.net = FCNet(in_dim,self.option.max_dim,self.option.num_layers,classes,local_optim,
                            prenorm, bn, self.option.dropout, self.option.patch_fc,self.option.linear,
                            temp=self.option.temp,method=self.option.method)
            
        elif 'vgg' in self.option.arch:

            self.net = VGG(classes,img_size=img_size,local_optimizer=local_optim,
                           dropout=self.option.dropout,arch=self.option.arch,temp=self.option.temp,
                           method=self.option.method)
        
        if local_optim is None:
            self._build_optim()

        print(self.net)

    def _print_net(self):
        gen_hyper_params = {
            'arch': self.option.arch,
            'start_ch': self.option.start_dim,
            'max_ch': self.option.max_dim,
            'lr': self.option.lr,
            'batch_size': self.option.batch_size,
        }
        
        self.logger.info(f"[PARAMETER:Classifier]: {self._count_parameters(self.net)}")
        self.logger.info(gen_hyper_params)
        #self.logger.info(self.lwp.netS.augment)

    def _set_cuda(self):
        self.net.to(self.device)

    def forward_model(self,images, labels, val=False):
        pred_label, pred_layer, loss, loss_dict = self.net(images, labels)
        if not val: self.optim_step(loss)
        return pred_label, pred_layer, loss_dict
    
    def _train_one_epoch(self, data_loader, step):
            self.net.train()

            loss_total = {}
            accuracy = 0.0
            correct = 0
            layer_correct = 0
            total_num_train = 0
            total_iter = 0
            for batch in tqdm(data_loader):
                if isinstance(batch,list):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    keys = list(batch.keys())
                    if len(keys) < 3:
                        img_key, label_key = keys
                    else:
                        img_key, label_key, coarse_key = keys
                    
                    if self.option.data == "CIFAR20":
                        label_key = coarse_key

                    images, labels = batch[img_key].to(self.device), batch[label_key].to(self.device) 

                bsize = images.shape[0]
                total_num_train += bsize
                total_iter += 1

                self.forward_model
                pred_label, pred_layer, loss_dict = self.forward_model(images, labels)

                if total_iter == 2 and self.option.print_memory:
                    i=self.option.gpu
                    print(f'train: device:{i}','mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(torch.cuda.memory_allocated(i)/1e6,
                            torch.cuda.max_memory_allocated(i)/1e6))
                
                for i in loss_dict:
                    if i not in loss_total:
                        loss_total[i] = loss_dict[i]
                    loss_total[i] += loss_dict[i]

                correct += get_correct(pred_label, labels)
                if 'DCL' in self.option.method or pred_layer != 0:
                    layer_correct += get_correct(pred_layer, labels)
            
            for i in loss_total:
                loss_total[i] /=total_iter

            accuracy = correct/total_num_train
            msg = f"[TRAIN][{step:>3}] train acc: {accuracy:.4f} [{correct}/{total_num_train}], "
            loss_total['train_acc']=accuracy
            if layer_correct > 0:
                layer_accuracy = layer_correct/total_num_train
                loss_total['train_acc_layer'] = layer_accuracy
                msg+f'train layer acc: {layer_accuracy:.4f}'

            self.logger.info(msg)
            self.global_step += 1

            return loss_total
   
    def global_schedule(self, acc):

        if hasattr(self,'global_scheduler'):
            if self.option.MS_gamma > 0:
                acc = None
            self.global_scheduler.step(acc)
        lr = self.optimizer.param_groups[0]['lr']
        print("lr: ", lr)
        return lr 

    def train_task(self, train_loader, val_loader=None):
        self.net.train()
        start_epoch = max(1,self.resume_epoch+1)
        for step in range(start_epoch, self.option.epoch+1):
            print('training: ', self.option.exp_name)

            loader_train = train_loader
            logs = self._train_one_epoch(loader_train, step)
            if not self.option.no_logging:
                self._log_tensorboard(logs, self.global_step, tag='train')

            if step == 1 or step % self.option.save_step == 0 or step == self.option.epoch:
                test_acc, logs, cf_matrix = self._validate(val_loader, step=step, msg='[TEST]')
                logs['lr']= self.global_schedule(test_acc)
                if not self.option.no_logging:
                    self._log_tensorboard(logs, self.global_step, tag='val')
               
                if 'test_acc_layer' in logs:
                    acc_layer = logs['test_acc_layer']
                    if acc_layer > self.best_acc_layer:
                        if self.best_acc_layer != 0:
                            os.remove(os.path.join(self.save_path,f'acc_layer_{self.best_acc_layer:.4f}.txt'))
                        self.best_acc_layer = acc_layer
                        with open(os.path.join(self.save_path,f'acc_layer_{self.best_acc_layer:.4f}.txt'), 'w') as f:
                            f.write('best acc_layer')
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self._save_model(step, acc=test_acc, update=True)
                    if self.option.data not in ['SVHN','STL10','fMNIST','kMNIST']:
                        print('Saving Confusion Matrix...')
                        for i, cf in enumerate(cf_matrix):
                            torch.save(cf, os.path.join(self.save_path,f'cf_matrix{i}.pt'))
                            names = get_str_labels('CIFAR20') if cf.shape[0] == 20 else self.str_classes 
                            df_cm = pd.DataFrame(cf / torch.sum(cf, axis=1), index = [i for i in names],
                            columns = [i for i in names])
                            df_cm.to_csv(os.path.join(self.save_path,f'cf_matrix{i}.csv'))
                            if cf.shape[0] != 100:
                                plt.figure(figsize = (12,7))
                                sn.heatmap(df_cm, annot=True)
                                plt.savefig(os.path.join(self.save_path,'cf_matrix.png'))
        return test_acc
    
    @torch.no_grad()
    def _validate(self, data_loader, step=None, msg="Validation"):
        self.logger.info(msg)
        #self._mode_setting(is_train=False)
        self.net.eval()
        total_num_correct = 0
        total_num_test = 0
        total_iter = 0
        loss_total = {}
        cf_matrices = []
        total_num_correct_layer = 0
        cf_matrix = torch.zeros(self.classes,self.classes)
   
        for batch in tqdm(data_loader):
            if isinstance(batch,list):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                keys = list(batch.keys())
                if len(keys) < 3:
                    img_key, label_key = keys
                else:
                    img_key, label_key, coarse_key = keys
                if self.option.data == "CIFAR20":
                    label_key = coarse_key
                images, labels = batch[img_key].to(self.device), batch[label_key].to(self.device) 
            
            batch_size = images.shape[0]
            total_num_test += batch_size
            total_iter+=1
            # self.optim.zero_grad()
            
            pred_label, pred_layer, loss_dict = self.forward_model(images, labels, val=True)

            for i in loss_dict:
                if i not in loss_total:
                    loss_total[i] = loss_dict[i]
                loss_total[i] += loss_dict[i]
            
            total_num_correct += sum(pred_label==labels)
            if 'DCL' in self.option.method:
                total_num_correct_layer +=sum(pred_layer==labels)
            

        avg_acc = total_num_correct/total_num_test

        for i in loss_total:
            loss_total[i] /=total_iter
        cf_matrices.append(cf_matrix)
        
        

        msg = f"[EVALUATION][{step:>3}], ACCURACY : {avg_acc:.4f} [{total_num_correct}/{total_num_test}], "
        loss_total['val_accuracy'] = avg_acc

        if total_num_correct_layer > 0:
            avg_acc_layer = total_num_correct_layer/total_num_test
            msg+=f'LAYER ACCURACY : {avg_acc_layer:.4f}'
            loss_total['test_acc_layer'] = avg_acc_layer
        self.logger.info(msg)

        return avg_acc, loss_total, cf_matrices
    
    def _save_model(self, step, task=0, acc=0, update=False):
        
        #replace previous chkpoint
        if update and self.prev_path is not None:
            os.remove(self.prev_path)
            
        save_path = os.path.join(self.save_path, 
                                    f'chpt_epoch_{step}_{acc:.4f}.pth')
        self.prev_path = save_path

        optimizers = self.optimizer.state_dict()
        state_dict = self.net.state_dict()
        torch.save({
            'epoch': step,
            'global_step': self.global_step,
            'option': self.option,
            'accuracy': acc,
            'net_state_dict': state_dict,
            'optimizers': optimizers,
        }, save_path)

        print(f'[SAVE] checkpoint step: {step}')

    def _load_checkpoint(self, path, override, load_optimizers=True):
        
        checkpoint = torch.load(path,map_location='cpu')

        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.net = self.net.to(self.device)
        self.resume_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['accuracy']
        self.global_step = checkpoint['global_step']
        if override: self.option = checkpoint['option']
        
        if load_optimizers:
            optimizers = checkpoint['optimizers']
            self.optimizer.load_state_dict(optimizers)

        for i in range(self.resume_epoch):
            self.global_schedule(self.best_acc)
        del checkpoint
        print(f'[LOAD] checkpoint loaded: {path}')

    def _count_parameters(self, *models):
        num = 0
        for model in models:
            num += sum(p.numel() for p in model.parameters() if p.requires_grad)
        msg = f"{num/(1000000):.4f} M"
        return msg
    
    def _log_tensorboard(self, logs: Dict[str, float], step: int, tag=""):
        for key in logs.keys():
            name = f"{tag}/{key}" if tag else f"{key}"
            self.tb_writer.add_scalar(name, logs[key], global_step=step)
        self.tb_writer.flush()

class Trainer_BP(Trainer):
    def __init__(self, option: Config, rank: int):
        super().__init__(option,rank)
        self.loss = nn.CrossEntropyLoss()
        self._set_optimizer()
        
    def _build_optim(self):
        return None
    def forward_model(self, images, labels,val=False):
        logits = self.net(images)
        loss = self.loss(logits,labels)
        pred = logits.argmax(1)
        if not val:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return pred, 0, {}   
    """
    def _train_one_epoch(self, data_loader, step):
            #self._mode_setting(is_train=True)
            self.net.train()
            logs = {}
            loss_total = 0
            accuracy = 0.0
            correct = 0
            total_num_train = 0
            total_iter = 0

            for batch in tqdm(data_loader):
                if isinstance(batch,list):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    keys = list(batch.keys())
                    if len(keys) < 3:
                        img_key, label_key = keys
                    else:
                        img_key, label_key, coarse_key = keys
                    
                    images, labels = batch[img_key], batch[label_key] 
                images = images.to(self.device)
                labels = labels.to(self.device)
                bsize = images.shape[0]
                total_num_train += bsize
                total_iter += 1

                self.optimizer.zero_grad()
                #one_hot =  torch.zeros(labels.shape[0], self.classes, device=self.device).scatter_(1, labels.unsqueeze(1), 1.0)

                logits = self.net(images)
                
                loss = self.loss(logits,labels)
                pred = logits.argmax(1)
                
                loss.backward()
                self.optimizer.step()   
                loss_total+=loss.item()
                #accuracy += get_accuracy(pred_label,labels)
                correct += get_correct(pred,labels)
                if total_iter == 2 and self.option.print_memory:
                   
                    i = self.option.gpu
    
                    print(f'train: device:{i}','mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(torch.cuda.memory_allocated(i)/1e6,
                            torch.cuda.max_memory_allocated(i)/1e6))
            
            
            loss_total /=total_iter
            accuracy = correct/total_num_train
            msg = f"[TRAIN][{step:>3}] train acc: {accuracy:.4f} [{correct}/{total_num_train}]"
            logs['train_loss'] = loss_total
            logs['train_acc']=accuracy
            self.logger.info(msg)
            self.global_step += 1


            return logs
    """
    def _set_optimizer(self):
        lr = self.option.lr
        MS_gamma = self.option.MS_gamma
        MS_milestones = self.option.milestones
        optim_type=self.option.optimizer
        
        self.optimizer = getattr(optim,optim_type)(self.net.parameters(), lr=lr)
        if MS_gamma>0:
            self.global_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=MS_milestones, gamma=MS_gamma)
    """
    @torch.no_grad()
    def _validate(self, data_loader, valid_type=None, step=None, msg="Validation"):
        if self.option.local_rank ==0:
            self.logger.info(msg)

        self.net.eval()
        logs = {}
        total_num_correct = 0
        total_num_test = 0
        total_iter = 0
        loss_total = 0
        cf_matrix = torch.zeros(self.classes,self.classes)
        for batch in tqdm(data_loader):
            if isinstance(batch,list):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                keys = list(batch.keys())
                if len(keys) < 3:
                    img_key, label_key = keys
                else:
                    img_key, label_key, coarse_key = keys
                images, labels = batch[img_key], batch[label_key] 
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.shape[0]
            total_num_test += batch_size
            total_iter+=1
            
            one_hot =  torch.zeros(labels.shape[0], self.classes, device=self.device).scatter_(1, labels.unsqueeze(1), 1.0)
            logits = self.net(images,one_hot)
            loss = self.loss(logits,labels)
            pred = logits.argmax(1)
            
            loss_total+=loss.item()
            
            total_num_correct += sum(pred==labels)
            cf_matrix += self.cf_matrix(pred.cpu(),labels.cpu())

        loss_total/=total_iter
        avg_acc = total_num_correct/total_num_test

        if valid_type != None:
            msg = f"[EVALUATION - {valid_type}], ACCURACY : {avg_acc:.4f} [{total_num_correct}/{total_num_test}]"
        else:
            msg = f"[EVALUATION][{step:>3}], ACCURACY : {avg_acc:.4f} [{total_num_correct}/{total_num_test}]"

        logs['val_loss'] = loss_total
        logs['val_accuracy'] = avg_acc
        if self.option.local_rank ==0:
            self.logger.info(msg)

        return avg_acc, logs, [cf_matrix]

    """

class Trainer_Layerwise(Trainer):
    def __init__(self, option: Config, rank: int):
        super().__init__(option,rank)
        #self.setup_pipeline()
        self.net =self.net.to(self.device)
    
    def _build_optim(self,lr=None):
        if lr is None:
            lr = self.option.lr
        MS_gamma = self.option.MS_gamma
        MS_milestones = self.option.milestones
        optim_type=self.option.optimizer
        
        def local_optimizer(self):
            if self.emb_params is not None:
                optimizer = getattr(optim,optim_type)(list(self.parameters())+list(self.emb_params), lr=lr)
            else:
                optimizer = getattr(optim,optim_type)(list(self.parameters()), lr=lr)

            setattr(self,'optimizer',optimizer)

            if MS_gamma > 0:
               scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MS_milestones, gamma=MS_gamma)
               setattr(self, 'global_scheduler', scheduler) 

            def optim_step(self, loss):
                if self.training:
                    loss.backward()
                    self.optimizer.step()
                    if hasattr(self,'local_scheduler'): self.local_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            setattr(self,"optim_step",optim_step)
        
        return local_optimizer
    
    #no need for global optim_step. weights are updated by local optimizers
    def optim_step(self, loss):
        pass

    """
    def _train_one_epoch(self, data_loader, step):
            #self._mode_setting(is_train=True)
            self.net.train()

            loss_total = {}
            accuracy = 0.0
            correct = 0
            total_num_train = 0
            total_iter = 0
            correct_layer = 0
            for batch in tqdm(data_loader):
                if isinstance(batch,list):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    keys = list(batch.keys())
                    if len(keys) < 3:
                        img_key, label_key = keys
                    else:
                        img_key, label_key, coarse_key = keys
                    if self.option.data == "CIFAR100_C":
                        label_key = coarse_key
                    
                    images, labels = batch[img_key].to(self.device), batch[label_key].to(self.device) 
                bsize = images.shape[0]
                total_num_train += bsize
                total_iter += 1

                pred_label, pred_layer, _,loss_dict = self.net(images, labels)

                if total_iter == 5 and self.option.print_memory:
                    for i in range(torch.cuda.device_count()):
                        print(f'train: device:{i}','mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(torch.cuda.memory_allocated(i)/1e6,
                            torch.cuda.max_memory_allocated(i)/1e6))
                for i in loss_dict:
                    if i not in loss_total:
                        loss_total[i] = loss_dict[i]
                    loss_total[i] += loss_dict[i]
                #accuracy += get_accuracy(pred_label,labels)
                correct += get_correct(pred_label.cpu(), labels.cpu())
                if pred_layer is not None and 'FEAT' not in self.option.method:
                    correct_layer += get_correct(pred_layer.cpu(), labels.cpu())

            for i in loss_total:
                loss_total[i] /=total_iter
            accuracy = correct/total_num_train
            layer_accuracy = correct_layer/total_num_train
            msg = f"[TRAIN][{step:>3}] train acc: {accuracy:.4f} [{correct}/{total_num_train}] Layer train acc: {layer_accuracy:.4f} "
            loss_total['train_acc']=accuracy
            self.logger.info(msg)
            self.global_step += 1

            return loss_total
    """
    def global_schedule(self, acc):
        lr = self.option.lr

        if self.option.MS_gamma>0:
            for module in self.net.modules():
                #if isinstance(module,SelfLocalLayer) or isinstance(module,LocalClfLayer):
                if hasattr(module,'global_scheduler'):
                    module.global_scheduler.step()
                    #print('last lr: ',module.MS_scheduler.get_last_lr())
                    lr = module.optimizer.param_groups[0]['lr']
                    print("lr: ", lr)
        return lr
    """
    @torch.no_grad()
    def _validate(self, data_loader, valid_type=None, step=None, msg="Validation"):
        self.logger.info(msg)
        #self._mode_setting(is_train=False)
        self.net.eval()
        total_num_correct = 0
        total_num_test = 0
        total_iter = 0
        loss_total = {}
        cf_matrices = []
        total_num_correct_layer=0
        cf_matrix = torch.zeros(self.classes,self.classes)


        for batch in tqdm(data_loader):
            if isinstance(batch,list):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                keys = list(batch.keys())
                if len(keys) < 3:
                    img_key, label_key = keys
                else:
                    img_key, label_key, coarse_key = keys

                images, labels = batch[img_key].to(self.device), batch[label_key].to(self.device) 
            
            batch_size = images.shape[0]
            total_num_test += batch_size
            total_iter+=1
            # self.optim.zero_grad()
            
            #pred_label, _, loss_dict = self.net.asynch(images, labels)
            pred_label, pred_layer , _, loss_dict = self.net(images, labels)

            for i in loss_dict:
                if i not in loss_total:
                    loss_total[i] = loss_dict[i]
                loss_total[i] += loss_dict[i]
            
            total_num_correct += sum(pred_label.cpu()==labels.cpu())
            if pred_layer is not None and 'FEAT' not in self.option.method:
                total_num_correct_layer +=sum(pred_layer.cpu()==labels.cpu())
            cf_matrix += self.cf_matrix(pred_label.cpu(),labels.cpu())


        for i in loss_total:
            loss_total[i] /=total_iter
        avg_acc = total_num_correct/total_num_test
        avg_acc_layer = total_num_correct_layer/total_num_test
        cf_matrices.append(cf_matrix)
        
        
        if valid_type != None:
            msg = f"[EVALUATION - {valid_type}], ACCURACY : {avg_acc:.4f} [{total_num_correct}/{total_num_test}] LAYER ACCURACY : {avg_acc_layer:.4f}"
        else:
            msg = f"[EVALUATION][{step:>3}], ACCURACY : {avg_acc:.4f} [{total_num_correct}/{total_num_test}] LAYER ACCURACY : {avg_acc_layer:.4f}"
        loss_total['val_accuracy'] = avg_acc
        loss_total['test_acc_layer'] = avg_acc_layer
        self.logger.info(msg)

        return avg_acc, loss_total, cf_matrices
    """
    def _save_model(self, step, task=0, acc=0, update=False):
        
        #replace previous chkpoint
        if update and self.prev_path is not None:
            os.remove(self.prev_path)
            
        save_path = os.path.join(self.save_path, 
                                    f'chpt_epoch_{step}_{acc:.4f}.pth')
        self.prev_path = save_path

        
        optimizers = []

        for module in self.net.modules():
            if isinstance(module,SelfLocalLayer) or isinstance(module,LocalClfLayer):
                optimizers.append(module.optimizer.state_dict())
        
        #optimizers = self.optimizer.state_dict()
        torch.save({
            'epoch': step,
            'global_step': self.global_step,
            'option': self.option,
            'accuracy': acc,
            'net_state_dict': self.net.state_dict(),
            'optimizers': optimizers,
        }, save_path)

        print(f'[SAVE] checkpoint step: {step}')

    def _load_checkpoint(self, path, override, load_optimizers=True):
        
        checkpoint = torch.load(path,map_location='cpu')

        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.net = self.net.to(self.device)
        self.resume_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['accuracy']
        self.global_step = checkpoint['global_step']
        if override: self.option = checkpoint['option']
        
        if load_optimizers:
            optimizers = checkpoint['optimizers']
            self.optimizer.load_state_dict(optimizers)
            """
            for module in self.net.modules():
                if isinstance(module,SelfLocalLayer) or isinstance(module,LocalClfLayer):
                    module.optimizer.load_state_dict(optimizers.pop(0))
            """
        for i in range(self.resume_epoch):
            self.global_schedule(self.best_acc)
        del checkpoint
        print(f'[LOAD] checkpoint loaded: {path}')
