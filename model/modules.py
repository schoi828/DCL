import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from model.hook import *
#import torchsummary
from model.criterion import SupConLoss, dict_cont_loss

#For intermediate layers and local loss calculation 
class SelfLocalLayer(nn.Module):
    def __init__(
        self,
        module,
        init_function,
        patch=1,                  
        loss = dict_cont_loss,
        temp=0.07,
        emb_params = [],
        method = 'DCL'
    ):
        super().__init__()
        self.module = module
        self.emb_params = emb_params
        if init_function is not None:
            init_function(self)
        self.loss=loss
        self.patch=patch
        if 'FEAT' in method:
            self.sup_con = SupConLoss()
        self.method = method
        self.temp = temp

    def forward(self, latent, labels, imgs, inference=False, emb_dict=None,out_pred=False):
        
        #forward
        x = self.module(latent.detach())
        
        loss_dict = {}
        pred = None

        if inference:
            return x
        
        #feature loss without auxiliary networks
        if 'FEAT' in self.method:
            loss = self.sup_con(x,labels,mean=True)
            loss_dict['feat_loss'] = loss.item()
        

        if 'DCL' in self.method and emb_dict is not None:
            #if len(x.shape) > 2 and x.shape[1] < emb_dict.shape[-1]:
            #    emb_dict = emb_dict.detach()
            normalize = 'COS' in self.method #use cosine simiarity instead of dot product
            temp = self.temp if normalize else 1
            
            pred, loss = dict_cont_loss(x,labels, emb_dict, self.patch, temperature=temp, norm=normalize)
            loss_dict['dict_loss'] = loss.item()
        
        if hasattr(self,'optim_step'):
            self.optim_step(self, loss) #local weight updates

        if emb_dict is not None and out_pred:
            pred = pred.argmax(1)
            return x.detach(), pred, loss, loss_dict
        else:
            pred = 0

        return x.detach(), pred, loss, loss_dict
    
#for the final layer of non-BP methods
class LocalClfLayer(nn.Module):
    def __init__(
        self,
        module,
        init_function,
        loss = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.module = module
        self.emb_params=[]
        if init_function is not None:
            init_function(self)
        self.loss = loss

    def forward(self, latent, target, inference=False, output_pred = False):
        
        #detach to prevent BP to previous layers
        x = self.module(latent.detach())

        if inference:
            return x.detach(), 0
        
        loss = self.loss(x, target)
        
        if hasattr(self,'optim_step'):
            self.optim_step(self, loss)
        
        if output_pred:
            pred = torch.argmax(x,dim=1)
            return pred, loss
        
        return x.detach(), loss


#https://github.com/facebookresearch/ConvNeXt/blob/64aba990d8477fa920685a694df65bdeb302c905/object_detection/mmdet/models/backbones/convnext.py
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel=3, stride=1,residual=False, prenorm=False, eql=False,bn=True, leaky=False,dropout=0,global_pool=False, prepool=True, method='BP'):
        super().__init__()
        
        pad = kernel//2

        conv = nn.Conv2d 
        self.conv = conv(in_planes, planes, kernel_size=kernel,stride=stride,padding=pad,bias=True)
        if method == 'FA':
            self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape, stride=stride, padding=pad)
        norm_dim = in_planes if prenorm else planes
        self.bn = nn.BatchNorm2d(norm_dim) if bn else nn.Identity()
        if bn:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        self.residual = residual
        self.act = F.leaky_relu if leaky else nn.ReLU()#F.relu
        self.pool = nn.MaxPool2d(2,2) if global_pool else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.prepool = prepool
        self.prenorm = prenorm


    def forward(self, x):
        if self.prenorm:
            out = self.conv(self.bn(x))
        else:
            out = self.bn(self.conv(x))
        if self.residual:
            out = self.act(x + out)
        else:
            out = self.act(out)
        if self.prepool:
            return self.dropout(self.pool(out))
        else:
            return self.pool(self.dropout(out))
        
class FCBlock(nn.Module):
    def __init__(self, in_planes, planes,dropout=0, prenorm=True, identity=False, ch_mixer=False,residual=False,leaky=False,bn=False,method='BP'):
        super().__init__()
        
        self.fc = nn.Linear(in_planes, planes)
        if method == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        norm = nn.BatchNorm1d if bn else nn.LayerNorm
        norm_dim = in_planes if prenorm else planes
        self.bn = norm(norm_dim) if not identity else nn.Identity() 
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.act = F.leaky_relu if leaky else F.relu
        self.ch_mixer=ch_mixer
        self.prenorm = prenorm

    def forward(self, x):
        if self.ch_mixer:
            x = x.transpose(1,2)
        if self.prenorm:
            out = self.fc(self.bn(x))
        else:
            out = self.bn(self.fc(x))
        if self.residual:
            out = self.dropout(self.act(x + out))
        else:
            out = self.dropout(self.act(out))
        if self.ch_mixer:
            out = out.transpose(1,2)
        return out
    
class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args,**kwargs):
        if 'BP' != self.method:
            return self.forward_F(*args,**kwargs)
        return self.forward_BP(*args,**kwargs)
    
    def forward_BP(self,x,return_feature=None):
        
        out = x

        for name in self.module_list:
            net = getattr(self, name)            
            out = net(out)
            if return_feature == name:
                return out

        if hasattr(self,'linear'):
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out  = self.linear(out)

        return out
        
    def forward_F(self, x, labels, inference=False, return_feature=None):
        
        loss_dict = {}
        out = x
        total_loss = 0
        for name in self.module_list:
            net = getattr(self, name)
            out_pred = False
            emb_dict = None
            pred = None
            
            if hasattr(self,'embedding'):
                if 'DCL' in self.method:
                    emb_dict = self.embedding.weight
                    if name == self.module_list[-1]:
                        out_pred = True
                    if 'DCL-S' in self.method:
                        emb_dict = emb_dict.detach()
                
            infer = inference and emb_dict is None
            out, pred, loss, local_loss_dict = net(out, labels, x, inference=infer, emb_dict=emb_dict,out_pred=out_pred)
            total_loss+=loss
            if not inference:
                loss_dict[name] = loss.item()
                for i in local_loss_dict:
                    loss_dict[f'{name}_{i}'] = local_loss_dict[i]
            if return_feature == name:
                return out

        if hasattr(self,'linear'):
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)

            out, loss = self.linear(out,labels,inference=inference,output_pred=return_feature!='linear')
            total_loss+=loss
            if not inference:
                loss_dict['final'] = loss.item()
       
        return out, pred, total_loss, loss_dict
        

class SimpleNet(BaseNet):
    def __init__(self, in_dim, out_dim, num_classes,
                 local_optimizer,
                 linear,
                 dropout,
                 temp=0.07,
                 method='BP'):
        
        super().__init__()

        conv = ConvBlock 
        self.dropout = nn.Dropout(dropout)
        prenorm=False
        
        setattr(self, 'layer64', conv(in_dim, 64,stride=1,bn=not prenorm,prenorm=prenorm,dropout=dropout,method=method))
        setattr(self,'layer256', conv(64,256, stride=2,prenorm=prenorm,dropout=dropout,method=method))
        module_list = ['layer64', 'layer256']
        
        if out_dim == 512:
            setattr(self,'layer512', conv(256,512, stride=2,dropout=dropout,method=method))
            module_list.append('layer512')
        self.max_dim = out_dim
        
        if 'DCL' in method: 
            num_emb=num_classes
            self.embedding = nn.Embedding(num_emb,self.max_dim)
            scale= np.sqrt(23.12) if out_dim == 512 else np.sqrt(16.52)
            if 'ORTH' in method:
                self.embedding.weight.data.copy_(scale*nn.init.orthogonal_(torch.empty(num_emb,self.max_dim)))
                print('orthogonal init')
            
        self.linear = nn.Linear(self.max_dim, num_classes)
        if local_optimizer is not None: self.linear = LocalClfLayer(self.linear,local_optimizer)
        
        if local_optimizer is not None:
            for i, name in enumerate(module_list):
                emb_params = self.embedding.parameters() if hasattr(self, 'embedding') else None
                net = SelfLocalLayer(getattr(self, name), local_optimizer,emb_params=emb_params,temp=temp,method=method)
                setattr(self, name, net)

        self.module_list = module_list
        self.no_linear = not linear
        self.method = method
    
class VGG(BaseNet):
    def __init__(self, num_classes,dropout=0, img_size=32,
                 local_optimizer=None,arch='vgg8b',temp=0.07,alpha=1,
                 method='BP'):
        
        super().__init__()
        self.max_dim=512
        conv = ConvBlock 
        self.no_linear = False
        if 'vgg8b' in arch:
            #[128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
            if img_size==96:
                setattr(self,'layer128', conv(3,128,kernel=7, stride=2,dropout=dropout,global_pool=False,method=method))
            elif img_size==28:
                setattr(self,'layer128', conv(1,128,stride=1,dropout=dropout,global_pool=False,method=method))
            else:
                setattr(self,'layer128', conv(3,128, stride=1,dropout=dropout,global_pool=False,method=method))
            setattr(self,'layer256_0', conv(128,256, stride=1,dropout=dropout,global_pool=True,method=method))
            setattr(self,'layer256_1', conv(256,256, stride=1,dropout=dropout,global_pool=False,method=method))
            setattr(self,'layer512_0', conv(256,512, stride=1,dropout=dropout,global_pool=True,method=method))
            setattr(self,'layer512_1', conv(512,512, stride=1,dropout=dropout,global_pool=True,method=method))      
            setattr(self,'layer512_2', conv(512,512, stride=1,dropout=dropout,global_pool=True,method=method))
            module_list = ['layer128', 'layer256_0','layer256_1','layer512_0','layer512_1','layer512_2']
            
        num_emb = num_classes
        if 'DCL' in method: 
            self.embedding = nn.Embedding(num_emb,self.max_dim)
            scale= np.sqrt(23.12)
            if 'ORTH' in method:
                self.embedding.weight.data.copy_(scale*nn.init.orthogonal_(torch.empty(num_emb,self.max_dim)))
                print('orthogonal init')

        #self.
        if img_size == 96:
            FC_in_dim = 4608
        elif img_size==28:
            FC_in_dim = self.max_dim
        else:
            FC_in_dim = self.max_dim*4
        
        self.linear = FCBlock(FC_in_dim, self.max_dim*2,prenorm=False,dropout=dropout,bn=True,method=method)
        self.final = nn.Linear(self.max_dim*2,num_classes)
        
        if local_optimizer is not None: 
            emb_params = self.embedding.parameters() if hasattr(self, 'embedding') else None
            self.linear = SelfLocalLayer(self.linear,local_optimizer,patch=2,emb_params=emb_params,temp=temp, method=method)
            self.final = LocalClfLayer(self.final,local_optimizer)

        if local_optimizer is not None:
            for i, name in enumerate(module_list):
                emb_params = self.embedding.parameters() if hasattr(self, 'embedding') else None
                net = SelfLocalLayer(getattr(self, name), local_optimizer,emb_params=emb_params,temp=temp, method=method)
                setattr(self, name, net)
        module_list.append('linear')
        self.module_list = module_list
        self.method = method

    def forward_F(self, x, labels, inference=False, return_feature=None):
        
        loss_dict = {}
        out = x
        total_loss = 0
        for name in self.module_list:
            net = getattr(self, name)
            out_pred = False
            emb_dict = None

            if name == 'linear':
                out = out.flatten(1)
                out = out.view(out.size(0), -1)

            if hasattr(self, 'embedding'):
                emb_dict = self.embedding.weight

                if name == self.module_list[-1]:
                    out_pred = True
                if 'DCL-S' in self.method:
                    emb_dict = emb_dict.detach()

            infer = inference# and emb_dict is None
            out, pred, loss, local_loss_dict = net(out, labels,x, inference=infer, emb_dict=emb_dict,out_pred=out_pred)
            total_loss+=loss

            if not inference:
                loss_dict[name] = loss.item()
                for i in local_loss_dict:
                    loss_dict[f'{name}_{i}'] = local_loss_dict[i]
            if return_feature == name:
                return out

        if hasattr(self,'final'):
            out, loss = self.final(out,labels,inference=inference,output_pred=return_feature!='final')
            total_loss+=loss
            if not inference:
                loss_dict['final'] = loss.item()

        return out, pred, total_loss, loss_dict
    

    def forward_BP(self,x,return_feature=None):
        
        out = x

        for name in self.module_list:
            net = getattr(self, name)
            if name == 'linear':
                out = out.flatten(1)
                out = out.view(out.size(0), -1)            
            out = net(out)
            if return_feature == name:
                return out

        if hasattr(self,'final'):
            out  = self.final(out)
        
        return out

class FCNet(BaseNet):
    def __init__(self, 
                start_dim,
                out_dim, 
                num_layers=3,
                num_classes=10,
                local_optimizer=None,
                prenorm=True,
                bn=False,
                dropout=0,
                patch=0,
                linear=True,
                temp=0.07,
                method='FL',
                ):
        super().__init__()

        self.no_linear = not linear
        num_layers-=1*self.no_linear
        self.num_layers=num_layers
        self.method=method

        if patch == 0:
            if start_dim%6==0:
                patch = 12#12#6#1
            elif start_dim%4==0:
                patch = 8#4

        if 'DCL' in method: 
            self.embedding = nn.Embedding(num_classes,out_dim//patch)
            scale = np.sqrt(55.76)

            if 'ORTH' in method:
                scale= np.sqrt(out_dim//patch)
                self.embedding.weight.data.copy_(scale*nn.init.orthogonal_(torch.empty(num_classes,out_dim//patch)))
                print('orthogonal init')

        self.patch = patch
        fc_dropout = dropout
        for i in range(num_layers):
            if i == num_layers-1 and linear:
                out_dim = num_classes
                fc_dropout = 0            
            setattr(self, f'layer_{i}', FCBlock(start_dim,out_dim,fc_dropout,identity=i==0,prenorm=prenorm,bn=bn,method=method))
            start_dim = out_dim
                    
        if 'BP' != method:
            for i in range(num_layers):
                
                name =  f'layer_{i}'
                module = getattr(self, name)
                if i < num_layers-1 or self.no_linear:
                    emb_params=self.embedding.parameters() if hasattr(self,'embedding') else None
                    net = SelfLocalLayer(module,local_optimizer,patch=patch,emb_params=emb_params,temp=temp,method=method)
                else:
                    net = LocalClfLayer(module, local_optimizer)

                setattr(self, name, net)
        
    def forward_BP(self,x,return_feature=None):
        
        out = x.flatten(1)

        for i in range(self.num_layers):
            name = f'layer_{i}'
            net = getattr(self, name)            
            out = net(out)
            
            if return_feature == name:
                return out

        return out

    def forward_F(self, x, labels, inference=False, return_feature=None):
        
        loss_dict = {}
        total_loss = 0
        out = x.flatten(1)
        for i in range(self.num_layers):
            name = f'layer_{i}'
            net = getattr(self, name)
            emb_dict=None
            
            if 'DCL' in self.method:
                emb_dict = self.embedding.weight
                if '-S' in self.method:
                    emb_dict = emb_dict.detach()
            
            infer = inference# and emb_dict is None
            
            if i == self.num_layers-1 and not self.no_linear:
                    out, loss =  net(out, labels, inference=infer, output_pred = True)
                    loss_dict['final'] = loss.item()
            else:
                out, pred, loss,local_loss_dict = net(out, labels,x, inference=infer, emb_dict=emb_dict,out_pred=i == self.num_layers-2)
                for i in local_loss_dict:
                    loss_dict[f'{name}_{i}'] = local_loss_dict[i]

            total_loss+=loss

            if return_feature == name:
                out = rearrange(out,"b (p c) -> p b c", p=self.patch)
                out=out.mean(0)
                return out
        

        return out, pred, total_loss,loss_dict
    