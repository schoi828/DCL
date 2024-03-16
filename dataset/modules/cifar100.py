import os
from typing import Optional
import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as tv_datasets
import torch
from dataset.modules.common import channels_to_last, ImagePreprocessor, lift_transform
from dataset.modules.cifar10 import CIFAR10DataModule
from dataset.modules.base import DataModule

class CIFAR100Preprocessor(ImagePreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = False):
        super().__init__(cifar100_transform(normalize, channels_last))


class CIFAR100DataModule(DataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", "cifar10"),
        normalize: bool = True,
        channels_last: bool = False,
        random_crop: Optional[bool] = False,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        classes_per_batch:int = 0,
        nprocs=None
    ):
        super().__init__(dataset_dir=dataset_dir,
        normalize=normalize,
        channels_last=channels_last,
        random_crop=random_crop,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        classes_per_batch=classes_per_batch,
        nprocs=nprocs)
        
        crop = 32 if random_crop else None

        self.tf_train = cifar100_transform(normalize, channels_last, flip = True, random_crop=crop)
        self.tf_valid = cifar100_transform(normalize, channels_last, flip = False,random_crop=None)

        self.ds_train = None
        self.ds_valid = None
        self.cpb = classes_per_batch
    @property
    def num_classes(self):
        return 100
    
    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 32, 32, 3
        else:
            return 3, 32, 32

    def load_dataset(self, split: Optional[str] = None):
        return load_dataset("cifar100", split=split, cache_dir=self.hparams.dataset_dir)
            
    def load_tv_dataset(self,train,transform):
        return tv_datasets.CIFAR100('../data/CIFAR100', train=train, download=True, transform=transform)

    def prepare_data(self) -> None:
        self.load_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.classes_per_batch > 0:
            self.ds_train = self.load_tv_dataset(train=True,transform=self.tf_train)
        else:
            self.ds_train = self.load_dataset(split="train")
            self.ds_train.set_transform(lift_transform(self.tf_train))

        self.ds_valid = self.load_dataset(split="test")
        self.ds_valid.set_transform(lift_transform(self.tf_valid))

    def train_dataloader_cpb(self):
        batch_size = self.hparams.batch_size
        sampler = NClassRandomSampler(self.ds_train.targets, self.hparams.classes_per_batch, batch_size)
        print('NclassRandomSampler initialized')
        return DataLoader(
            self.ds_train,
            shuffle=self.hparams.shuffle and sampler is None,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
        )

        
def cifar100_transform(normalize: bool = True, channels_last: bool = True, flip = False, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop,padding=4))

    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010)))
        #transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)


class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.

    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch
        
        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))
        
        np.random.shuffle(ts_i)
        #algorithm outline: 
        #1) put n examples in batch
        #2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:] #pop n off the list
                
            t_slice_set = set([ts[i] for i in idxs])
            
            #fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n*10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1
            
            #fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j)) #pop is O(n), can we do better?
                else:
                    j += 1
            
            if len(idxs) < self.batch_size:
                needed = self.batch_size-len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]
                    
            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)

"""
class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.
    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.data_source = targets
        self.targets = targets['fine_label']
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch
        
        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))
        
        np.random.shuffle(ts_i)
        #algorithm outline: 
        #1) put n examples in batch
        #2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:] #pop n off the list
                
            t_slice_set = set([ts[i] for i in idxs])
            
            #fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n*10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1
            
            #fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j)) #pop is O(n), can we do better?
                else:
                    j += 1
            
            if len(idxs) < self.batch_size:
                needed = self.batch_size-len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]
                    
            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)
"""