import os
from typing import Optional
import torch
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from dataset.modules.common import channels_to_last, lift_transform

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", ""),
        normalize: bool = True,
        channels_last: bool = False,
        random_crop: Optional[bool] = True,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        classes_per_batch:int = 0,
        nprocs:tuple=(1,0),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.channels_last = channels_last

    def train_dataloader(self):
        batch_size = self.hparams.batch_size
        sampler = None
        nprocs, rank = self.hparams.nprocs
        if nprocs > 1:
            batch_size = batch_size // nprocs
            sampler = DistributedSampler(self.ds_train,num_replicas=nprocs,rank=rank,shuffle=self.hparams.shuffle)
            print('distributed train sampler initialized')
        return DataLoader(
            self.ds_train,
            shuffle=self.hparams.shuffle and sampler is None,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
        )

    def val_dataloader(self,shuffle=False):
        batch_size = self.hparams.batch_size
        sampler = None
        nprocs, rank = self.hparams.nprocs
        if nprocs > 1:
            batch_size = batch_size // nprocs
            sampler = DistributedSampler(self.ds_valid,num_replicas=nprocs,rank=rank,shuffle=shuffle)
            print('distributed val sampler initialized')

        return DataLoader(
            self.ds_valid,
            shuffle=shuffle and sampler is None,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
        )
