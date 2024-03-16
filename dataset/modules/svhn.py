import os
from typing import Optional
import torch

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

from dataset.modules.common import channels_to_last, ImagePreprocessor, lift_transform
from dataset.modules.base import DataModule


class SVHNPreprocessor(ImagePreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = False):
        super().__init__(svhn_transform(normalize, channels_last))


class SVHNDataModule(DataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", "svhn"),
        normalize: bool = True,
        channels_last: bool = False,
        random_crop: Optional[bool] = True,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        classes_per_batch:int = 0,
        nprocs=None
    ):
        super().__init__(dataset_dir,
        normalize,
        channels_last,
        random_crop,
        batch_size,
        num_workers,
        pin_memory,
        shuffle,
        classes_per_batch,
        nprocs)
        
        crop = 32 if random_crop else None

        self.tf_train = svhn_transform(normalize, channels_last,  random_crop=crop)
        self.tf_valid = svhn_transform(normalize, channels_last, random_crop=None)

        self.ds_train = None
        self.ds_valid = None
        self.cpb = classes_per_batch

    @property
    def num_classes(self):
        return 10

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 32, 32, 3
        else:
            return 3, 32, 32

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = torch.utils.data.ConcatDataset((
        datasets.SVHN('../data/SVHN', split='train', download=True, transform=self.tf_train),
        datasets.SVHN('../data/SVHN', split='extra', download=True, transform=self.tf_train)))

        self.ds_valid = datasets.SVHN('../data/SVHN', split='test', download=True,transform=self.tf_valid)


def svhn_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop,padding=4))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199)))
        #transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)
