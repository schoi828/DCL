import os
from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

from dataset.modules.common import channels_to_last, ImagePreprocessor, lift_transform
from dataset.modules.base import DataModule

class kMNISTPreprocessor(ImagePreprocessor):
    def __init__(self, normalize: bool = True, channels_last: bool = False):
        super().__init__(kmnist_transform(normalize, channels_last))

class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]

class kMNISTDataModule(DataModule):
    def __init__(
        self,
        dataset_dir: str = os.path.join(".cache", "kmnist"),
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

        crop = 28 if random_crop else None

        self.tf_train = kmnist_transform(normalize, channels_last, random_crop=crop)
        self.tf_valid = kmnist_transform(normalize, channels_last, random_crop=None)

        self.ds_train = None
        self.ds_valid = None
        self.cpb = classes_per_batch

    @property
    def num_classes(self):
        return 10

    @property
    def image_shape(self):
        if self.hparams.channels_last:
            return 28, 28, 1
        else:
            return 1, 28, 28


    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = KuzushijiMNIST('../data/KuzushijiMNIST', train=True, download=True, transform=self.tf_train)
        self.ds_valid = KuzushijiMNIST('../data/KuzushijiMNIST', train=False, download=True, transform=self.tf_valid)

def kmnist_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    if random_crop is not None:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize((0.1904,), (0.3475,)))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)