import os, sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from option import get_option
import trainer as Trainer
from dataset.modules import (CIFAR10DataModule,CIFAR100DataModule, MNISTDataModule,
                            SVHNDataModule,STL10DataModule,
                              fashionMNISTDataModule,kMNISTDataModule)

def main(option):

    option.device = "cuda:{}".format(option.gpu)
    torch.cuda.set_device(option.device)

    if not option.no_logging:
        from torch.utils.tensorboard import SummaryWriter
        # Define Tensorboard Writer
        tb_writer = SummaryWriter(log_dir=os.path.join(option.save_dir, option.exp_name))
    else:
        tb_writer = None
    dataset = {
        "MNIST": MNISTDataModule,
        "CIFAR10": CIFAR10DataModule,
        "CIFAR100": CIFAR100DataModule,
        "CIFAR100_C": CIFAR100DataModule,
        "SVHN": SVHNDataModule,
        "STL10": STL10DataModule,
        'fMNIST':fashionMNISTDataModule,
        'kMNIST':kMNISTDataModule
    }
    # Load Dataset
    datamodule = dataset[option.data]
    ngpus =  1
    dataset = datamodule(batch_size=option.batch_size, num_workers=option.num_workers,classes_per_batch=option.cpb,nprocs=(ngpus,option.gpu))
    dataset.setup()
    option.classes = dataset.num_classes
    option.img_shape = dataset.image_shape
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    cpb_loader = None if option.cpb == 0 else dataset.train_dataloader_cpb()
    # Define Trainer
    gpu = option.gpu 
    if 'ASY' in option.method:
        trainer = Trainer.Trainer_LayerwiseFL(option, gpu)  
    elif 'FL' in option.method:
        trainer = Trainer.Trainer_FL(option, gpu)
    else: # Baseline
        trainer = Trainer.Trainer_BP(option, gpu)
    trainer.tb_writer = tb_writer

    option._backend_setting(option.gpu)

    print(f"[START TRAINING] {option.data}")
    trainer.logger.info(' '.join(sys.argv))
    trainer.logger.info("Use Initialization of weights.")
    trainer.train_task(train_loader=train_loader, val_loader=val_loader,cpb_loader=cpb_loader)

    # Elapsed Time
    if option.local_rank == 0:
        trainer.logger.info(f"Elapsed Time : {(time.time() - start)/3600:.1f} hour")

if __name__ == "__main__":
    start = time.time()
    option = get_option()
    option._backend_setting()

    main(option)