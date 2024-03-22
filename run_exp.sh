#architecture: simple256, simple512, fc, vgg8b
#dataset: MNIST, CIFAR10, CIFAR100, CIFAR20, SVHN, STL10, fMNIST, kMNIST
#method: BP, DCL, FEAT
#                architecture    dataset     method   exp_name   gpu    seed
#bash train.sh    simple256       CIFAR100    BP       test       0      0     &
#bash train.sh    vgg8b       CIFAR100    BP       test       0      0     &
#bash train.sh    fc       CIFAR100    BP       test       0      0     &
#bash train.sh    simple256       CIFAR100    DCL       test       1      0     &
#bash train.sh    vgg8b       CIFAR100    DCL       test       1      0     &
#bash train.sh    fc       CIFAR100    DCL       test       1      0     &
#bash train.sh    simple512       CIFAR100    DCL-S       test       0      0     &
#bash train.sh    vgg8b       CIFAR100    DCL-S       test       0      0     &
#bash train.sh    fc       CIFAR10    DCL-S       test       0      0     &
#bash train.sh    simple512       CIFAR100    FEAT       test       1      0     &
#bash train.sh    vgg8b       STL10    FEAT       test       1      0     &
bash train.sh    fc       MNIST    FEAT       test       1      0     &
