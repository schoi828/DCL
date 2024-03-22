arch=$1     
data=$2   ## ##MNIST, CIFAR10,CIFAR100            ##training hyperparameters are automatically preset for each dataset 
method=$3    #BP=Byackpropagation, FL_FEAT=Feature contrastive loss, ASY_FL_DICT=Dictionary Contrastive loss
name='name'
seed=0 
gpu=0
num_workers=4

python -O train.py --exp_name $name \
    --method $method \
    --num_workers $num_workers \
    --data $data \
    --gpu $gpu \
    --seed $seed \
    --arch $arch \
    --linear --cuda_setting --save_dir ../exps --pre_config --print_memory