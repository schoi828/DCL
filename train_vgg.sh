
batch_size=128
LR=0.0005        
EPOCHS=400      
optim=AdamW
dropout=0.05 
arch=vgg8b 
data=MNIST         ##MNIST, fMNIST, CIFAR10,CIFAR100,SVHN,STL10
MS_GAMMA=0.25      ##training hyperparameters are automatically preset for each dataset    
method=asy_fl_dict    #BP, FL_FEAT, ASY_FL_DICT  #BP=Backpropagation, FL_FEAT=Feature contrastive loss, ASY_FL_DICT=Dictionary Contrastive loss
name='name'
seed=0 
gpu=0
num_workers=4

python -O train.py --exp_name $name \
    --epoch $EPOCHS \
    --method $method \
    --lr $LR \
    --num_workers $num_workers \
    --data $data \
    --gpu $gpu \
    --seed $seed \
    --dropout $dropout \
    --batch_size $batch_size \
    --arch $arch \
    --MS_gamma $MS_GAMMA --milestones 200 300 350 375  \
    --linear --print_memory --cuda_setting --pre_config 