#CUDA_VISIBLE_DEVICES="0,1,2,3"
batch_size=512
LR=0.0002        
EPOCHS=400      
num_layers=3
max_ch=3072
start_ch=64
step_size=0
optim=AdamW
dropout=0  
arch=simple     #simple for Conv .  fc for FC
data=CIFAR100   ## ##MNIST, CIFAR10,CIFAR100            ##training hyperparameters are automatically preset for each dataset 
MS_GAMMA=0.5 
method=ASY_FL_DICT    #BP=Byackpropagation, FL_FEAT=Feature contrastive loss, ASY_FL_DICT=Dictionary Contrastive loss
patch=48   #
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
    --start_dim $start_ch \
    --max_dim $max_ch \
    --num_layers $num_layers \
    --dropout $dropout \
    --batch_size $batch_size \
    --arch $arch \
    --MS_gamma $MS_GAMMA --milestones 0  \
    --patch_fc $patch  --linear --cuda_setting --save_dir FCONV --pre_config