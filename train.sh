#!/bin/bash


model="miigan"  # or infragan or pix2pix or thermal_gan or cycle_gan
name="MIIGAN_VEDAI_512"  # Custom checkpoint file directory
which_model_netG="MIIGANGenerator"
# - Generator
#    -- infragan  ->  unet_512
#    -- pix2pix   ->  unet_256
#    -- thermal_gan  ->  unet_512
#    -- cycle_gan  ->  unet_512

which_model_netD="MIIGANDiscriminator"
# - Discriminator
#    -- infragan  ->  unetdiscriminator
#    -- pix2pix   ->  basic
#    -- thermal_gan  ->  basic
#    -- cycle_gan  ->  basic

dataset_mode="VEDAI"  # The code has been modified, and all datasets now default to using VEDAI.
dataroot="./datasets/VEDAI_512"   #  Dataset path

which_direction="AtoB"
input_nc=3
output_nc=1
lambda_A=100
no_lsgan=""
norm="batch"
pool_size=0
loadSize=512
fineSize=512
gpu_ids="0"
nThreads=1
batchSize=1

# python -m visdom.server

python train.py \
    --dataset_mode $dataset_mode \
    --dataroot $dataroot \
    --name $name \
    --model $model \
    --which_model_netG $which_model_netG \
    --which_model_netD $which_model_netD \
    --which_direction $which_direction \
    --input_nc $input_nc \
    --output_nc $output_nc \
    --lambda_A $lambda_A \
    --no_lsgan  \
    --norm $norm \
    --pool_size $pool_size \
    --loadSize $loadSize \
    --fineSize $fineSize \
    --gpu_ids $gpu_ids \
    --nThreads $nThreads \
    --batchSize $batchSize