#!/bin/bash


model="miigan"
name="MIIGAN_VEDAI_512"
which_model_netG="MIIGANGenerator"
which_model_netD="MIIGANDiscriminator"
which_epoch="best_196"
dataset_mode="VEDAI"
dataroot="./datasets/VEDAI"

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

python test.py \
    --dataset_mode $dataset_mode \
    --dataroot $dataroot \
    --name $name \
    --model $model \
    --which_model_netG $which_model_netG \
    --which_model_netD $which_model_netD \
    --which_epoch $which_epoch \
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