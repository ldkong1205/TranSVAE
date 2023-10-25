#!/bin/bash

# Jester_src -> Jester_tar
python3 train_TransferVAE_jester.py \
    --epochs 1000 \
    --gpu 1 \
    --num_segments 6 \
    --tar_psuedo_thre 0.95 \
    --optimizer Adam \
    --lr 0.001 \
    --batch_size 128 \
    --use_psuedo 'Y' \
    --weight_MI 0.001 \
    --weight_cls 10 \
    --weight_triplet 100 \
    --weight_adv 10 \
    --start_psuedo_step 100 \
    --backbone 'I3Dpretrain' \
    --exp_dir 'experiments_I3D' &
