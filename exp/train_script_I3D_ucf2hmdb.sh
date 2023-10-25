#!/bin/bash

# U -> H
python3 train_TransferVAE_hmdb_ucf.py \
    --src 'ucf101' \
    --tar 'hmdb51' \
    --epochs 1000 \
    --gpu 1 \
    --num_segments 8 \
    --tar_psuedo_thre 0.93 \
    --optimizer Adam \
    --lr 0.001 \
    --batch_size 128 \
    --use_psuedo 'Y' \
    --weight_MI 50 \
    --weight_cls 1 \
    --weight_triplet 1 \
    --weight_adv 1 \
    --start_psuedo_step 100 \
    --backbone 'I3Dpretrain' \
    --exp_dir 'experiments_I3D' &