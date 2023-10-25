#!/bin/bash

# H -> U
python3 train_TransferVAE_hmdb_ucf.py \
    --src 'hmdb51' \
    --tar 'ucf101' \
    --epochs 1000 \
    --gpu 0 \
    --num_segments 9 \
    --tar_psuedo_thre 0.96 \
    --optimizer Adam \
    --lr 0.001 \
    --batch_size 128 \
    --use_psuedo 'Y' \
    --weight_MI 0.5 \
    --weight_cls 1 \
    --weight_triplet 10 \
    --weight_adv 0.1 \
    --start_psuedo_step 100 \
    --backbone 'I3Dpretrain' \
    --exp_dir 'experiments_I3D' &