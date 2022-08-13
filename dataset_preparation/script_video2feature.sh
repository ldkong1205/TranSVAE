#!/bin/bash
# ----------------------------------------------------------------------------------
# variable
data_path=../dataset/hmdb51/ # depend on users
#data_path=../dataset/ucf101/ # depend on users
video_in=RGB
feature_in=I3D-feature_pretrain
input_type=video # video | frames
structure=tsn # tsn | imagenet
num_thread=2
batch_size=16 # need to be larger than 16 for c3d
base_model=i3d # resnet101 | c3d | i3d
pretrain_weight=../models/rgb_imagenet.pt # depend on users (only used for C3D model)
start_class=1 # start from 1
end_class=-1 # -1: process all the categories
#class_file=../data/ucf101_splits/class_list_hmdb_ucf.txt # none | XXX/class_list_DA.txt (depend on users)
class_file=../data/hmdb51_splits/class_list_hmdb_ucf.txt # none | XXX/class_list_DA.txt (depend on users)

python3 video2I3D.py --data_path $data_path --video_in $video_in \
--feature_in $feature_in --input_type $input_type --structure $structure \
--num_thread $num_thread --batch_size $batch_size --base_model $base_model --pretrain_weight $pretrain_weight \
--start_class $start_class --end_class $end_class --class_file $class_file

#----------------------------------------------------------------------------------
exit 0
