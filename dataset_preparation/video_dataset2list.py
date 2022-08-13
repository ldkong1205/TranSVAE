import argparse
import os
import numpy as np
import random
import cv2
from colorama import init
from colorama import Fore

init(autoreset=True)


parser = argparse.ArgumentParser(description='Generate train/val splits from dataset ')
parser.add_argument('dataset', type=str,
                    help='ucf101 | hmdb51 | ps_train | ps_val | ra_train | ra_val | ps_unlabeled')
parser.add_argument('--class_select', action="store_true",
                    help='select some classes only')
parser.add_argument('--data_path', default='data/',
                    help='data path', type=str, required=False)
parser.add_argument('--video_in', default='RGB',
                    help='raw video folder name', type=str, required=False)
parser.add_argument('--frame_in', default='RGB-feature',
                    help='video frame/feature folder name', type=str, required=False)
parser.add_argument('--max_num', default=-1,
                    help='max number of training images/category (-1: all/0: avg #)', type=int, required=False)
parser.add_argument('--random_each_video', default='N', type=str,
                    choices=['Y', 'N'], help='randomly select videos for each video')
parser.add_argument('--method_read', default='video', type=str,
                    choices=['video', 'frame'], help='approach to load data')
parser.add_argument('--DA_setting', default='hmdb_ucf', type=str, choices=[
                    'hmdb_ucf', 'hmdb_phav', 'ps_kinetics', 'kinetics_phav', 'ucf_olympic'], help='datasets for DA')
parser.add_argument('--suffix', default=None,
                    help='additional string for filename', type=str, required=False)
args = parser.parse_args()


print(Fore.GREEN + 'dataset:', args.dataset)
path_frame_dataset = args.data_path + args.dataset + '/' + args.frame_in + '/'
list_video = os.listdir(path_frame_dataset)
list_video.sort()

path_video_dataset = args.data_path + args.dataset + '/' + args.video_in + '/'
list_class = os.listdir(path_video_dataset)
list_class.sort()


if args.dataset == 'hmdb51' or args.dataset == 'ucf101':
    if args.class_select:
        file_suffix = '_' + args.DA_setting
    else:
        file_suffix = '_full'

    class_file = '../data/' + args.dataset + \
        '_splits/class_list' + file_suffix + '.txt'

    class_id = [int(line.strip().split(' ', 1)[0])
                for line in open(class_file)]
    class_names = [line.strip().split(' ', 1)[1] for line in open(class_file)]

elif 'unlabeled' in args.dataset:
    class_id = [-1]
    class_names = ['unlabeled']

elif 'ps' in args.dataset or 'kinetics' in args.dataset or 'phav' in args.dataset or 'olympic' in args.dataset:
    name_dataset = args.dataset.split('_')[0]

    if args.class_select:
        file_suffix = '_' + args.DA_setting
    else:
        file_suffix = '_full'

    class_file = '../data/' + name_dataset + \
        '_splits/class_list' + file_suffix + '.txt'

    class_id = [int(line.strip().split(' ', 1)[0])
                for line in open(class_file)]
    class_names = [line.strip().split(' ', 1)[1] for line in open(class_file)]

else:
    raise ValueError('Unknown dataset '+args.dataset)


num_class = len(set(class_id))
list_class_video = [[] for i in range(num_class)]
num_class_video = np.zeros(num_class, dtype=int)


for i in range(len(class_names)):
    print(i, class_names[i])
    list_video = os.listdir(path_video_dataset + class_names[i])
    list_video.sort()
    list_video_name = [v.split('.')[0] for v in list_video]
    id_category = class_id[i]

    if args.method_read == 'video':
        lines_path = [path_frame_dataset + list_video_name[t] + ' ' + str(int(cv2.VideoCapture(path_video_dataset + class_names[i] + '/' + list_video_name[t] + '.mp4').get(
            cv2.CAP_PROP_FRAME_COUNT))) + ' ' + str(id_category) + '\n' for t in range(len(list_video))]

    elif args.method_read == 'frame':
        lines_path = [path_frame_dataset + list_video_name[t] + ' ' + str(len(os.listdir(
            path_frame_dataset + list_video_name[t]))) + ' ' + str(id_category) + '\n' for t in range(len(list_video))]

    list_class_video[id_category] = list_class_video[id_category] + lines_path
    num_class_video[id_category] += len(list_video)


num_avg_class = int(num_class_video.mean())
max_num = num_avg_class if args.max_num == 0 else args.max_num


if args.suffix:
    file = open(args.data_path + args.dataset + '/' + 'list_' +
                args.dataset + args.suffix + '.txt', 'w')
else:
    file = open(args.data_path + args.dataset + '/' +
                'list_' + args.dataset + '.txt', 'w')

print(args.video_in, ': ')


for i in range(len(list_class_video)):
    list_video_clips = list_class_video[i]
    num_videos = len(list_video_clips)
    full_list = range(num_videos)

    if args.random_each_video == 'Y':
        list_video_name = [j.rsplit('_', 2)[0] for j in list_video_clips]
        list_video_name = list(set(list_video_name))

        list_video_clips_nest = [[] for j in range(len(list_video_name))]
        list_video_id_nest = [[] for j in range(len(list_video_name))]

        for j in list_video_clips:
            id_unique = list_video_name.index(j.rsplit('_', 2)[0])
            id_single = list_video_clips.index(j)

            list_video_clips_nest[id_unique].append(j)
            list_video_id_nest[id_unique].append(id_single)

        select_list = []
        for j in list_video_id_nest:
            select_list += random.sample(j, max_num) if max_num > 0 and max_num < len(j) else j

    else:
        select_list = random.sample(
            full_list, max_num) if max_num > 0 and max_num < num_videos else full_list

    for j in select_list:
        file.write(list_class_video[i][j])

    print(i, len(full_list), '-->', len(select_list))

file.close()
