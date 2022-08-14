from pytorch_i3d import InceptionI3d
import argparse
import imageio
import os
import re
import time
from colorama import init
from colorama import Fore, Back
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

init(autoreset=True)


parser = argparse.ArgumentParser(description='Dataset Preparation')
parser.add_argument('--data_path', type=str, required=False,
                    default='', help='source path')
parser.add_argument('--video_in', type=str, required=False,
                    default='RGB', help='name of input video dataset')
parser.add_argument('--feature_in', type=str, required=False,
                    default='RGB-feature', help='name of output frame dataset')
parser.add_argument('--input_type', type=str, default='video',
                    choices=['video', 'frames'], help='input types for videos')
parser.add_argument('--structure', type=str, default='tsn',
                    choices=['tsn', 'imagenet'], help='data structure of output frames')
parser.add_argument('--base_model', type=str, required=False, default='resnet101',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'c3d', 'i3d'])
parser.add_argument('--pretrain_weight', type=str,
                    required=False, default='', help='model weight file path')
parser.add_argument('--num_thread', type=int, required=False,
                    default=-1, help='number of threads for multiprocessing')
parser.add_argument('--batch_size', type=int,
                    required=False, default=1, help='batch size')
parser.add_argument('--start_class', type=int, required=False,
                    default=1, help='the starting class id (start from 1)')
parser.add_argument('--end_class', type=int, required=False,
                    default=-1, help='the end class id')
parser.add_argument('--class_file', type=str, default='class.txt',
                    help='process the classes only in the class_file')
args = parser.parse_args()


max_thread = 8
num_thread = args.num_thread if args.num_thread > 0 and args.num_thread <= max_thread else max_thread
print(Fore.CYAN + 'thread #:', num_thread)
pool = ThreadPool(num_thread)


path_input = args.data_path + args.video_in + '/'
feature_in_type = '.t7'


path_output = args.data_path + args.feature_in + '_' + args.base_model + '/'
if args.structure != 'tsn':
    path_output = args.data_path + args.feature_in + '-' + args.structure + '/'
if not os.path.isdir(path_output):
    os.makedirs(path_output)


print(Fore.GREEN + 'Pre-trained model:', args.base_model)

model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load(args.pretrain_weight))
extractor = torch.nn.DataParallel(model.cuda())
extractor.eval()

cudnn.benchmark = True
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if args.class_file == 'none':
    class_names_proc = ['unlabeled']
else:
    class_names_proc = [line.strip().split(' ', 1)[1]
                        for line in open(args.class_file)]
    print(class_names_proc)


def im2tensor(im):
    im = Image.fromarray(im)
    t_im = data_transform(im)
    return t_im


def extract_frame_feature_batch(list_tensor):
    with torch.no_grad():
        batch_tensor = torch.stack(list_tensor)
        batch_tensor = torch.transpose(batch_tensor, 0, 1)

        batch_tensor = batch_tensor.unsqueeze(0)
        features = extractor(batch_tensor)

        features = features.view(features.size(0), -1).cpu()
        return features


def extract_features(video_file):
    video_name = os.path.splitext(video_file)[0]
    video_name = re.sub('[),(,&]', '', video_name)
    if args.structure == 'tsn':
        if not os.path.isdir(path_output + video_name + '/'):
            os.makedirs(path_output + video_name + '/')

    num_exist_files = len(os.listdir(path_output + video_name + '/'))

    frames_tensor = []
    if args.input_type == 'video':
        reader = imageio.get_reader(path_input + class_name + '/' + video_file)

        try:
            for t, im in enumerate(reader):
                if np.sum(im.shape) != 0:
                    id_frame = t+1
                    frames_tensor.append(im2tensor(im))
        except RuntimeError:
            print(Back.RED + 'Could not read frame',
                  id_frame+1, 'from', video_file)
    elif args.input_type == 'frames':
        list_frames = os.listdir(path_input + class_name + '/' + video_file)
        list_frames.sort()

        try:
            for t in range(len(list_frames)):
                im = imageio.imread(
                    path_input + class_name + '/' + video_file + '/' + list_frames[t])
                if np.sum(im.shape) != 0:
                    id_frame = t+1
                    frames_tensor.append(im2tensor(im))
        except RuntimeError:
            print(Back.RED + 'Could not read frame',
                  id_frame+1, 'from', video_file)

    num_frames = len(frames_tensor)
    if num_frames == num_exist_files:
        return

    num_numpy = int(args.batch_size/2)
    for i in range(num_numpy):
        frames_tensor = [torch.zeros_like(
            frames_tensor[args.batch_size])] + frames_tensor
        frames_tensor.append(torch.zeros_like(frames_tensor[args.batch_size]))

    features = torch.Tensor()

    for t in range(0, num_frames, 1):
        frames_batch = frames_tensor[t:t+args.batch_size]
        features_batch = extract_frame_feature_batch(frames_batch)
        features = torch.cat((features, features_batch))

    for t in range(features.size(0)):
        id_frame = t+1
        id_frame_name = str(id_frame).zfill(5)
        if args.structure == 'tsn':
            filename = path_output + video_name + '/' + \
                'img_' + id_frame_name + feature_in_type
        elif args.structure == 'imagenet':
            filename = path_output + class_name + '/' + \
                video_name + '_' + id_frame_name + feature_in_type
        else:
            raise NameError(Back.RED + 'not valid data structure')

        if not os.path.exists(filename):
            torch.save(features[t].clone(), filename)


list_class = os.listdir(path_input)
list_class.sort()

id_class_start = args.start_class-1
id_class_end = len(list_class) if args.end_class <= 0 else args.end_class
start = time.time()

for i in range(id_class_start, id_class_end):
    start_class = time.time()
    class_name = list_class[i]
    if class_name in class_names_proc:
        print(Fore.YELLOW + 'class ' + str(i+1) + ': ' + class_name)

        if args.structure == 'imagenet':
            if not os.path.isdir(path_output + class_name + '/'):
                os.makedirs(path_output + class_name + '/')

        list_video = os.listdir(path_input + class_name + '/')
        list_video.sort()

        pool.map(extract_features, list_video, chunksize=1)

        end_class = time.time()
        print('Elapsed time for ' + class_name +
              ': ' + str(end_class-start_class))
    else:
        print(Fore.RED + class_name + ' is not selected !!')

end = time.time()
print('Total elapsed time: ' + str(end-start))
print(Fore.GREEN + 'All the features are generated for ' + args.video_in)
