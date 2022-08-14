import os
import imageio.v2 as imageio
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
import pickle


batch_size = 16


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


from pytorch_i3d import InceptionI3d
model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load('../models/rgb_imagenet.pt'))
extractor = torch.nn.DataParallel(model.cuda())
extractor.eval()

path_input = '../dataset/epic-kitchens/rgb/'
path_output = '../dataset/epic-kitchens/I3D-feature-pretrain/'
csv_file = '../dataset/epic-kitchens/D3_test.pkl'
list_path = '../dataset/epic-kitchens/list/'
with open(csv_file, 'rb') as f:
    dataset_pd = pickle.load(f)
uid = dataset_pd["uid"].to_numpy()
start_frame = dataset_pd["start_frame"].to_numpy()
stop_frame = dataset_pd["stop_frame"].to_numpy()
video_id = dataset_pd["video_id"].to_numpy()
verb_class = dataset_pd["verb_class"].to_numpy()
    
if csv_file[:-4].endswith("train"):
    path_output += "train/"
    path_input += "train/"
    mode = 'train'
else: 
    path_output += "test/"
    path_input += "test/"
    mode = 'test'
    
stripped_csv_file = csv_file.split("/")[-1]
if stripped_csv_file.startswith("D1"):
    path_output += "P08/"
    path_input += "P08/"
    domain = 'P08'
if stripped_csv_file.startswith("D2"):
    path_output += "P01/"
    path_input += "P01/"
    domain = 'P01'
if stripped_csv_file.startswith("D3"):
    path_output += "P22/"
    path_input += "P22/"
    domain = 'P22'

feature_in_type = '.t7'

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

start = time.time()
path_list = []
label_list = []
save_list = []
for i in range(len(video_id)):
    path_list.append(path_input + video_id[i] + '/')
    save_list.append(path_output + video_id[i] + '/')
    label_list.append(verb_class[i]) 

list_file = list_path + 'list_%s_%s.txt' % (domain, mode)
file1 = open(list_file, "w")

for j in range(len(path_list)):
    rgb_files = [i for i in os.listdir(path_list[j])]
    rgb_files.sort()
    list_frames = rgb_files[start_frame[j]:stop_frame[j]]
    save_path = save_list[j] + str(j)
    list_content = save_list[j][3:] + str(j) + ' ' + str(len(list_frames)) + ' ' + str(label_list[j]) + '\n'
    file1.write(list_content)
   
    
    if not os.path.isdir(save_list[j] + str(j)):
        os.makedirs(save_list[j] + str(j))

    frames_tensor = []
    try:
        for t in range(len(list_frames)):
            im = imageio.imread(path_list[j] + '/' + list_frames[t])
            if np.sum(im.shape) != 0:
                id_frame = t+1
                frames_tensor.append(im2tensor(im))
    except RuntimeError:
        print('Could not read frame', id_frame+1, 'from', video_file)
    
    num_frames = len(frames_tensor)
    
    num_numpy = int(batch_size/2) 
    for i in range(num_numpy):
        frames_tensor = [torch.zeros_like(frames_tensor[0])] + frames_tensor
        frames_tensor.append(torch.zeros_like(frames_tensor[0]))

    features = torch.Tensor()

    for t in range(0, num_frames, 1):
        frames_batch = frames_tensor[t:t+batch_size]    
        features_batch = extract_frame_feature_batch(frames_batch)
        features = torch.cat((features,features_batch))

    for t in range(features.size(0)):
        id_frame = t+1
        id_frame_name = str(id_frame).zfill(5)
        filename =save_path + '/img_' + id_frame_name + feature_in_type            

        if not os.path.exists(filename):
            torch.save(features[t].clone(), filename)
    print('Save ' + save_path + ' done!')


end = time.time()
print('Total elapsed time: ' + str(end-start))