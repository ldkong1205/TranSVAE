import os
import imageio
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
import torchvision.transforms as transforms
import torch
import time


num_thread = 4
print('thread #:', num_thread)
pool = ThreadPool(num_thread)
batch_size = 16


def im2tensor(im):
    im = Image.fromarray(im)
    t_im = data_transform(im)
    return t_im

def extract_frame_feature_batch(list_tensor):
    with torch.no_grad():
        batch_tensor = torch.stack(list_tensor)
        batch_tensor = torch.transpose(batch_tensor, 0, 1) # 

        batch_tensor = batch_tensor.unsqueeze(0) 
        features = extractor(batch_tensor)

        features = features.view(features.size(0), -1).cpu()     
        return features


from pytorch_i3d import InceptionI3d
model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load('../models/rgb_imagenet.pt'))
extractor = torch.nn.DataParallel(model.cuda())
extractor.eval()

path_input = '../dataset/jester/'
path_output = '../dataset/I3D-feature-pretrain/'
feature_in_type = '.t7'
if not os.path.isdir(path_output):
    os.makedirs(path_output)

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def extract_features(video_file):
    rgb_files = [i for i in os.listdir(path_input + video_file)]
    if not os.path.isdir(path_output + video_file + '/'):
        os.makedirs(path_output + video_file + '/')
        
    num_exist_files = len(os.listdir(path_output + video_file + '/'))
    frames_tensor = []

    list_frames = os.listdir(path_input + video_file)
    list_frames.sort()

    try:
        for t in range(len(list_frames)):
            im = imageio.imread(path_input + video_file + '/' + list_frames[t])
            if np.sum(im.shape) != 0:
                id_frame = t+1
                frames_tensor.append(im2tensor(im))
    except RuntimeError:
        print('Could not read frame', id_frame+1, 'from', video_file)

    num_frames = len(frames_tensor)
    if num_frames == num_exist_files:
        return
    
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
        filename = path_output + video_file + '/' + 'img_' + id_frame_name + feature_in_type            

        if not os.path.exists(filename):
            torch.save(features[t].clone(), filename)
    print('Save ' + video_file + ' done!')


start = time.time()
list_videos = os.listdir(path_input)
list_videos.sort()
pool.map(extract_features, list_videos, chunksize=1)
end = time.time()
print('Total elapsed time: ' + str(end-start))