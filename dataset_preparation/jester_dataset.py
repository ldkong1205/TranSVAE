import os
import math
import pandas as pd
import numpy as np
import torch


class VideoDataset_Jester(Dataset):
	'''
    Input : 
	    csv_file     : Path to file where path to videos is stored - <path>, label
        frequency    : Sampling frequency for the i3d.
        num_nodes    : Number of graph nodes. Set to 16
        is_test      : Test/Train
	    transform    : if specified applies the transform to the frames.
        base_dir     : Base directory to the dataset. Please follow the instructions mentioned in the ReadMe
    
    Returns : 
        Tensor : [num_nodes, C, 8, H, W] ### num_nodes clips each consisting of 8 consecutive frames: 
        label  : Label of the video. (Not used for target dataset during training)    
	'''
	def __init__(self, csv_file, frequency = 4, num_nodes = 16, is_test = False, transform = None, base_dir='./data/jester'):
		self.transform = transform
		self.dataset = pd.read_csv(csv_file, header=None)
		self.min_frames = 72
		self.frequency = frequency
		self.chunk_size = 8
		self.num_nodes = num_nodes
		self.is_test = is_test
		if not base_dir.endswith("/"):
			base_dir += "/"
		self.bg_dir = base_dir + "jester_BG/"
		self.video_dir = base_dir + "jester_videos/"
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx) :
		path = self.video_dir + str(self.dataset.iloc[idx, 0])
		label = self.dataset.iloc[idx, 1]
		bg_path = self.bg_dir + '/' + path.split('/')[-1]
		rgb_files = [i for i in os.listdir(path)]
		bg_rgb_files = [i for i in os.listdir(bg_path)]
		rgb_files.sort()
		bg_rgb_files.sort()

		frame_indices = np.arange(len(rgb_files))
		bg_frame_indices = np.arange(len(bg_rgb_files))
		num_frames = len(rgb_files)
		if num_frames == 0:
			print("No images found inside the directory : ", path)
			raise Exception
		frames_tensor = load_rgb_batch(path, rgb_files, frame_indices, resize=True)
		bg_frames_tensor = load_rgb_batch(bg_path, bg_rgb_files, bg_frame_indices, resize=True)

		if self.transform and not self.is_test : 
			frames_tensor = self.transform(frames_tensor)

		frames_tensor = video_to_tensor(frames_tensor)
		bg_frames_tensor = video_to_tensor(bg_frames_tensor)

		if num_frames < self.min_frames :
			frames_tensor = torch.repeat_interleave(frames_tensor, math.ceil(self.min_frames/frames_tensor.shape[1]), dim=1)

		max_num_feats = frames_tensor.shape[1] // self.frequency - math.ceil(self.chunk_size/self.frequency)
		allRange = np.arange(max_num_feats)
		splitRange = np.array_split(allRange, self.num_nodes)
		try: 
			if not self.is_test : 
				fidx = [np.random.choice(a) for a in splitRange]
			else : 
				fidx = [a[0] for a in splitRange]
		except:
			print("Path : ", path)
			print("Split range : ", splitRange)
			print("All range : ", allRange)
			raise Exception
			
		ind = [np.arange(start=i*self.frequency, stop=i*self.frequency + self.chunk_size, step=1) for i in fidx]	
		frames_tensor_chunks = torch.empty(self.num_nodes, frames_tensor.shape[0], self.chunk_size, frames_tensor.shape[2], frames_tensor.shape[3])
		for chunk_ind, i in zip(ind, range(self.num_nodes)) : 
			frames_tensor_chunks[i, :, :, :, :] = frames_tensor[:, chunk_ind, :, :]

		return [frames_tensor_chunks, bg_frames_tensor], label