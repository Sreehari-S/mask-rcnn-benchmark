from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import glob
import random
import time
import pdb
# import imgaug
# from imgaug import augmenters as iaa
from PIL import Image
from tqdm import tqdm
import numpy as np 
from six.moves import range
# import tensorflow as tf
from torchvision import transforms  # noqa
from torch.utils.data import DataLoader, Dataset
from collections import deque
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# from helpers.utils import *

def rotate_list(input_list, N):
	input_list = deque(input_list) 
	input_list.rotate(N) 
	input_list = list(input_list)
	return input_list 

def ReadWholeSlideImage(image_path):
	img = Image.open(image_path)
	return img

def getImagePatch(image, coords, size):
	(x, y) = coords
	if len(image.shape) == 3:
		return image[x:x+size, y:y+size, :]
	else:
		return image[x:x+size, y:y+size]

class SignetCellTrainingDataset(Dataset):
	def __init__(self,data_root_dir, image_size=(256, 256),n_classes=2,n_channels =3,
				  shuffle=True,batch_size=4, samples_per_epoch=500, transform=None,mode='pos_only'):
		'Initialization'
		self.n_classes = n_classes
		self.mode = mode
		self.tumor_coord_path = os.path.join(data_root_dir,'training_tumor.txt')
		if self.mode == 'pos_and_fp':
			self.normal_coord_path = os.path.join(data_root_dir,'training_fp.txt')
		elif self.mode =='pos_only':
			self.normal_coord_path = None
		else:
			raise Exception("Dataloader Error!!")
		self.image_size = image_size
		self.n_channels = n_channels
		self.transforms = transform
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.tumor_coords = []
		self.normal_coords = []

		t = open(self.tumor_coord_path)
		for line in t:
			line_list = line.strip('\n').split(',')
			pid_path = line_list[0]
			center = [int(line_list[1]), int(line_list[2])]
			patch_list = line_list[3:]
			boxlist = []
			for i in range(0,len(patch_list),4):
				boxlist.append([int(patch_list[i]) - center[1],int(patch_list[i+1]) - center[0],int(patch_list[i+2]) - center[1],int(patch_list[i+3]) - center[0]])
			if len(boxlist) == 0:
				print(pid_path)
				raise Exception("Jaba")
			self.tumor_coords.append((pid_path,center,boxlist))
		print("No of tumor samples : {}".format(len(self.tumor_coords)))
		t.close()

		if self.normal_coord_path is not None:
			n = open(self.normal_coord_path)
			for line in n:
				line_list = line.strip('\n').split(',')
				pid_path = line_list[0]
				center = [int(line_list[1]), int(line_list[2])]
				patch_list = line_list[3:]
				boxlist = []
				for i in range(0,len(patch_list),4):
					boxlist.append([int(patch_list[i]) - center[1],int(patch_list[i+1]) - center[0],int(patch_list[i+2]) - center[1],int(patch_list[i+3]) - center[0]])
				self.normal_coords.append((pid_path,center,boxlist))
			print("No of False positive samples : {}".format(len(self.normal_coords)))
			n.close()

		self._num_image = len(self.tumor_coords) + len(self.normal_coords)
		if samples_per_epoch is None:
			self.samples_per_epoch = self._num_image
		else:
			self.samples_per_epoch = samples_per_epoch

		self._shuffle_counter = 1        
		self._shuffle_reset_idx = int(np.floor(self._num_image / self.samples_per_epoch))          
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		########return int(np.floor(self.samples_per_epoch / self.batch_size))
		return 180000

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		X, y = self.__data_generation(index)
		return X, y,index

	def on_epoch_end(self):
		'Updates indexes after each epoch'                        
		if self.shuffle == True:
			if self._shuffle_counter % self._shuffle_reset_idx == 0:
				random.shuffle(self.tumor_coords)
				random.shuffle(self.normal_coords)
				self._shuffle_counter = 1
		self.tumor_coords = rotate_list(self.tumor_coords, self.samples_per_epoch//2)
		self.normal_coords = rotate_list(self.normal_coords, self.samples_per_epoch//2)
		self._shuffle_counter += 1
	
  
	def __data_generation(self, index):
		'Generates data containing batch_size samples' 
		# Initialization
		X = []
		y = []
		norm_batch_size = self.batch_size//2
		# tumor_batch_size = self.batch_size - self.batch_size//2
		tumor_batch_size = self.batch_size################################################################

		for i in range(self.batch_size):

			# try:
			if i >= self.batch_size//2 or self.mode=='pos_only':
				pid_path, (x_center,y_center), boxlist = self.tumor_coords[(index)*norm_batch_size+i]
				pid_path = pid_path[6:]
				label = 1
			else:        
				pid_path, (x_center, y_center), boxlist = self.normal_coords[(index)*tumor_batch_size+i] 
				pid_path = pid_path[6:]
				label = 0

			# Generate data
			x_top_left = int(x_center)
			y_top_left = int(y_center)

			image = np.array(ReadWholeSlideImage(pid_path).convert('RGB'))
			# print ("=============================")
			# print (image.shape, self.image_size, x_top_left, y_top_left)
			image = Image.fromarray(getImagePatch(image, (x_top_left, y_top_left),
							min(self.image_size[0], self.image_size[1])))
			labels = []
			for l in boxlist:
				labels.append(label)
			labels = torch.tensor(labels)
			boxlist = BoxList(boxlist, image.size, mode="xyxy")
			boxlist.add_field("labels",labels)

			boxlist = boxlist.clip_to_image(remove_empty=True)
			#TODO: Check the below two lines
			if self.transforms is not None:
				image, boxlist = self.transforms(image, boxlist)
			X.append(image)
			y.append(boxlist)
			# except:
			#     continue
		return torch.stack(X), y

if __name__ == '__main__':
	dir_path = '/media/balaji/Kori/histopath/coordinate_data/train_test_points_fold_1'
	train_tumor_coord_path = os.path.join(dir_path, 'train_tumor.txt')
	train_normal_coord_path = os.path.join(dir_path, 'train_normal.txt')


 
	# augmentation = iaa.SomeOf((0, 3), 
	#         [
	#             iaa.Fliplr(0.5),
	#             iaa.Flipud(0.5),
	#             iaa.Noop(),
	#             iaa.OneOf([iaa.Affine(rotate=90),
	#                        iaa.Affine(rotate=180),
	#                        iaa.Affine(rotate=270)]),
	#             iaa.GaussianBlur(sigma=(0.0, 0.5)),
	#         ])
	# Parameters
	train_transform_params = {'image_size': (256,256),
						  'batch_size': 4,
						  'n_classes': 2,
						  'n_channels': 3,
						  'shuffle': True,
						  'level': 0,
						  'samples_per_epoch': 60000,
						  'transform': augmentation
						 }

	valid_transform_params = {'image_size': (256, 256),
						  'batch_size': 4,
						  'n_classes': 2,
						  'n_channels': 3,
						  'shuffle': True,
						  'level': 0,
						  'samples_per_epoch': 30000,                        
						  'transform': None
						 }
	# Generators
	# training_generator = DataGeneratorCoordFly(train_tumor_coord_path, train_normal_coord_path, **train_transform_params)
	# print (training_generator.__len__())
	# # # Enable Test Code
	# for X, y in training_generator:
	#     imshow(normalize_minmax(X[0]), y[0][:,:,1], normalize_minmax(X[1]), y[1][:,:,1], \
	#         normalize_minmax(X[2]), y[2][:,:,1], normalize_minmax(X[3]), y[3][:,:,1])        

	valid_tumor_coord_path = os.path.join(dir_path, 'validation_tumor.txt')
	valid_normal_coord_path = os.path.join(dir_path, 'validation_normal.txt')
	validation_generator = DataGeneratorCoordFly(valid_tumor_coord_path, valid_normal_coord_path, **valid_transform_params)
	print (validation_generator.__len__())
	# Enable Test Code

	# for X, y in validation_generator:
	#     imshow(normalize_minmax(X[0]), y[0][:,:,1], normalize_minmax(X[1]), y[1][:,:,1], \
	#         normalize_minmax(X[2]), y[2][:,:,1], normalize_minmax(X[3]), y[3][:,:,1])  
	import time        
	start_time = time.time()    
	for i, X in enumerate(validation_generator):
		elapsed_time = time.time() - start_time
		start_time = time.time()    
		print (i, "Elapsed Time", np.round(elapsed_time, decimals=2), "seconds")
		pass

