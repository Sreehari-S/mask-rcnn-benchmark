import sys
import os
import argparse
import logging
import time
import glob
from shutil import copyfile
from multiprocessing import Pool, Value, Lock
import openslide
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import pandas as pd
import cv2
import pdb
# import multiresolutionimageinterface as mir
import skimage
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import points_in_poly
from skimage import feature
from skimage.feature import canny
from sklearn.model_selection import KFold
import copy
import glob
import json
import random
import tqdm
from operator import itemgetter 
from collections import defaultdict
np.random.seed(0)
from PIL import Image
Image.MAX_IMAGE_PIXELS= 999999999999999

ROOT_PATH = '../'

# CV fold: 3 folds exists
fold_no = 2
base_path = ROOT_PATH + 'DigestPath2019/coordinate_data'

Train_train_split = base_path + '/train_test_points_fold_{0}/training.csv'.format(fold_no)
Train_valid_split = base_path + '/train_test_points_fold_{0}/validation.csv'.format(fold_no)

# Output path for text files of coordinates
out_path = base_path + '/train_test_points_fold_{0}'.format(fold_no)
if not os.path.exists(out_path):
	os.makedirs(out_path)

def RandomUniformSample(data, n=1000, factor=1):
	data=copy.deepcopy(data);
	if len(data) <= n:
		sample_n = len(data)*factor        
	else:
		sample_n = n
		
	idxs = [];
	while len(idxs)<sample_n:
		rand=int(random.uniform(0, len(data)))
		if rand in idxs:
			pass
		else:
			idxs.append(rand);
	sample=[data[i] for i in idxs];
	return sample


def TissueMask(img_RGB):
	RGB_min = 50
	img_HSV = rgb2hsv(img_RGB)

	background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
	background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
	background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
	tissue_RGB = np.logical_not(background_R & background_G & background_B)
	tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
	min_R = img_RGB[:, :, 0] > RGB_min
	min_G = img_RGB[:, :, 1] > RGB_min
	min_B = img_RGB[:, :, 2] > RGB_min
	tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
	return tissue_mask

def extract_normal_patches_from_images(image_path, mask_path, out_path, mode, max_normal_points=180):                    
	'''
	Extract Normal Patches coordinates and write to text file
	'''
	patch_level = 0
	patch_size = 256
	tumor_threshold = 0
	target_file = open(os.path.join(out_path, "{}_normal.txt".format(mode)), 'a')
	
	if os.path.exists(mask_path):
		mask  = skimage.io.imread(mask_path)
		image = skimage.io.imread(image_path)  
		tissue_mask = TissueMask(image)
		sampled_normal_pixels = list(np.transpose(np.nonzero(tissue_mask)))
		# Perform Uniform sampling
		sampled_normal_pixels = RandomUniformSample(sampled_normal_pixels, max_normal_points) 
		sampled_normal_pixels_verified = []             
		for coord in sampled_normal_pixels:   
			scoord = (int(coord[0]), int(coord[1]))
			# shifted_point = (max(int(scoord[0]-patch_size//2),0), max(int(scoord[1]-patch_size//2),0))
			mask_patch = mask[scoord[0]:scoord[0]+patch_size,scoord[1]:scoord[1]+patch_size]  
			tumor_fraction = np.count_nonzero(mask_patch)/np.prod(mask_patch.shape) 
			if tumor_fraction <= tumor_threshold:
				sampled_normal_pixels_verified.append(scoord)
	else:
		# return 100
		mask_path = '0'      
		image = skimage.io.imread(image_path)         
		tissue_mask = TissueMask(image)
		sampled_normal_pixels_verified = list(np.transpose(np.nonzero(tissue_mask)))
		
	# Perform Uniform sampling
	sampled_normal_pixels_verified = RandomUniformSample(sampled_normal_pixels_verified, max_normal_points) 
	print(len(sampled_normal_pixels_verified))   
	for tpoint in sampled_normal_pixels_verified:
		target_file.write(image_path +','+mask_path +','+ str(tpoint[0]) + ',' + str(tpoint[1]))        
		target_file.write("\n")
	target_file.close()    
	return(len(sampled_normal_pixels_verified))


def extract_tumor_patches_from_images(image_path, mask_path, out_path, mode, max_tumor_points=500):
	'''
	Extract Patches coordinates and write to text file
	'''
	patch_size = 256
	  
	target_file = open(os.path.join(out_path, "{}_tumor.txt".format(mode)), 'a')
	mask  = skimage.io.imread(mask_path)
	tumor_pixels = list(np.transpose(np.nonzero(mask)))
	tumor_pixels = RandomUniformSample(tumor_pixels, max_tumor_points)       
	
	# for coord in tumor_pixels:
		# scaled_shifted_point = (max(coord[0]-patch_size//2,0), max(coord[1]-patch_size//2,0))
		# wsi_obj, _, level = ReadWholeSlideImage(image_path, sampling_level, RGB=True, read_image=False)
		# slide_patch = np.array(wsi_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('RGB'))
		# mask_patch = np.array(mask_obj.read_region(scaled_shifted_point, patch_level, (patch_size, patch_size)).convert('L'))
		# imshow(slide_patch, mask_patch)  
	print(len(tumor_pixels))
	for tpoint in tumor_pixels:
#         target_file.write(os.path.basename(image_path) +','+ str(tpoint[0]) + ',' + str(tpoint[1]))
		target_file.write(image_path +','+mask_path +','+ str(tpoint[0]) + ',' + str(tpoint[1]))        
		target_file.write("\n")

	target_file.close()
	return(len(tumor_pixels))


train_count = 0
valid_count = 0

# DP19 Train Random Sample Coordinates
mode = 'training'
train_split_df = pd.read_csv(Train_train_split)
for index, row in train_split_df.iterrows():
	image_path = row['Image_Path']
	print(index+1,' ',image_path)
	mask_path  = row['Mask_Path']
	# if index >= 341:
	train_count+=extract_normal_patches_from_images(image_path, mask_path, out_path, mode)
	if os.path.exists(mask_path):
		train_count+=extract_tumor_patches_from_images(image_path, mask_path, out_path, mode)
print ('Points sampled:', train_count)
		
mode = 'validation'    
valid_split_df = pd.read_csv(Train_valid_split)
for index, row in valid_split_df.iterrows():
	image_path = row['Image_Path']
	print(index+1,' ',image_path)
	mask_path  = row['Mask_Path']
	valid_count+=extract_normal_patches_from_images(image_path, mask_path, out_path, mode)
	if os.path.exists(mask_path):    
		valid_count+=extract_tumor_patches_from_images(image_path, mask_path, out_path, mode)
print ('Points sampled:', valid_count)