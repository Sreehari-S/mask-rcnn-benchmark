import os
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
import logging
import time
import glob
from shutil import copyfile
import fnmatch
import pdb

# Importing defaultdict 
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter 

ROOT_PATH = '../'
SPLITS = 3

# DigestPath 2019
data_path = ROOT_PATH  + 'DigestPath2019/Colonoscopy_tissue_segment_dataset'
pos_path   = data_path + '/tissue-train-pos'
neg_path   = data_path + '/tissue-train-neg'
# Save Path
save_path = ROOT_PATH + 'DigestPath2019/coordinate_data'
if not os.path.exists(save_path):
	os.makedirs(save_path)


train_image_list = []
train_mask_list  = []
train_label      = []


for (root,dirs,files) in os.walk(neg_path):
	for filename in files:
		train_image_list.append(os.path.join(root,filename))
		train_label.append(0)
		train_mask_list.append(0)

neg_len = len(train_image_list)
print('Number of negative images found : {}'.format(neg_len))

for (root,dirs,files) in os.walk(pos_path):
	for filename in fnmatch.filter(files,'*_mask.jpg'):
		mask_name = os.path.join(root,filename)
		img_name  = os.path.splitext(mask_name)[0][:-5] + '.jpg'
		train_image_list.append(img_name)
		train_mask_list.append(mask_name)
		train_label.append(1)

print('Number of positive images found : {}'.format(len(train_image_list)- neg_len))

#Train-val CV split
	
skf = StratifiedKFold(n_splits=SPLITS)
#skf.get_n_splits(train_image_list, train_label)
counter = 1
for train_index, test_index in skf.split(train_image_list, train_label):
	# print("TRAIN:", train_index, "TEST:", test_index)    
	image_train, image_test = itemgetter(*train_index)(train_image_list), itemgetter(*test_index)(train_image_list)  
	mask_train , mask_test  = itemgetter(*train_index)(train_mask_list), itemgetter(*test_index)(train_mask_list) 
	label_train,label_test  = itemgetter(*train_index)(train_label), itemgetter(*test_index)(train_label) 
	train_data = {'Image_Path': image_train, 'Label': label_train, 'Mask_Path': mask_train}
	df_train = pd.DataFrame(train_data)
	save_dir = os.path.join(save_path, 'train_test_points_fold_{}'.format(counter))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	df_train.to_csv(os.path.join(save_dir, 'training.csv'), index = False)
	validation_data = {'Image_Path': image_test, 'Label': label_test, 'Mask_Path': mask_test}
	df_validation = pd.DataFrame(validation_data)
	df_validation.to_csv(os.path.join(save_dir, 'validation.csv'), index = False)
	# print (df_validation)
	counter+= 1

