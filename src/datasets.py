#coding:utf-8

import torch
import torch.utils.data as data
import torchvision.transforms as transfroms

import os, math, random, sys
from os.path import *
import numpy as np
from glob import glob
from PIL import Image

dataset_directory = '../'
sys.path.append(dataset_directory + '/motion')
import BVH as BVH
import Animation as Animation

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def read_bvh(filename):
	anim, names, frametime = BVH.load(filename)
	glob_positions = Animation.positions_global(anim) # ndarray (15000, 22, 3)
	
	return glob_positions * 10

def is_image_file(filename):
	
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_transform(args):
	transfrom_list = []
	
	if args.resize_or_crop == 'resize':
		osize = args.crop_shape
		transfrom_list.append(transfroms.Resize(osize, Image.BICUBIC))
	elif args.resize_or_crop == 'crop':
		transfrom_list.append(transfroms.RandomCrop(args.fineSize))
	elif args.resize_or_crop == 'resize_and_crop':
		transfrom_list.append(transfroms.Resize(osize, Image.BICUBIC))
		transfrom_list.append(transfroms.RandomCrop(args.fineSize))
	else:
		raise ValueError('--resize_or_crop %s is not a valid option.' % args.resize_or_crop)
	
	transfrom_list += [transfroms.ToTensor(), transfroms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
	
	return transfroms.Compose(transfrom_list)

class CG_data(data.Dataset):
	def __init__(self, args, is_cropped = True, root = '/home/yanqi/pose/my_pose/dataset'):
		self.args = args
		self.is_cropped = is_cropped
		
		# CG data root
		CG_data_root = join(root, 'CG')
		
		# bvh file root
		BVH_data_root = join(root, 'bvh')
		
		# data list
		self.CG_data_list = []
		self.BVH_data_list = []
		self.Pos_bvh_list = []
		
		for root, _, fnames in sorted(os.walk(CG_data_root)):
			for fname in sorted(fnames):
				if is_image_file(fname):
					path = os.path.join(root, fname)
					self.CG_data_list.append(path)
					
		for root, _, fnames in sorted(os.walk(BVH_data_root)):
			for fname in sorted(fnames):
				path = os.path.join(root, fname)
				self.BVH_data_list.append(path)

		# read bvh file
		for i in range(len(self.BVH_data_list)):
			Pos_gt = read_bvh(self.BVH_data_list[i])  ##(15000, 22, 3)
			num, _, _ = Pos_gt.shape
			for j in range(num):
				Pos_gt_frame = Pos_gt[j:j+1, :, :]
				self.Pos_bvh_list.append(Pos_gt_frame)
				
		self.CG_data_size = len(self.CG_data_list)
		self.BVH_data_size = len(self.Pos_bvh_list)
		
		self.transform = get_transform(args)
		
	def __getitem__(self, index):
		
		# read CG images   one image---> one fram
		index = index % self.CG_data_size
		CG_data = Image.open(self.CG_data_list[index]).convert('RGB')
		
		Pos_GT = self.Pos_bvh_list[index]  ##(1, 22, 3)
		
		# PIL to tensor
		CG = self.transform(CG_data)
		
		# numpy to tensor
		Pos_gt = torch.from_numpy(Pos_GT)
		
		return {'CG' : CG, 'Pos' : Pos_gt}
	
	def __len__(self):
		return max(self.CG_data_size, self.BVH_data_size)
	
class Test_img(data.Dataset):
	def __init__(self, args, is_cropped=True, root='/home/yanqi/pose/my_pose/dataset'):
		self.args = args
		self.is_cropped = is_cropped
		
		# images root
		img_root = join(root, 'test_img/test_1')
	
		# data list
		self.img_list = []

		img_data = sorted(glob(join(img_root, '*png')))

		for i in range(len(img_data)):
			self.img_list += [img_data[i]]
		
		self.img_size = len(self.img_list)
	
		self.transform = get_transform(args)
	
	def __getitem__(self, index):
		
		# read CG images   one image---> one fram
		index = index % self.img_size
		img_data = Image.open(self.img_list[index]).convert('RGB')
		
		# PIL to tensor
		img_test = self.transform(img_data)
		
		return {'CG': img_test}
	
	def __len__(self):
		return self.img_size
	