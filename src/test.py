#coding:utf-8
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.model import *
import matplotlib.pyplot as plt
import numpy as np
import datasets
import argparse
import os
import errno
import scipy.misc

from PIL import Image
from glob import glob
from os.path import *
from mpl_toolkits.mplot3d import axes3d

import sys
sys.path.append('dataset/motion')
import BVH as BVH
import Animation as Animation
from AnimationPlot_000 import animation_plot

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--isTrain', action='store_true', help='is train or not (test)')
parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
parser.add_argument('--nJoints', type=int, default=22, help='# number of joins')
parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--checkpoints_dir', type=str, default='/home/yanqi/pose/my_pose/checkpoints', help='models are saved here')
parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize|crop|resize_and_crop]')
parser.add_argument('--crop_shape', type=int, default=[256, 256], help='the crop size')
parser.add_argument('--name', type=str, default='3D_pose', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

parser.set_defaults(model='test')
args = parser.parse_args()

def read_bvh(filename):
	anim, names, frametime = BVH.load(filename)
	glob_positions = Animation.positions_global(anim)  # ndarray (15000, 22, 3)
	
	return glob_positions * 10

bone22_rest_path = '../dataset/standard22.bvh'

def save_bvh_from_position_point(key, bone_size, save_path):
	if bone_size == 59:
		rest_path = bone59_rest_path
	elif bone_size == 80:
		rest_path = bone80_rest_path
	elif bone_size == 22:
		rest_path = bone22_rest_path
	elif bone_size == 31:
		rest_path = bone31_rest_path
	else:
		print('bone size error!!!!!!')
		return False
	
	anim, names, frametimes = BVH.load(rest_path)
	
	key = np.array(key, dtype=np.float64)
	key_size = key.shape[1]
	
	anim.rotations = anim.rotations[:1, :1]
	anim.rotations = np.repeat(anim.rotations, repeats=len(key), axis=0)
	anim.rotations = np.repeat(anim.rotations, repeats=key_size, axis=1)
	
	anim.names = names
	anim.positions = anim.positions[:1, :1]
	anim.positions = np.repeat(anim.positions, repeats=len(key), axis=0)
	anim.positions = np.repeat(anim.positions, repeats=key_size, axis=1)
	anim.positions = key
	
	anim.parents = np.zeros(key_size, dtype=int)
	anim.parents[0] = -1
	
	anim.offsets = key[0,]
	
	BVH.save(save_path, anim, names, frametimes, positions=True)
	
	
def evaluate(pred_3d, label_3d, output_joints_num):

	error_mm = np.sqrt(np.sum((pred_3d[:, :] - label_3d[0, :, :]) ** 2, axis = 1)).mean()
	#error_avg_mm = 10 * np.sqrt(np.sum((pred_3d[:, :] - label_3d[0, :, :]) ** 2, axis = 1)).mean(axis = 0)
	
	return error_mm

	
def test(args):
	dataset = datasets.Test_img(args)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=8,
		pin_memory=True)

	dataset_size = len(dataloader)
	print('******testing images = %d************' % dataset_size)

	# ========================================================================================
	# build model
	model_3d = Pose3dModel(args)
	model_3d.setup(args)
	
	pred_pos_result = []
	label_pose = []
	error = []
	
	gt_pos = read_bvh('../dataset/bvh/anim_0_15000.bvh')
	
	num, nJoint, c = gt_pos.shape
	for i in range(num):
		label_pose.append(gt_pos[i : i+1, :, :])
		
	for i, data in enumerate(dataloader):
		if i >= args.how_many:
			break
		model_3d.set_input(data)
		model_3d.test()

		if i % 5 == 0:
			print('processing %04d image has tested done' % i)

		#coords = model_3d.xy_pred_1
		# pred_xy = coords.data.cpu().numpy()
		# pred_z  = model_3d.z_pred.data.cpu().numpy() #(1, 22, 1)
		
		coords_3d = model_3d.pred_xyz
		pred_xyz = coords_3d.data.cpu().numpy()
		
		pred_xyz[:, :,  :1] = (pred_xyz[:, :,  :1] + 0.5) * 2000.0 - 1000.0
		pred_xyz[:, :, 1:2] = (pred_xyz[:, :, 1:2] + 0.5) * 2000.0 - 1000.0
		pred_xyz[:, :, -1:] = (pred_xyz[:, :, -1:] + 0.5) * 1000.0 -  500.0
		
		# single fram evaluate
		error_mm = evaluate(pred_xyz, label_pose[i], 22)
		error.append(np.float32(error_mm))
		print('Single fram error = %0.4f mm' % error_mm)
		
		pred_pos_result.append(pred_xyz / 10.0)
		
	error_avg_mm = sum([i.item() for i in error]) / len(error)
	print('----------Averge frams error = %0.4f mm----------' % error_avg_mm)
	
	# save npz
	output_directory = '/home/yanqi/pose/my_pose/results'
	output_npz_filename = '100_pred'
	try:
		# Create output directory if it does not exist
		os.makedirs(output_directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
	output_file_path = output_directory + '/' + output_npz_filename
	np.savez_compressed(output_file_path, pos_xyz = pred_pos_result)
	
	print('-----------------------start to show------------------------------')
	
	# 可视化
	pred_np_result = np.array(pred_pos_result)
	animation_plot([pred_np_result, gt_pos[:, :, :]], interval=15.15)
	
	print('-----------------------start to save result------------------------')
	# save bvh
	save_path_pred = output_directory + '/' + 'pred.bvh'
	save_path_label = output_directory + '/' + 'label.bvh'
	save_bvh_from_position_point(pred_np_result, 22, save_path_pred)
	save_bvh_from_position_point(gt_pos, 22, save_path_label)
	print('Save Done!')

if __name__ == '__main__':
	test(args)

