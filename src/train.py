import torch
import torch.nn as nn
import torch.utils.data as data

from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.model import *

import datasets
import time
import argparse
import test

def train():
	parser = argparse.ArgumentParser(description='my 3D human pose estimation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--start_epoch', type=int, default=1)
	parser.add_argument('--total_epochs', type=int, default=200)
	parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
	parser.add_argument('--isTrain', action='store_false', help='is train or not (test)')
	
	parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize|crop|resize_and_crop]')
	parser.add_argument('--crop_shape', type=int, default=[256, 256], help='the crop size')
	parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size --h')
	
	parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
	parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
	parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
	parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
	parser.add_argument('--alpha', type=float, default=0.999, help='')
	parser.add_argument('--epsilon', type=float, default=1e-8, help='')
	parser.add_argument('--weightDecay', type=float, default=0.0, help='')
	parser.add_argument('--momentum', type=float, default=0.0, help='')
	parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
	parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
	parser.add_argument('--gradient_clip', type=float, default=None)
	
	parser.add_argument('--nJoints', type=int, default=22, help='# number of joins')
	
	parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
	parser.add_argument('--number_gpus', '-ng', type=int, default=1, help='number of GPUs to use')
	parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	
	parser.add_argument('--name', type=str, default='3D_pose', help='name of the experiment. It decides where to store samples and models')
	parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
	parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
	parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
	parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
	
	parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
	parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

	args = parser.parse_args()
	
	dataset = datasets.CG_data(args)
	dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=8,
			pin_memory=True
	)
	
	dataset_size = len(dataloader) * args.batch_size
	print('*********training images = %d**************' % dataset_size)

# ========================================================================================
	
	model = Pose3dModel(args)
	model.setup(args)
	total_steps = 0
	
	for epoch in range(args.total_epochs):
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0
		
		for i, data in enumerate(dataloader):
			iter_start_time = time.time()
			if  total_steps % 50 == 0:
				t_data = iter_start_time - iter_data_time
				
			total_steps += args.batch_size
			epoch_iter += args.batch_size
			model.set_input(data)
			model.optimize_parameters()
			
			if total_steps % 10 == 0:
				model.summaries(total_steps)
			
			if total_steps % 10 == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / args.batch_size
				model.print_current_losses(epoch, epoch_iter, losses, t, t_data)
			
		if total_steps % args.save_latest_freq == 0:
			print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
			model.save_networks('latest')
			
		if epoch % args.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
			      (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)
			
		print('End of epoch %d / %d \t Time Taken: %d sec' %
		      (epoch, args.niter + args.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()
			
if __name__ == '__main__':
	train()
