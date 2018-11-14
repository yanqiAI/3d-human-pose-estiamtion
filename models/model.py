#coding:utf-8
import torch
import os
import itertools
import torchvision.transforms as transfroms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import sys

import dsntnn
from tensorboardX import SummaryWriter
#from . import networks_3d
from .respose.resnet_pose import *
from . import integral
from . import losses
from .base_model import BaseModel

class Pose3dModel(BaseModel):
	def name(self):
		return 'Pose3dModel'
	
	def __init__(self, args):
		self.nJoints = args.nJoints

		BaseModel.initialize(self, args)
		
		self.loss_names = ['G_L1', 'euc', 'G']  # loss names
		self.model_names = ['G']
	
		# TODO resnet_pose_3d
		res_parameters = get_default_network_config()
		self.netG = get_pose_net(res_parameters, self.nJoints).to(self.device)
		init_pose_net(self.netG, res_parameters)
		
		if  self.isTrain:
			self.train_logger = SummaryWriter(os.path.join(args.checkpoints_dir, 'train_3d'))
			
			# define loss function  L1_loss
			self.criterion_G_xyz = torch.nn.L1Loss().to(self.device)
	
			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
			# self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG.parameters()),lr=args.lr,
			#                                       alpha=args.alpha,
			#                                       eps=args.epsilon,
			#                                       weight_decay=args.weightDecay,
			#                                       momentum=args.momentum
			#                                        )
			self.optimizers = []
			self.optimizers.append(self.optimizer_G)
		
	def set_input(self, input):
		if self.isTrain:
			self.CG     = input['CG' ].float().to(self.device)
			self.Pos_gt = input['Pos'].float().to(self.device)
		else:
			self.CG = input['CG'].float().to(self.device)
			
	def forward(self):
		
		#self.CG_pos3d, self.heatmaps = self.netG(self.CG) #list x y z  2*(B, 22,1) (B,22)    hm (B, 22, 64, 64)
		self.heatmaps = self.netG(self.CG) #(B, 22*64, 64, 64)
	
		hm_width  = self.heatmaps.shape[-1]
		hm_height = self.heatmaps.shape[-2]
		hm_depth  = self.heatmaps.shape[-3] // self.nJoints
		
		self.pred_xyz = integral.softmax_integral_tensor(self.heatmaps, self.nJoints, hm_width, hm_height, hm_depth) #(B, 22*3)

		self.pred_xyz = self.pred_xyz.view(-1, self.nJoints, 3) #(B, 22, 3)
		
		'''pred data process 需要将值域归一化到与输入图像一样的尺寸下再做loss'''
		# project to original image size
		
		# scale = 200.0 / 64.0
		
		# self.pred_xyz[:, :,   :1] = (self.pred_xyz[:, :,  :1] + 0.5) * scale
		# self.pred_xyz[:, :,  1:2] = (self.pred_xyz[:, :, 1:2] + 0.5) * scale
		# self.pred_xyz[:, :,  -1:] =  self.pred_xyz[:, :, -1:] * scale
		
		return self.pred_xyz

	def backward_G(self):
		
		'''target data process 需要将值域归一化到与预测值一样的值域再做loss'''
		
		#scale = 2000.0 / 64.0  # 世界坐标系到像素坐标系（三维）缩放系数
	
		self.target_x = (self.Pos_gt[:, 0, :,  :1] + 1000.0) / 2000.0 - 0.5
		self.target_y = (self.Pos_gt[:, 0, :, 1:2] + 1000.0) / 2000.0 - 0.5
		self.target_z = (self.Pos_gt[:, 0, :, -1:] + 500.0)  / 1000.0 - 0.5
		
		self.target_xyz = torch.cat((self.target_x, self.target_y, self.target_z), 2)
		
		#self.target_xyz = self.Pos_gt[:, 0, :, :]
		'''xyz-L1 loss'''
		self.loss_G_L1 = self.criterion_G_xyz(self.pred_xyz, self.target_xyz)
		
		'''xyz-euc loss'''
		self.loss_euc_pix = dsntnn.euclidean_losses(self.pred_xyz, self.target_xyz)
		self.loss_euc = dsntnn.average_loss(self.loss_euc_pix)
		'''total loss'''
		self.loss_G = 10 * self.loss_G_L1 + self.loss_euc
		
		self.loss_G.backward()
		
	def optimize_parameters(self):
		# forward
		self.forward()
		
		# train netG
		self.set_requires_grad(self.netG, True)
		self.optimizer_G.zero_grad()
		
		self.backward_G()
		
		self.optimizer_G.step()
		
	def summaries(self, total_steps):
		# normolizer image 0-1
		self.CG = vutils.make_grid(self.CG, normalize=True, scale_each=True)
	
		# add images
		self.train_logger.add_image('train/CG', self.CG, global_step=total_steps)
		
		# add losses
		self.train_logger.add_scalar('train/loss_G_L1', self.loss_G_L1, global_step=total_steps)
		self.train_logger.add_scalar('train/loss_euc', self.loss_euc, global_step=total_steps)
		self.train_logger.add_scalar('train/loss_G', self.loss_G, global_step=total_steps)

		