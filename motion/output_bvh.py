# coding: utf-8
import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

import motion.BVH as BVH
import motion.Animation as Animation
from motion.Quaternions import Quaternions
from motion.Pivots import Pivots
"""
global position和local rotation
"""
bone59_names = [
    'Hipx', 'LowBack', 'BRHip', 'BLHip', 'FLHip', 'RLHip',
    'TopSpine', 'BackOffset', 'DOWN_Sternum', 'UP_Sternum',
    'Top', 'Head_RF', 'Head_LF', 'Head_RB', 'Head_LB',
    'LShoulder_Top', 'LShoulder_B', 'LShoulder_F',
    'L_Bicep', 'LU_Elbow', 'LD_Elbow',
    'LForeArm', 'LOutside_Wrist', 'Linside_Wrist',
    'Linside_Hand', 'LOutside_Hand',
    'RShoulder_B', 'RShoulder_Top', 'RShoulder_F',
    'R_Bicep', 'RD_Elbow', 'RU_Elbow',
    'RForeArm', 'ROutside_Wrist', 'Rinside_Wrist',
    'ROutside_Hand', 'Rinside_Hand',
    'RThigh', 'RBKnee', 'RFKnee',
    'RB_Ankle', 'RShank', 'RF_Ankle',
    'RHeel', 'ROutsideMidfoot',
    'R_OutsideToe', 'RToe', 'R_InsideToe',
    'LThigh', 'LBKnee', 'LFKnee',
    'LB_Ankle', 'LShank', 'LF_Ankle',
    'LHeel', 'LOutsideMidfoot',
    'L_OutsideToe', 'LToe', 'L_InsideToe'
]
bone59_rest_path = '/home/dms/project/animation/interpolation/vae_positions/datasets/bone59_rest/fbx_rest.bvh'
bone80_rest_path = '/home/dms/project/animation/interpolation/vae_positions/datasets/bone80_rest/fbx_rest.bvh'
bone22_rest_path = '/home/dms/datasets/2d_3dpose/animation/mocap_pro/bvh/results1/anim_0_15000.bvh'
save_path = 'results'

def save_bvh_from_position_point(key, bone_size, save_path):
    if bone_size == 59:
        rest_path = bone59_rest_path
    elif bone_size == 80:
        rest_path = bone80_rest_path
    elif bone_size == 22:
        rest_path = bone22_rest_path
    else:
        print 'bone size error!!!!!!'
        return False

    anim, names, frametimes = BVH.load(rest_path)
    # names = bone59_names

    key = np.array(key, dtype=np.float64)

    key = key - key[:,0:1,:]
    key[:,0:1,:] = 0.0
    # root_pos = np.zeros([len(key), 1, 3], dtype=np.float64)
    # #
    # key = np.concatenate([root_pos, key], axis=1)
    key_size = key.shape[1]

    anim.rotations = anim.rotations[:1, :1]
    anim.rotations = np.repeat(anim.rotations, repeats=len(key), axis=0)
    anim.rotations = np.repeat(anim.rotations, repeats=key_size, axis=1)

    anim.names = names
    anim.positions = anim.positions[:1, :1]
    anim.positions = np.repeat(anim.positions, repeats=len(key), axis=0)
    anim.positions = np.repeat(anim.positions, repeats=key_size, axis=1)
    # anim.positions[:,0,:] =0.0
    anim.positions = key

    anim.parents = np.zeros(key_size, dtype=int)
    anim.parents[0] = -1

    anim.offsets = key[0,]


    BVH.save(save_path, anim, names, frametimes, positions=True)

def save_bvh_from_glopos_glorot(path, rest_path, data, joint_num=22, is_ref=False):

    root_poss = data[:, :3]
    root_poss = root_poss[:, np.newaxis, :]
    global_positions = data[:, 3:3 + joint_num * 3]
    local_rotations = data[:, 3 + joint_num * 3:]

    global_positions = global_positions.reshape([len(global_positions), joint_num, -1])
    local_rotations = local_rotations.reshape([len(local_rotations), joint_num, -1])

    global_positions[:, :, 0] += root_poss[:, :, 0]
    global_positions[:, :, 2] += root_poss[:, :, 2]

    """ Extract Forward Direction """
    if joint_num == 31:
        sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    elif joint_num == 52:
        sdr_l, sdr_r, hip_l, hip_r = 7, 26, 48, 44
    across = (
        (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
        (global_positions[:, hip_l] - global_positions[:, hip_r]))
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    """ Smooth Forward Direction """

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    root_rotation = Quaternions.between(forward,
                                        np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]

    global_rotations = -root_rotation * Quaternions.exp(local_rotations)
    global_rotations = global_rotations.transforms()

    output_anim, names, frametime = BVH.load(rest_path)
    output_anim = output_anim[0:1]
    if output_anim.shape[0] >= global_positions.shape[0]:
        output_anim = output_anim[:global_positions.shape[0], ]
    else:
        output_anim = np.repeat(output_anim, repeats=global_positions.shape[0], axis=0)
    output_xforms = Animation.transforms_global(output_anim)
    output_xforms[:, :, :3, :3] = global_rotations
    output_xforms[:, :, :3, 3] = global_positions

    local_xforms = Animation.transforms_blank(output_anim)
    local_xforms[:, 0] = output_xforms[:, 0]
    for k in range(output_xforms.shape[0]):
        for j in range(1, output_xforms.shape[1]):
            local_xforms[k, j] = np.mat(output_xforms[k, output_anim.parents[j]]).I * np.mat(output_xforms[k, j])
    local_rotations = local_xforms[:, :, :3, :3]
    local_positions = local_xforms[:, :, :3, 3]
    trans_rotations = Quaternions.from_transforms(local_rotations)
    # b = -trans_rotations.qs[:, 0, :]
    # trans_rotations.qs[:, 0, :] = b
    output_anim.rotations = trans_rotations
    output_anim.positions = local_positions
    BVH.save(path, output_anim, names=names, positions=False, frametime=frametime)

def save_bvh_from_xform(path, rest_path, data, joint_num=22, is_ref=False):
    if is_ref:
        root_poss = data[:, :3]
        root_poss = root_poss[:, np.newaxis, :]
        global_positions = data[:, 3:3+joint_num*3]
        global_rotations = data[:, 3+joint_num*3:]
    else:
        root_poss = np.zeros(shape=[data.shape[0], 3])
        root_poss = root_poss[:, np.newaxis, :]
        global_positions = data[0, :joint_num*3]
        global_rotations = data[0, joint_num*3:]

    global_positions = global_positions.reshape([-1, joint_num, 3])
    global_rotations = global_rotations.reshape([-1, joint_num, 3, 3])

    global_positions[:,:,0] += root_poss[:,:,0]
    global_positions[:,:,2] += root_poss[:,:,2]

    output_anim, names, frametime = BVH.load(rest_path)
    output_anim = output_anim[0:1]
    if output_anim.shape[0] >= data.shape[0]:
        output_anim = output_anim[:data.shape[0],]
    else:
        output_anim = np.repeat(output_anim, repeats=data.shape[0], axis=0)
    output_xforms = Animation.transforms_global(output_anim)
    output_xforms[:, :, :3, :3] = global_rotations
    output_xforms[:, :, :3, 3] = global_positions

    local_xforms = Animation.transforms_blank(output_anim)
    local_xforms[:,0]=output_xforms[:,0]
    for k in range(output_xforms.shape[0]):
        for j in range(1, output_xforms.shape[1]):
            local_xforms[k, j] = np.mat(output_xforms[k, output_anim.parents[j]]).I * np.mat(output_xforms[k, j])
    local_rotations = local_xforms[:, :, :3, :3]
    local_positions = local_xforms[:, :, :3, 3]
    trans_rotations = Quaternions.from_transforms(local_rotations)
    b = -trans_rotations.qs[:, 0, :]
    trans_rotations.qs[:, 0, :] = b
    output_anim.rotations = trans_rotations
    output_anim.positions = local_positions

    BVH.save(path, output_anim, names=names, positions=False, frametime=frametime)

def save_bvh_from_root_orgrot(path, rest_path, data, joint_num=22, is_ref=False):

    data = data.reshape([len(data), -1, joint_num * 3])

    global_positions = data[:, 0, :]
    global_rotations = data[:, 1, :]

    # global_positions = global_positions.reshape([-1, joint_num, 3])
    global_rotations = global_rotations.reshape([len(global_rotations), joint_num, 3])
    # global_rotations = Quaternions.from_euler(global_rotations)
    # global_rotations = Quaternions.exp(global_rotations)
    order = 'xyz'
    world = False
    global_rotations = Quaternions.from_euler(global_rotations, order=order, world=world)
    # global_positions[:,:,0] += root_poss[:,:,0]
    # global_positions[:,:,2] += root_poss[:,:,2]
    output_anim, names, frametime = BVH.load(rest_path)

    output_anim = output_anim[0:1]
    if output_anim.shape[0] >= data.shape[0]:
        output_anim = output_anim[:data.shape[0],]
    else:
        output_anim = np.repeat(output_anim, repeats=data.shape[0], axis=0)
    output_xforms = Animation.transforms_global(output_anim)
    # output_xforms[:, :, :3, :3] = global_rotations
    output_xforms[:, 0, :3, 3] = global_positions[:, :3]
    #  = global_positions

    local_xforms = Animation.transforms_blank(output_anim)
    local_xforms[:,0]=output_xforms[:,0]
    for k in range(output_xforms.shape[0]):
        for j in range(1, output_xforms.shape[1]):
            local_xforms[k, j] = np.mat(output_xforms[k, output_anim.parents[j]]).I * np.mat(output_xforms[k, j])
    local_rotations = local_xforms[:, :, :3, :3]
    local_positions = local_xforms[:, :, :3, 3]
    trans_rotations = Quaternions.from_transforms(local_rotations)
    b = -trans_rotations.qs[:, 0, :]
    trans_rotations.qs[:, 0, :] = b
    output_anim.rotations = global_rotations
    output_anim.positions = local_positions

    BVH.save(path, output_anim, names=names, positions=False, frametime=frametime)

# def save_bvh_from_root_orgrot(path, rest_path, data, is_ref=False):
#
#     root_poss = data[:, :3]
#     global_rotations = data[:, 3:]
#
#     # global_positions = global_positions.reshape([-1, joint_num, 3])
#     global_rotations = global_rotations.reshape([-1, 22, 3])
#     # global_rotations = Quaternions.from_euler(global_rotations)
#     global_rotations = Quaternions.exp(global_rotations)
#     # global_positions[:,:,0] += root_poss[:,:,0]
#     # global_positions[:,:,2] += root_poss[:,:,2]
#     output_anim, names, frametime = BVH.load(rest_path)
#
#     output_anim = output_anim[0:1]
#     if output_anim.shape[0] >= data.shape[0]:
#         output_anim = output_anim[:data.shape[0],]
#     else:
#         output_anim = np.repeat(output_anim, repeats=data.shape[0], axis=0)
#     output_xforms = Animation.transforms_global(output_anim)
#     # output_xforms[:, :, :3, :3] = global_rotations
#     output_xforms[:, 0, :3, 3] = root_poss
#     #  = global_positions
#
#     local_xforms = Animation.transforms_blank(output_anim)
#     local_xforms[:,0]=output_xforms[:,0]
#     for k in range(output_xforms.shape[0]):
#         for j in range(1, output_xforms.shape[1]):
#             local_xforms[k, j] = np.mat(output_xforms[k, output_anim.parents[j]]).I * np.mat(output_xforms[k, j])
#     local_rotations = local_xforms[:, :, :3, :3]
#     local_positions = local_xforms[:, :, :3, 3]
#     trans_rotations = Quaternions.from_transforms(local_rotations)
#     b = -trans_rotations.qs[:, 0, :]
#     trans_rotations.qs[:, 0, :] = b
#     output_anim.rotations = global_rotations
#     output_anim.positions = local_positions
#
#     BVH.save(path, output_anim, names=names, positions=False, frametime=frametime)

def save_bvh_from_root_rot(path, rest_path, data, joint_num=22, is_ref=False):

    global_positions = data[:, :3 * joint_num]
    root_poss = data[:, :3]
    global_rotations = data[:, 3 * joint_num:]

    # global_positions = global_positions.reshape([-1, joint_num, 3])
    global_rotations = global_rotations.reshape([-1, joint_num, 3, 3])

    # global_positions[:,:,0] += root_poss[:,:,0]
    # global_positions[:,:,2] += root_poss[:,:,2]

    output_anim, names, frametime = BVH.load(rest_path)
    output_anim = output_anim[0:1]
    if output_anim.shape[0] >= data.shape[0]:
        output_anim = output_anim[:data.shape[0],]
    else:
        output_anim = np.repeat(output_anim, repeats=data.shape[0], axis=0)
    output_xforms = Animation.transforms_global(output_anim)
    output_xforms[:, :, :3, :3] = global_rotations
    output_xforms[:, 0, :3, 3] = root_poss
    #  = global_positions

    local_xforms = Animation.transforms_blank(output_anim)
    local_xforms[:,0]=output_xforms[:,0]
    for k in range(output_xforms.shape[0]):
        for j in range(1, output_xforms.shape[1]):
            local_xforms[k, j] = np.mat(output_xforms[k, output_anim.parents[j]]).I * np.mat(output_xforms[k, j])
    local_rotations = local_xforms[:, :, :3, :3]
    local_positions = local_xforms[:, :, :3, 3]
    trans_rotations = Quaternions.from_transforms(local_rotations)
    b = -trans_rotations.qs[:, 0, :]
    trans_rotations.qs[:, 0, :] = b
    output_anim.rotations = trans_rotations
    output_anim.positions = local_positions

    BVH.save(path, output_anim, names=names, positions=False, frametime=frametime)