# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Multi-view Pose transformer
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import pprint

import _init_paths
import dataset
import models

from core.config import config
from core.config import update_config, update_config_dynamic_input
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_checkpoint_best
from utils.utils import load_backbone_panoptic

from getpass import getuser
from socket import gethostname

import lib.utils.misc as utils
import numpy as np
import random
from torch.utils.data import DistributedSampler
from models.util.misc import is_main_process, collect_results
from mmcv.runner import get_dist_info
import torch.distributed as dist
from prettytable import PrettyTable

import models.dq_transformer
from torchvision.utils import save_image
import torchvision

from utils.vis import save_debug_3d_images

from lib.structural.structural_triangulation import create_human_tree
from lib.structural.adapter import structural_triangulate_points

from lib.models.dq_decoder import get_proj_matricies_batch
from mvn.utils import multiview

import time

def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # if set then update the config.
    args, unknown = parser.parse_known_args()
    update_config(args.cfg)

    update_config_dynamic_input(unknown)

    return args


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_optimizer(model_without_ddp, weight_decay, optim_type):
    lr = config.TRAIN.LR
    if model_without_ddp.backbone is not None:
        for params in model_without_ddp.backbone.parameters():
            # If you want to train the whole model jointly, set it to be True.
            params.requires_grad = False

    lr_linear_proj_mult = config.DECODER.lr_linear_proj_mult
    lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, lr_linear_proj_names)
                 and p.requires_grad],
            "lr": lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad],
            "lr": lr * lr_linear_proj_mult,
        }
    ]

    if optim_type == 'adam':
        optimizer = optim.Adam(param_dicts, lr=lr)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=1e-4)

    return optimizer

def save_image_from_loader(loader, save_dir, prefix):
    for ind, (inputs, meta) in enumerate(loader):
        # save images
        for i,im in enumerate(inputs):
            size_arr = np.array(im.shape[-2:])
            size_arr = np.round(size_arr / 8.0).astype(int)
            size_list = size_arr.tolist()
            resize = torchvision.transforms.Resize(size_list)
            im_rs = resize(im)
            save_name = f'{prefix}_{ind}_v{i}.png'
            save_image(im_rs, save_dir + save_name)
            print('save to:', save_name)

def visualize_joints_ids(loader, save_dir, prefix):
    for ind, (inputs, meta) in enumerate(loader):
        # save images
        save_debug_3d_images(config, meta[0], None, os.path.join(save_dir, prefix, "idimage.png"), show_id=True)
            
        break # only for one

def poses_to_bone_length(poses_3d, n_person, conv_J2B):
    # (valid people, 15, 3)
    valid_poses_3d = poses_3d.squeeze()[:n_person,...]
    
    # (valid people, 45)
    n_valid = valid_poses_3d.shape[0]
    valid_poses_3d_flat = valid_poses_3d.reshape(n_valid,-1)
    
    # matrix to bone
    bones = valid_poses_3d_flat @ conv_J2B.T
    
    bones = bones.reshape(n_valid, 15, 3)
    
    bone_length = bones.norm(dim=2)[:,1:]
    
    return bone_length

def average_bone_length(loader, save_dir, prefix='bone_length', consider_n=300):
    # create human tree
    human_tree = create_human_tree(data_type='cmupanoptic')
    G = human_tree.conv_J2B
    
    # save all bones
    print('consider num:', consider_n)
    bones_list = []
    for ind, (inputs, meta) in enumerate(loader):
        # save images
        poses_3d = meta[0]['joints_3d']
        n_person = meta[0]['num_person']
        
        bone_length = poses_to_bone_length(poses_3d, n_person, G)
        
        bones_list.append(bone_length)
        
        if ind > consider_n:
            break
    all_bones = torch.cat(bones_list, dim=0)
    average_bone_length = all_bones.mean(0)
    return average_bone_length

def unit_test_st(loader, save_dir, prefix='unit_test_st', sigma = 50):
    
    consider_im_num = 10
    error_st_list = []
    error_t_list = []
    error_st_gtbone_list = []
    for ind, (inputs, meta) in enumerate(loader):
        # save images
        poses_3d_gt = meta[0]['joints_3d']
        n_person = meta[0]['num_person']
        if n_person > 0:
            n_person = 1
        else:
            print('no person.')
            continue
        
        # (valid people, 15, 3)
        valid_poses_3d_gt = poses_3d_gt.squeeze()[:n_person,...]
        
        n_views = 5

        # TODO: get projection mat
        device = poses_3d_gt.device
        proj_matricies_batch = get_proj_matricies_batch(meta, n_views, device, inv_trans=True)
        
        # extract 2d projections
        method_2d = "proj"
        if method_2d == 'gt':
            poses_2d_valid_views = []
            for view in range(n_views):
                poses_2d = meta[view]['joints']
                # get valid ones
                poses_2d_valid = poses_2d.squeeze()[:n_person,...]
                poses_2d_valid_views.append(poses_2d_valid)
            poses_2d_valid_views = torch.stack(poses_2d_valid_views)  # n_view, n_people, 15, 2
            poses_2d_valid_views = poses_2d_valid_views.transpose(0,1) #n_people, n_view, 15, 2
        elif method_2d == "proj":
            # proj points
            n_frames = 1
            n_joints = 15
            poses2D = np.zeros((n_frames, n_views, n_joints, 2))
            for i in range(n_frames):
                X3d = valid_poses_3d_gt[i, ...]
                for c in range(n_views):
                    X2d = proj_matricies_batch[0][c] @ np.concatenate((X3d, np.ones((n_joints, 1))), axis=1).T
                    X2d = (X2d[0:2, :] / X2d[2, :]).T
                    poses2D[i, c, :, :] = X2d
            poses_2d_valid_views = torch.from_numpy(poses2D)
        
        # adding noise to 2d views
        poses_2d_valid_views += torch.normal(0, torch.ones(poses_2d_valid_views.shape) * sigma)

        # re calculated 3d poses
        keypoints_2d_single_batch = poses_2d_valid_views[0,...].unsqueeze(0)
        n_steps = 3
        poses_3d_est = structural_triangulate_points(proj_matricies_batch, keypoints_2d_single_batch, confidences_batch=None, 
                                  n_steps=n_steps, method='ST')
        
        # ST with gt bone lengths
        human_tree = create_human_tree(data_type='cmupanoptic')
        G = human_tree.conv_J2B
        bone_length = poses_to_bone_length(poses_3d_gt, n_person, G) # A function changing poses to bone length
        bone_length = bone_length.unsqueeze(-1)
        
        t1 = time.time()
        poses_3d_est_gtbone = structural_triangulate_points(proj_matricies_batch, keypoints_2d_single_batch, confidences_batch=None, 
                                  n_steps=n_steps, method='ST', bone_length=bone_length)
        t2 = time.time()
        print('Time:', t2-t1)
        
        # Batch Version
        t1 = time.time()
        poses_3d_est_gtbone = structural_triangulate_points(proj_matricies_batch, keypoints_2d_single_batch, confidences_batch=None, 
                                  n_steps=n_steps, method='ST', bone_length=bone_length, batch=True)
        t2 = time.time()
        print('Time:', t2-t1)
        
        poses_3d_est_t = multiview.triangulate_batch_of_points_batch_version(
                    proj_matricies_batch, keypoints_2d_single_batch,
                    confidences_batch=None,
                    solver='linalg'
                )
        
        # evo pose error
        error_st = (poses_3d_est - valid_poses_3d_gt).square().sum(dim=-1).sqrt().mean()
        error_t = (poses_3d_est_t - valid_poses_3d_gt).square().sum(dim=-1).sqrt().mean()
        error_st_gtbone = (poses_3d_est_gtbone - valid_poses_3d_gt).square().sum(dim=-1).sqrt().mean()
        error_st_list.append(error_st)
        error_t_list.append(error_t)
        error_st_gtbone_list.append(error_st_gtbone)
        print(f'ST/ST-gt/T: {error_st}/{error_st_gtbone}/{error_t}')
        
        # Visualize 3d pose and compare with gt
        prefix_in = os.path.join(prefix, f'sigma-{sigma}-s{n_steps}')
        save_debug_3d_images(config, meta[0], poses_3d_est.unsqueeze(0), os.path.join(save_dir, prefix_in, f"ind-{ind}-st.png"), show_id=True)
        save_debug_3d_images(config, meta[0], poses_3d_est_t.unsqueeze(0), os.path.join(save_dir, prefix_in, f"ind-{ind}-t.png"), show_id=True)
        save_debug_3d_images(config, meta[0], poses_3d_est_gtbone.unsqueeze(0), os.path.join(save_dir, prefix_in, f"ind-{ind}-st-gt.png"), show_id=True)
        
        if ind > consider_im_num:
            break
    
    print('aver ST:', np.mean(np.array(error_st_list)))
    print('aver ST-gt:', np.mean(np.array(error_st_gtbone_list)))
    print('aver T:', np.mean(np.array(error_t_list)))
    print('sigma:', sigma)
    print('n_steps:', n_steps)
    
    result = {
        'aver_ST': np.mean(np.array(error_st_list)),
        'aver_ST_gt': np.mean(np.array(error_st_gtbone_list)),
        'aver_T': np.mean(np.array(error_t_list))
    }
    return result


def unit_test_st_least_square(loader, save_dir, prefix='unit_test_st_least_square'):
    consider_im_num = 0
    error_st_list = []
    error_t_list = []
    for ind, (inputs, meta) in enumerate(loader):
        # save images
        poses_3d_gt = meta[0]['joints_3d']
        n_person = meta[0]['num_person']
        if n_person > 0:
            n_person = 1
        else:
            print('no person.')
            continue
        
        # (valid people, 15, 3)
        valid_poses_3d_gt = poses_3d_gt.squeeze()[:n_person,...]
        
        # Calculate bone length 
        # (valid people, 45)
        n_valid = valid_poses_3d_gt.shape[0]
        valid_poses_3d_flat = valid_poses_3d_gt.reshape(n_valid,-1)
        
        # matrix to bone
        human_tree = create_human_tree(data_type='cmupanoptic')
        G = human_tree.conv_J2B
        bones = valid_poses_3d_flat @ G.T
        
        bones = bones.reshape(n_valid, 15, 3)
        
        bone_length = bones.norm(dim=2)[:,1:]
        print("bone_length:", bone_length)
        #####
        
        n_views = 5

        # TODO: get projection mat
        device = poses_3d_gt.device
        proj_matricies_batch = get_proj_matricies_batch(meta, n_views, device, inv_trans=True)
        
        # extract 2d projections
        method_2d = "proj"
        if method_2d == 'gt':
            poses_2d_valid_views = []
            for view in range(n_views):
                poses_2d = meta[view]['joints']
                # get valid ones
                poses_2d_valid = poses_2d.squeeze()[:n_person,...]
                poses_2d_valid_views.append(poses_2d_valid)
            poses_2d_valid_views = torch.stack(poses_2d_valid_views)  # n_view, n_people, 15, 2
            poses_2d_valid_views = poses_2d_valid_views.transpose(0,1) #n_people, n_view, 15, 2
        elif method_2d == "proj":
            # proj points
            n_frames = 1
            n_joints = 15
            poses2D = np.zeros((n_frames, n_views, n_joints, 2))
            for i in range(n_frames):
                X3d = valid_poses_3d_gt[i, ...]
                for c in range(n_views):
                    X2d = proj_matricies_batch[0][c] @ np.concatenate((X3d, np.ones((n_joints, 1))), axis=1).T
                    X2d = (X2d[0:2, :] / X2d[2, :]).T
                    poses2D[i, c, :, :] = X2d
            poses_2d_valid_views = torch.from_numpy(poses2D)
        
        # adding noise to 2d views
        sigma = 20
        poses_2d_valid_views += torch.normal(0, torch.ones(poses_2d_valid_views.shape) * sigma)

        ####################################
        # 2D initialization with noise: poses_2d_valid_views
        # 3D Groundtruth: valid_poses_3d_gt
        
        # Now we iteratively update 2d poses to match 3d gt 
        triangulation_method = 't'
        iter_num = 10
        loss_list = []
        lr = 1
        
        poses_2d_est = poses_2d_valid_views.clone().requires_grad_(True)
        
        param_dicts = [poses_2d_est]
        optimizer = optim.Adam(param_dicts, lr=lr)

        for it in range(iter_num):
            optimizer.zero_grad()
            
            # re calculated 3d poses
            if triangulation_method == 'st-bone':
                keypoints_2d_single_batch = poses_2d_est[0,...].unsqueeze(0)
                n_steps = 3
                poses_3d_est = structural_triangulate_points(proj_matricies_batch, keypoints_2d_single_batch, confidences_batch=None, 
                                        n_steps=n_steps, method='ST', bone_length=bone_length)
            elif triangulation_method == 'st':
                keypoints_2d_single_batch = poses_2d_est[0,...].unsqueeze(0)
                n_steps = 3
                poses_3d_est = structural_triangulate_points(proj_matricies_batch, keypoints_2d_single_batch, confidences_batch=None, 
                                        n_steps=n_steps, method='ST', bone_length=None)
            elif triangulation_method == 't':
                poses_3d_est = multiview.triangulate_batch_of_points_batch_version(
                            proj_matricies_batch, poses_2d_est,
                            confidences_batch=None,
                            solver='linalg'
                        )
        
            # Loss MPJPE
            loss = (poses_3d_est - valid_poses_3d_gt).square().sum(dim=-1).sqrt().mean()
            
            # Gradients
            loss.backward()
            optimizer.step()
            
            # evo pose error
            loss_list.append(loss.item())
            
            e = it
            if e % 1 == 0:
                grad = optimizer.param_groups[0]['params'][0].grad
                print(f'[it {e}] loss:', loss.item(), 'gd-norm:', grad.norm().item())
            
            # Visualize 3d pose and compare with gt
            prefix_in = os.path.join(prefix, f'{triangulation_method}-sigma-{sigma}', f'ind-{ind}')
            save_debug_3d_images(config, meta[0], poses_3d_est.unsqueeze(0), os.path.join(save_dir, prefix_in, f"it-{it}.png"))
            
        if ind > consider_im_num:
            break
    
    print('aver ST:', np.mean(np.array(error_st_list)))
    print('aver T:', np.mean(np.array(error_t_list)))
    print('sigma:', sigma)
    print('n_steps:', n_steps)        

def main():
    args = parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    if is_main_process():
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ]))

    num_views = train_dataset.num_views

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ]))

    if args.distributed:
        rank, world_size = get_dist_info()
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(test_dataset,
                                         world_size, rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.TRAIN.BATCH_SIZE, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        # collate_fn=utils.collate_fn,
        num_workers=config.WORKERS)

    save_dir = './process/see_dataset/'
    os.makedirs(save_dir, exist_ok=True)
    
    FLAG_OPEN = False
    if FLAG_OPEN:
        save_image_from_loader(train_loader, save_dir, prefix='train')
        save_image_from_loader(test_loader, save_dir, prefix='test')
    
    # Visualize joints IDs
    # visualize_joints_ids(train_loader, save_dir, prefix='vis_id')
    
    # Load all datasets and calculate average bone length
    FLAG_BONE_LENGTH = False
    if FLAG_BONE_LENGTH:
        lengths = average_bone_length(train_loader, save_dir, prefix='bone_length', consider_n = 300)
        # save to files
        torch.save(lengths, './bone_length_cmupanoptic')
    
    # Unit test for ST
    FLAG_UNIT_TEST_ST = True
    if FLAG_UNIT_TEST_ST:
        sigma_list = [0,10,20,40,60,80,100]
        table_result = {}
        for sigma in sigma_list:
            result = unit_test_st(train_loader, save_dir, prefix='unit_test_st', sigma=sigma)
            table_result[sigma] = result
        # Print all
        print(table_result)
        print('Done')
        
    FLAG_UNIT_TEST_LEAST_SQUARE = False
    if FLAG_UNIT_TEST_LEAST_SQUARE:
        unit_test_st_least_square(train_loader, save_dir, prefix='unit_test_st_least_square')
    
    


if __name__ == '__main__':
    main()
