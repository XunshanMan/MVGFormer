# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

# Note: this is the adapter for the structural triangulation; an unfinished version.

from lib.structural.structural_triangulation import Pose3D_inference_torch, create_human_tree, Pose3D_inference_torch_batch

import numpy as np
import torch

import time

def structural_triangulate_points(proj_matricies_batch, keypoints_2d, confidences_batch=None, 
                                  n_steps=1, method='ST', bone_length=None, batch=False):
    '''
    @ bone_length: (batch_size, N_joints, 1) or (1, N_joints, 1)
    '''
    
    # Change input to st input
    # keypoints_3d = structural_triangulate_points(
    #         proj_matricies_batch, keypoints_2d_undistort,
    #         confidences_batch=alg_confidences
    #     )
    
    
    # Get human tree, we need transformation matrix
    human_tree = create_human_tree(data_type = "cmupanoptic")

    # TODO: How to load cam number
    n_cams = 5
          
    poses_2d_tensor = keypoints_2d
    confidences_tensor = confidences_batch

    device = poses_2d_tensor.device
    batch_size = poses_2d_tensor.shape[0]

    if bone_length is not None:
        lengths_tensor = bone_length
    else:
        gt_bone_dir='/home/X/data/panoptic/bone_length.pth'
        print('No gt bone length. Load from:', gt_bone_dir)
        lengths_tensor = torch.load(gt_bone_dir)
        lengths_tensor = lengths_tensor.reshape(-1,1)
        lengths_tensor = lengths_tensor.expand(batch_size,-1,-1)
    lengths_tensor = lengths_tensor.to(device)
    lengths_tensor = lengths_tensor.float()
    
    # Consider if it's one for all, or each contains one
    # lengths_tensor = lengths_tensor.float().reshape(-1,1).to(device)
    # n_length = len(lengths_tensor)
    # if n_length == 1:
    #     # expand to all
    #     lengths_tensor.expand(batch_size, -1, -1)
        
    
    # print('lengths_tensor:', lengths_tensor)
    
    Projections_tensor = proj_matricies_batch
   
    poses_list = []
    
    # t1= time.time()
    if batch:
        poses_output = Pose3D_inference_torch_batch(n_cams, human_tree, poses_2d_tensor, confidences_tensor,
                        lengths_tensor, Projections_tensor,
                        method, n_steps)
    else:
        for b in range(batch_size):
            if confidences_tensor is not None:
                conf_in = confidences_tensor[b,...]
            else:
                conf_in = None
            poses_b = Pose3D_inference_torch(n_cams, human_tree, poses_2d_tensor[b,...], conf_in,
                                    lengths_tensor[b,...], Projections_tensor[b,...],
                                    method, n_steps)
            poses_list.append(poses_b)
        poses_output = torch.stack(poses_list, dim = 0)
    # change output to origin output
    # t2= time.time()
    # print(f'Time ST for {batch_size} poses:', t2-t1)
    
    return poses_output
