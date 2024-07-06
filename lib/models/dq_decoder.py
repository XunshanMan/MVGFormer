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

# ------------------------------------------------------------------------
# Multi-view Pose transformer
# ----------------------------------------------------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# and Deformable Detr
# (https://github.com/fundamentalvision/Deformable-DETR)
# ----------------------------------------------------------------------------------------------------------------------------------------

import copy
from turtle import forward
# from typing import Optional, List
# import math

import torch
import torch.nn.functional as F
from torch import nn
from lib.core.function import time_synchronized
# from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from lib.models.util.misc import inverse_sigmoid
from lib.models.ops.modules import ProjAttn

import lib.utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
# from utils.transforms import affine_transform_pts_cuda as do_transform
from utils.transforms import \
    affine_transform_pts_cuda_batch as do_transform_batch

import time

from models.mvp_decoder import MvPDecoder, MvPDecoderLayer, _get_activation_fn
from models.multi_view_pose_transformer import MLP

## triangulation
from mvn.utils import op, multiview
import numpy as np

from lib.utils.vis import *

from lib.structural.adapter import structural_triangulate_points

from lib.structural.structural_triangulation import create_human_tree

count_layer = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

time_genearte_feat = AverageMeter()
time_features = AverageMeter()
time_prob = AverageMeter()
time_padding = AverageMeter()
time_triangulate = AverageMeter()
time_finalprocess = AverageMeter()
count_query_nums = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()] # 4


class offset_net(nn.Module):
    def __init__(self, in_dim, hid_dim, layer_num):
        super().__init__()
        
        self.MLP = MLP(in_dim, hid_dim, 3, layer_num)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, feature):
        out = self.MLP(feature) # batch, bins, 3
        
        offset = out[...,:2]
        # confidence = self.softmax(out[...,-1])    
        confidence_logits = out[...,-1]    

        return offset, confidence_logits

# def projection(self):
#     return self.K.dot(self.extrinsics)

# def extrinsics(self):
#     return np.hstack([self.R, self.t])

def undistort(X, meta, iter_num=5):
    # ref to matlab file: https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/matlab/kinoptic-tools/unproject.m
    # % Undistortion iterations.
    # % See cv::undistortPoints n modules/imgproc/src/undistort.cpp
    # % https://github.com/opencv/opencv

    # k contains:  k1,k2,p1,p2, k3
    # k = [reshape(cam.distCoeffs(1:5),[],1); zeros(12-5,1)]
    # k = [reshape(cam.distCoeffs(1:5),[],1); zeros(12-5,1)]
    distortion_param_batch = []
    camera_matrix_batch = []
    n_views = len(meta)
    device = X.device
    for n in range(n_views):
        k = meta[n]['camera']['k']
        p = meta[n]['camera']['p']
        k_p = torch.cat([k,p], 1) # 1,3,1 + 1,2,1 = 1,5,1
        # move k2 to the end (for opencv standard)
        # temp = k_p[:,2,:].copy()
        # k_p[:,2,:] = k_p[:,4,:]
        # k_p[:,4,:] = temp
        k_p_swap = torch.zeros_like(k_p)
        new_ind = [0,1,4,2,3]
        k_p_swap[:,new_ind,:]=k_p

        distortion_param_batch.append(k_p_swap)
        camera_matrix_batch.append(meta[n]['camera'])
    # batch_size, nviews, 5, 1
    # 5,1,5,1
    distortion_param_batch = torch.stack(distortion_param_batch)
    distortion_param_batch = distortion_param_batch.transpose(0,1)
    shape = distortion_param_batch.shape
    zeros_ = torch.zeros([shape[0],shape[1],7,shape[-1]]).to(device)
    distortion_param_batch = distortion_param_batch.to(device)
    distortion_param_batch = torch.cat([distortion_param_batch,zeros_], 2).to(device) #1,5,12,1

    ## normalize
    cam_batch = {}
    for k in meta[0]['camera'].keys():
        cam_batch[k] = []
    for v in range(n_views): ## views should be one
        for k, v in meta[v]['camera'].items():
            cam_batch[k].append(v)
    for k in meta[0]['camera'].keys():
        cam_batch[k] = torch.stack(cam_batch[k], dim=1)
    calib_matrix = get_calib_matrix(cam_batch, device=device)
    ones_shape = list(X.shape)
    ones_shape[-1] = 1
    p2d_homo = torch.cat([X, torch.ones(ones_shape).to(device)], dim=-1)
    # pn2d = ( calib_matrix / p2d_homo )
    nbins = p2d_homo.size(2)
    batchsize = p2d_homo.size(0)
    inv_calib_matrix_expand = calib_matrix.inverse().unsqueeze(2).expand(-1,-1,nbins,-1,-1) # 1,5,150,3,3
    p2d_homo_expand = p2d_homo.unsqueeze(-1)  # 1,5,150,3,1
    inv_calib_matrix_expand_view = inv_calib_matrix_expand.view(-1, nbins, 3,3)
    p2d_homo_expand_view = p2d_homo_expand.view(-1, nbins, 3,1)
    pn2d_view = torch.matmul(inv_calib_matrix_expand_view, p2d_homo_expand_view)  # should be 1,5,150,3,1
    pn2d = pn2d_view.view(batchsize, n_views, nbins, 3)

    # batch, views, nbins, 2
    x0=pn2d[...,0:1] # 1,5,150,1
    y0=pn2d[...,1:2] # 1,5,150,1
    x = x0
    y = y0

    k = distortion_param_batch.float() # 1,5,5,1
    for iter in range(0,iter_num):
        r2 = x*x + y*y # 1,5,150,1

        icdist = (1 + ((k[:,:,0+7,:].unsqueeze(2)*r2 + k[:,:,0+6,:].unsqueeze(2))*r2 + k[:,:,0+5,:].unsqueeze(2))*r2)/(1 + ((k[:,:,0+4,:].unsqueeze(2)*r2 + k[:,:,0+1,:].unsqueeze(2))*r2 + k[:,:,0+0,:].unsqueeze(2))*r2)

        deltaX = 2*k[:,:,0+2,:].unsqueeze(2)*x*y + k[:,:,0+3,:].unsqueeze(2)*(r2 + 2*x*x)+ k[:,:,0+8,:].unsqueeze(2)*r2+k[:,:,0+9,:].unsqueeze(2)*r2*r2

        deltaY = k[:,:,0+2,:].unsqueeze(2)*(r2 + 2*y*y) + 2*k[:,:,0+3,:].unsqueeze(2)*x*y+ k[:,:,0+10,:].unsqueeze(2)*r2+k[:,:,0+11,:].unsqueeze(2)*r2*r2

        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    pn2d_homo = torch.cat([x, y, torch.ones(ones_shape).to(device)], dim=-1)
    
    calib_matrix_expand = calib_matrix.unsqueeze(2).expand(-1,-1,nbins,-1,-1) # 1,5,150,3,3
    p2d_homo_expand = pn2d_homo.unsqueeze(-1)  # 1,5,150,3,1
    calib_matrix_expand_view = calib_matrix_expand.view(-1, nbins, 3,3)
    p2d_homo_expand_view = p2d_homo_expand.view(-1, nbins, 3,1)
    pn2d_view = torch.matmul(calib_matrix_expand_view, p2d_homo_expand_view)  # should be 1,5,150,3,1
    pn2d_output = pn2d_view.view(batchsize,n_views,nbins,3)[...,:2]
    return pn2d_output

# attention, this is a batch
def get_calib_matrix(camera, device):
    # (batch, views)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    
    cx = torch.as_tensor(camera['cx'], dtype=torch.float, device=device)
    cy = torch.as_tensor(camera['cy'], dtype=torch.float, device=device)

    batch, nviews = fx.shape[0], fx.shape[1]
    K = torch.zeros((batch,nviews,3,3), dtype=torch.float, device=device)
    K[:,:,0, 0], K[:,:,1, 1], K[:,:,0, 2], K[:,:,1, 2] = fx, fy, cx, cy
    K[:,:,2, 2] = 1

    return K

# inv_trans: please open it for CMU Panoptic dataset to correct the camera transform
def get_proj_matricies_batch(meta, nviews, device, inv_trans=False):
    # define camera
    cam_batch = {}
    for k in meta[0]['camera'].keys():
        cam_batch[k] = []
    for v in range(nviews): ## views should be one
        for k, v in meta[v]['camera'].items():
            cam_batch[k].append(v)
    for k in meta[0]['camera'].keys():
        cam_batch[k] = torch.stack(cam_batch[k], dim=1)

    R, T, f, c, k, p = cameras.unfold_camera_param_batch(cam_batch, device=device)
    calib_matrix = get_calib_matrix(cam_batch, device=device)

    if inv_trans:
        T = - R @ T
        # R = R.transpose(-1,-2)

    # proj_matricies_batch = K * [R|T]
    RT = torch.cat([R, T], -1) 
    proj_matricies_batch = calib_matrix.matmul(RT)

    # get proj matrix
    return proj_matricies_batch

class DQDecoderLayer(MvPDecoderLayer):
    def __init__(self, space_size, space_center,
                 img_size, pose_embed_layer, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 detach_refpoints_cameraprj=True,
                 fuse_view_feats='mean', n_views=5,
                 projattn_posembed_mode='use_rayconv',
                 feature_update_method='MLP',
                 init_self_attention=False,
                 open_forward_ffn=False,
                 query_filter_method='threshold',
                 visualization_jump_num=200,
                 bayesian_update=False,
                 triangulation_method='linalg',
                 filter_query=True):
        super().__init__(space_size, space_center,
                 img_size, d_model=d_model, d_ffn=d_ffn,
                 dropout=dropout, activation=activation,
                 n_levels=n_levels, n_heads=n_heads, n_points=n_points,
                 detach_refpoints_cameraprj=detach_refpoints_cameraprj,
                 fuse_view_feats=fuse_view_feats, n_views=n_views,
                 projattn_posembed_mode=projattn_posembed_mode)

        # projective attention
        self.proj_attn = ProjAttn(d_model, n_levels,
                                  n_heads, n_points,
                                  projattn_posembed_mode)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model,
                                               n_heads,
                                               dropout=dropout)
        self.feature_update_mlp = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.grid_size = torch.tensor(space_size)
        self.grid_center = torch.tensor(space_center)

        self.img_size = img_size

        self.detach_refpoints_cameraprj = detach_refpoints_cameraprj

        self.pose_embed = offset_net(d_model, d_model, pose_embed_layer)
        # confidence softmax
        self.softmax_conf = nn.Softmax(dim=0)
        
        self.open_bayesian_update = bayesian_update
        # print(' * Baysian Update:', self.open_bayesian_update)
        if self.open_bayesian_update:
            self.bayesian_conf = nn.Linear(d_model, 1)  # output should be 0,1

        self.use_confidences = False

        num_classes = 2
        self.class_embed = nn.Linear(d_model, num_classes)
        self.num_joints = 15

        self.feature_update_method = feature_update_method
        self.init_self_attention = init_self_attention
        self.open_forward_ffn = open_forward_ffn
        self.query_filter_method = query_filter_method

        self.visualization_jump_num = visualization_jump_num
        
        self.triangulation_method = triangulation_method # 'linalg'
        
        self.filter_query = filter_query


    # Output: normalized projected points
    def project_ref_points(self, reference_points,
        meta, 
        nviews,
        batch_size,
        nbins,
        device):

        if self.detach_refpoints_cameraprj:
            reference_points = reference_points.detach()

        cam_batch = {}
        for k in meta['camera'].keys():
            cam_batch[k] = []
        for k, v in meta['camera'].items():
            cam_batch[k].append(v)
        for k in meta['camera'].keys():
            cam_batch[k] = torch.stack(cam_batch[k], dim=1)

        reference_points_expand = reference_points.\
            unsqueeze(1).expand(-1, nviews, -1, -1, -1)
        reference_points_expand_flatten \
            = reference_points_expand\
            .contiguous().view(batch_size, nviews, nbins, 3)

        # reference_points_absolute = self.\
            # norm2absolute(reference_points_expand_flatten)
        reference_points_absolute = reference_points_expand_flatten
        reference_points_projected2d_xy = \
            cameras.project_pose_batch(reference_points_absolute, cam_batch)

        trans_batch = []
        for i in range(batch_size):
            temp = []
            temp.append(
                torch.as_tensor(
                    get_transform(meta['center'][i],
                                    meta['scale'][i],
                                    0, self.img_size),
                    dtype=torch.float,
                    device=device))
            trans_batch.append(torch.stack(temp))
        trans_batch = torch.stack(trans_batch)

        wh = meta['center'].unsqueeze(1) *2  # reference_points_projected2d_xy: 8, 1, 960, 2;  wh: 8,2
        bounding \
            = (reference_points_projected2d_xy[..., 0] >= 0) \
            & (reference_points_projected2d_xy[..., 1] >= 0) \
            & (reference_points_projected2d_xy[..., 0] < wh[..., 0:1]) \
            & (reference_points_projected2d_xy[..., 1] < wh[..., 1:2])

        # projected points will be clamped to be inside images
        reference_points_projected2d_xy \
            = torch.clamp(reference_points_projected2d_xy, -1.0, wh.max())

        ## transform to network image coordinate
        reference_points_projected2d_xy \
            = do_transform_batch(reference_points_projected2d_xy, trans_batch)  ## project ref/attention points

        ## img_size: network image
        reference_points_projected2d_xy \
            = reference_points_projected2d_xy \
            / torch.tensor(self.img_size, dtype=torch.float, device=device)

        ref_points_expand = reference_points_projected2d_xy\
            .flatten(0, 1).unsqueeze(2)

        return ref_points_expand, bounding

    def learnable_triangulate(self,keypoints_2d,projs_2d,
        meta,batch_size,n_views,device,confidence=None,meta_origin=None,indices_stgt=None):
        '''
        Learnable triangulation to recover 3D poses from refined 2D poses, in a differentiable way.

        input:
            @keypoints_2d: 2d pose, n_views, 1 (batch), 150, 2
            @projs_2d: 2d pose, n_views, 1 (batch), 150, 2
            
            @meta_origin: Temperally used for debugging groundtruth bone length. Will be abandoned soon.
            @indices_stgt: only used for ST-GT
        '''
        keypoints_2d_netim = keypoints_2d

        # inverse transform to original image for camera parameters
        trans_batch = []
        for n in range(n_views):
            trans_batch.append(meta[n]['inv_affine_trans'][:,:2,:])
        trans_batch = torch.stack(trans_batch)
        trans_batch = torch.transpose(trans_batch, 0,1).float()
        keypoints_2d_originim \
            = do_transform_batch(keypoints_2d_netim, trans_batch)  ## project ref/attention points

        keypoints_2d_originim_undistort = undistort(keypoints_2d_originim, meta, iter_num=5)

        keypoints_2d_undistort = keypoints_2d_originim_undistort

        alg_confidences = confidence

        proj_matricies_batch = get_proj_matricies_batch(meta, n_views, device, inv_trans=True)

        method = self.triangulation_method
        # t1 = time.time()
        if method == 'batch':
            keypoints_3d = multiview.triangulate_batch_of_points_batch_version(
                    proj_matricies_batch, keypoints_2d_undistort,
                    confidences_batch=alg_confidences
                )
        # t2 = time.time()

        # speed up: we only calculate triangulation for those queries with association
        elif method == 'default':
            keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matricies_batch, keypoints_2d_undistort,
                    confidences_batch=alg_confidences
                )
        # t3 = time.time()

        # cpu version
        elif method == 'cpu':
            keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matricies_batch, keypoints_2d_undistort,
                    confidences_batch=alg_confidences,
                    device='cpu'
                )
        # t4 = time.time()

        elif method == 'linalg':
            keypoints_3d = multiview.triangulate_batch_of_points_batch_version(
                    proj_matricies_batch, keypoints_2d_undistort,
                    confidences_batch=alg_confidences,
                    solver='linalg'
                )
            
        elif method == 'st':
            # Structural Triangulation
            keypoints_3d = structural_triangulate_points(
                    proj_matricies_batch, keypoints_2d_undistort,
                    confidences_batch=alg_confidences
                )
        
        elif method == 'st-gt':
            # use gt bone length
            # Extract gt poses from meta: n_poses, 15, 3
            poses_3d_gt = meta_origin[0]['joints_3d']
            n_person = meta_origin[0]['num_person']
            batch = len(n_person)
            poses_3d_gt_valid_batch_list = []
            for b in range(batch):
                poses_3d_gt_valid_batch = poses_3d_gt[b,...][:n_person[b]]
                poses_3d_gt_valid_batch_list.append(poses_3d_gt_valid_batch)
            poses_3d_gt_all = torch.cat(poses_3d_gt_valid_batch_list, dim=0)
            poses_3d_gt_all_flat = poses_3d_gt_all.reshape(poses_3d_gt_all.shape[0], -1)
            
            human_tree = create_human_tree(data_type = "cmupanoptic")
                
            # batch, 15, 1
            G = torch.from_numpy(human_tree.conv_J2B.T).to(device)
            bones = poses_3d_gt_all_flat @ G
            bones = bones.reshape(-1, 15, 3)
            bone_length_gt = bones.norm(dim=2)[:,1:].unsqueeze(-1)
            
            total_inds = []
            for batchid in range(len(indices_stgt)):
                indids = indices_stgt[batchid][1]  #0 query, 1 gt
                for indid in indids:
                    final_ind = n_person[:batchid].sum()+indid
                    total_inds.append(final_ind)
            if len(total_inds) > 0:
                total_inds = torch.stack(total_inds)
                bone_length = bone_length_gt [ total_inds ]
            
                # Debug: test close confidence
                alg_confidences = None
                
                keypoints_3d = structural_triangulate_points(
                        proj_matricies_batch, keypoints_2d_undistort,
                        confidences_batch=alg_confidences,
                        bone_length=bone_length,
                        batch=False
                    )
            else:
                bs, views, joints, dims = keypoints_2d_undistort.shape
                keypoints_3d = keypoints_2d_undistort.new_zeros((bs,joints,3))

        return keypoints_3d

    def generate_features(self, src_views, tgt, query_pos, src_padding_mask, reference_points,
            meta, src_spatial_shapes, level_start_index, src_views_with_rayembed=None):
        '''
        Use projective attention to aggregate features from different views.
        For each projected joints, we sample attention points nearby to aggregate features.
        '''
        batch_size = tgt.shape[0]
        device = tgt.device
        # h, w = src_spatial_shapes[0]
        nfeat_level = len(src_views)
        nbins = reference_points.shape[1]

        nviews = len(src_views[0]) // batch_size  ## batch, views, ...
        attn_feature_views = []
        ref_points_norm_views = []

        if self.init_self_attention:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1),
                                k.transpose(0, 1),
                                tgt.transpose(0, 1))[0].transpose(0, 1)

            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        ## expand input for multiple views.
        nviews_this = 1
        tgt_expand = tgt.unsqueeze(1).\
            expand((-1, nviews_this, -1, -1)).flatten(0, 1)
        if query_pos is not None:
            query_pos_expand = query_pos.unsqueeze(1).\
                expand((-1, nviews_this, -1, -1)).flatten(0, 1) ## 5,150,256
        else:
            query_pos_expand = None
        src_padding_mask_expand = torch.cat(src_padding_mask, dim=1)


        for n in range(nviews):
            # deal with each view
            ## spatial projection attention
            ## src_views: (n_views, C, H, W), size n_levels(3)
            ## src_views_this_view: (C,H,W), size n_levels;
            ## n_levels: 3, C: feature channels 256
            ## n:(n+1), keep dim
            src_views_this_view = [src[n * batch_size:(n * batch_size + batch_size),:,:,:] for src in src_views]

            # Project 3D query points into 2D images of each views
            ref_points_2d_norm, bounding = self.project_ref_points(reference_points,
                meta[n], 
                nviews_this,
                batch_size,
                nbins,
                device)
            
            ref_points_2d_norm_expand \
                = ref_points_2d_norm.expand(-1, -1, nfeat_level, -1) \
                * src_spatial_shapes.flip(-1).float() \
                / (src_spatial_shapes.flip(-1)-1).float()

            ## proj_attn for each view, all instances, all key points
            # src_views_with_rayembed = None
            src_padding_mask_expand = None ## TODO: no mask
            # print("* close mask.")
            attn_feature = self.proj_attn(
                self.with_pos_embed(tgt_expand, query_pos_expand), ref_points_2d_norm_expand, src_views_this_view, 
                src_views_with_rayembed, ## TODO: Check 
                src_spatial_shapes, level_start_index, src_padding_mask_expand)

            # filtering points out of the images with bounding
            attn_feature_filter = (bounding.unsqueeze(-1) * 
                    attn_feature.view(batch_size, nviews_this, nbins, -1))
                
            attn_feature_filter = attn_feature_filter.squeeze(1)

            ## store feature for this view
            attn_feature_views.append(attn_feature_filter)
            ref_points_norm_views.append(ref_points_2d_norm)
        return attn_feature_views, ref_points_norm_views


    def generate_valid_masks(self, outputs_class, method='threshold', value=0.5):
        # @larger: positive is larger than negative
        # @threshold: positive value is larger than threshold
        if method == 'larger':
            _, preds = outputs_class.topk(1)
            batch_ids, query_ids, _ = torch.where(preds) #?
        elif method == 'threshold':
            if value is None:
                raise ValueError('please specify value')
            preds = outputs_class[...,1] > value
            batch_ids, query_ids = torch.where(preds)
        elif method == 'all':
            # close filtering, calculate all the position
            preds_all = outputs_class[...,0] > 0
            batch_ids, query_ids = torch.where(preds_all)

        return batch_ids, query_ids


    def padding_query_with_mask(self, batch_ids, query_ids, batch_size, mask_ids=0):
        # padding untils each batch_id to same number of query
        dtype = batch_ids.dtype
        device = batch_ids.device
        # check for empty query
        if batch_ids.nelement() == 0:
            #! make sure that there is always a query
            batch_ids = torch.tensor([0],dtype = dtype,device = device)
            query_ids = torch.tensor([0],dtype = dtype,device = device)

        batch_ids_count = batch_ids.bincount(minlength = batch_size)
        batch_ids_max_count = batch_ids_count.max()
        padding_num  = batch_ids_max_count - batch_ids_count

        batch_ids_padding = torch.cat([
            torch.tensor(batch_id,dtype=dtype,device=device).repeat(padding_num) 
            for batch_id,padding_num in enumerate(padding_num)
        ])
        
        query_ids_padding = torch.cat(
            [torch.tensor(mask_ids,dtype=dtype,device=device).repeat(padding_num) 
            for batch_id,padding_num in enumerate(padding_num)
        ])

        batch_ids_padded = torch.cat((batch_ids,batch_ids_padding))
        query_ids_padded = torch.cat((query_ids,query_ids_padding))

        # sort batch_ids and query_ids
        batch_ids_padded, indices = torch.sort(batch_ids_padded, stable=True)
        query_ids_padded = query_ids_padded[indices]
        
        batch_ids_rev = torch.cat([
            torch.tensor(batch_id,dtype=dtype,device=device).repeat(count) 
            for batch_id,count in enumerate(batch_ids_count)
        ])
        
        query_ids_rev = torch.cat([
            torch.tensor(range(count),dtype=dtype,device=device)
            for count in batch_ids_count
        ])
        
        return batch_ids_padded, query_ids_padded, batch_ids_rev ,query_ids_rev


    def calculate_2d_offsets(self, nviews, device, attn_feature_views, ref_points_expand_views, rgb_views):
        '''
        From aggregated features, calculate 2d offsets for each projected 2D joints.
        
        TODO: Further speed up with batch processing.
        '''
        projs_2d = []
        offsets_2d = []
        refined_2d_poses = []
        conf_logits_list = []
        bayesian_conf_list = []

        for n in range(nviews):
            if self.pose_embed is not None: 
                ## get 2d offset  (pose_embed has been updated)
                ## 2 (batch_size), 90 (queries_num, 15), 256
                attn_feature = attn_feature_views[n]
                tmp_2d_offset_abs, conf_logits = self.pose_embed(attn_feature)

                tmp_2d_offset = tmp_2d_offset_abs / torch.tensor(self.img_size, dtype=torch.float, device=device)

                ## refined 2d pos
                ## get projected 2d pos from reference_points, in this view
                ref_points_expand = ref_points_expand_views[n]
                proj_2d = ref_points_expand.squeeze(2)
                refined_2d = proj_2d + tmp_2d_offset

                ## store 2d offsets to mat
                offsets_2d.append(tmp_2d_offset)
                refined_2d_poses.append(refined_2d)
                projs_2d.append(proj_2d)
                conf_logits_list.append(conf_logits)
                
                if self.open_bayesian_update:
                    bayesian_conf = self.bayesian_conf(attn_feature).sigmoid()
                    bayesian_conf_list.append(bayesian_conf)
         
        refined_2d_poses = torch.cat(refined_2d_poses, dim=0)

        refined_2d_poses_absolute = refined_2d_poses * torch.tensor(self.img_size, dtype=torch.float, device=device)

        projs_2d_poses = torch.cat(projs_2d, dim=0)
        projs_2d_poses_abs = projs_2d_poses * torch.tensor(self.img_size, dtype=torch.float, device=device)

        # batch, people_num, joints
        confidences_logits = torch.cat(conf_logits_list, dim=0)
        batch_size = attn_feature_views[0].size(0)
        confidences_logits = confidences_logits.view(nviews,batch_size,-1,self.num_joints)
        confidences = self.softmax_conf(confidences_logits)
        
        if self.open_bayesian_update:
            bayesian_conf_tensor = torch.cat(bayesian_conf_list, dim=0)
            bayesian_conf_tensor_view = bayesian_conf_tensor.view(nviews,batch_size,-1,self.num_joints)
            # average around nviews.
            bayesian_conf_tensor_aver = bayesian_conf_tensor_view.mean(0)
        else:
            bayesian_conf_tensor_aver = None
            
        return refined_2d_poses_absolute, confidences, projs_2d_poses_abs, bayesian_conf_tensor_aver

    def visualize_func(self, device, output_dir, nviews, rgb_views,
                src_views,
                attn_feature_views,
                refined_2d_poses_absolute,
                projs_2d_absolute,
                frame_id):
        '''
        A visualization utils function to output mid results.
        
        Visualization includes: each input view, projected points, offsets, refined 2d points.
        
        '''

        if frame_id is None:
            frame_id = 0
        
        config_jump_save_image = self.visualization_jump_num
        open_visualization = (config_jump_save_image >= 0)

        if open_visualization:
            if frame_id % config_jump_save_image == 0 and frame_id > 0:
                # final_output_dir = '{}/{}'.format(output_dir,frame_id)
                # final_output_dir = './debug/layers/'
                # prefix2 = os.path.join(final_output_dir, 'images')

                # # projs_2d_absolute = projs_2d.matmul(torch.diag(origin_rgb_spatial_shapes).float())
                # offsets_2d = torch.cat(offsets_2d, dim=0)
                # # offsets_2d_absolute = offsets_2d.matmul(torch.diag(origin_rgb_spatial_shapes).float())
                # offsets_2d_absolute = offsets_2d * torch.tensor(self.img_size, dtype=torch.float, device=device)
                prefix2 = output_dir

                visualize_proj_attention(
                    nviews,
                    rgb_views,
                    src_views,
                    attn_feature_views,
                    projs_2d_absolute, ## normallized
                    None, # offsets_2d_absolute,  ## normalized
                    refined_2d_poses_absolute,
                    save_dir=prefix2,
                    config_vis_refined=True,
                    draw_line=True
                )

    def update_feature(self, tgt, attn_feature_views, query_pos=None):
        '''
        Update query features with aggregated features from different views.
        '''
        # tgt: bs, 960, 256
        # 5 views, 960 queries x joints, 256 dims
        # average pooling
        attn_features_aver = torch.stack(attn_feature_views, 0).mean(0)

        method = self.feature_update_method
        if method == 'MLP':
            tgt2 = self.feature_update_mlp(attn_features_aver)

            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)
        elif method == 'attention':
            q = k = attn_features_aver
            # TODO: Check transpose
            tgt2, weights = self.self_attn(q, k, tgt)
            ## tgt2: 1,150,256
            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)
        elif method == 'attention_tgt':
            # update: fix the bug in attention
            q = k = attn_features_aver
            tgt2, weights = self.self_attn(q, k, attn_features_aver)  # tgt to attn_features_aver
            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)
        elif method == 'attention_tgt_trans':
            # update: fix the bug in attention
            q = k = attn_features_aver
            tgt2 = self.self_attn(q.transpose(0, 1),
                    k.transpose(0, 1),
                    attn_features_aver.transpose(0, 1))[0].transpose(0, 1)

            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)
        elif method == 'attention_tgt_embed':
            # add query embeddings
            q = k = self.with_pos_embed(attn_features_aver, query_pos)
            tgt2, weights = self.self_attn(q, k, attn_features_aver)  # tgt to attn_features_aver
            
            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)

        elif method == 'attention_tgt_embed_trans':
            q = k = self.with_pos_embed(attn_features_aver, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1),
                                k.transpose(0, 1),
                                attn_features_aver.transpose(0, 1))[0].transpose(0, 1)
            
            ## add and norm
            tgt_update = tgt + self.dropout2(tgt2)
            tgt_update = self.norm2(tgt_update)
        elif method == 'attention_tgt_embed_trans_direct':
            q = k = self.with_pos_embed(attn_features_aver, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1),
                                k.transpose(0, 1),
                                attn_features_aver.transpose(0, 1))[0].transpose(0, 1)
            
            ## add and norm
            tgt_update = self.dropout2(tgt2) # + tgt
            tgt_update = self.norm2(tgt_update)
        elif method == 'MLP0': # clear MLP
            tgt_update = self.feature_update_mlp(attn_features_aver)
        elif method == 'MLPr': # clear MLP
            tgt2 = self.feature_update_mlp(attn_features_aver)
            tgt_update = tgt + self.dropout2(tgt2)
        elif method == 'mean':
            # no MLP is needed for the baseline!
            tgt2 = attn_features_aver.mean(1)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            tgt_update = tgt

        # forward_ffn
        if self.open_forward_ffn:
            tgt_update = self.forward_ffn(tgt_update)

        return tgt_update

    def forward(self, tgt, query_pos, reference_points, src_views,
                src_spatial_shapes,
                level_start_index, meta, src_padding_mask=None, rgb_views = None, 
                output_dir='./', frame_id = None, indices=None, threshold=0.5, indices_all=None):
        '''
        During each decoder layer, each 3D poses of each queries will be projected into each camera view to aggregate features and
        update coarse projected 2D poses. Then, a triangulation will be performed to get updated 3D poses for each queries. Also,
        the feature of each query will be updated with the aggregated features. 

        Args:
            @tgt: query features
            @query_pos: query position embeddings
            @reference_points: 3D points of queries
            @src_views: source views
            @src_spatial_shapes: spatial shapes of source views
            @level_start_index: start index of each level
            @meta: meta information of cameras
        '''

        time0 = time.time()
        
        ## 1. Calculate Projection Attention Features
        ## attn_feature_views: attention features in each view images, 
        ## ref_points_expand_views: projected 2D query poses
        num_all_points = tgt.shape[1]
        attn_feature_views, ref_points_expand_views = self.generate_features(src_views, tgt, query_pos, src_padding_mask, reference_points,
            meta, src_spatial_shapes, level_start_index)

        time0_hf = time.time()
        time_genearte_feat.update(time0_hf-time0)

        # 2. Update qeury features
        tgt_update = self.update_feature(tgt, attn_feature_views, query_pos)

        time1 = time.time()
        time_features.update(time1-time0_hf)

        # Triangulation is time-consuming, so we filter most of queries according to the confidence.
        # 3. Filter: Generate Valid Masks from attn_feature_views
        outputs_class = self.class_embed(tgt_update)
        batch, num_points, class_dims = outputs_class.shape
        outputs_class_prob = outputs_class.\
                view(batch, -1, self.num_joints, class_dims).\
                sigmoid().mean(2)
        # outputs_class = inverse_sigmoid(outputs_class)

        # Now decide which parts of queries can go through triangulation:
        #   For training, indices is set from groundturth poses;
        #   For validation, indices are calculated from output_classes.
        if self.filter_query:
            if indices is not None:
                batch_ids, query_ids = generate_batch_query_ids(indices)
            else:
                # For val, or training without gt match
                batch_ids, query_ids = self.generate_valid_masks(outputs_class_prob, method=self.query_filter_method, value=threshold)
                # print('Process queries:', query_ids.shape)
        else:
            # Keep all queries
            batch_ids, query_ids = self.generate_valid_masks(outputs_class_prob, method='all')

        # Debug Code
        indices_stgt = None
        if indices_all is None:
            reference_points_valid = reference_points.view(batch, -1, self.num_joints, 3)[batch_ids,query_ids,...] # TODO: filter valid as new_ref_points
            indices_stgt = self.match_ref_points_to_gt(reference_points_valid, batch_ids, query_ids, meta)
        else:
            indices_stgt = indices_all
        
        time2 = time.time()
        time_prob.update(time2-time1)
        
        # Padding Query with Mask
        # Goal: Filtering invalid queries to speed up, and at the same time, keep the data structure
        # Eche batch_ids has same query_num => [batch_size, ..., query_num, ...]
        batch_ids, query_ids,batch_ids_rev ,query_ids_rev = self.padding_query_with_mask(batch_ids,query_ids,batch)

        attn_feature_views = retrieve_valid(attn_feature_views, batch_ids, query_ids, self.num_joints, input_list=True)
        ref_points_expand_views = retrieve_valid(ref_points_expand_views, batch_ids, query_ids, self.num_joints, input_list=True)

        # 4. Calculate 2D Offsets
        batch_size = batch
        nviews = len(src_views[0]) // batch_size ## batch, views, ...
        device = tgt.device
        refined_2d_poses_absolute, confidences, projs_2d_poses_abs, bayesian_conf_tensor_aver = self.calculate_2d_offsets(nviews, device, attn_feature_views, ref_points_expand_views, rgb_views)

        # Filter orginal tensor -> [sum_n_query,n_view,self.num_joints,...]
        refined_2d_poses_absolute_spite = refined_2d_poses_absolute.view(nviews,batch_size,-1,self.num_joints,class_dims).transpose(0,1)
        projs_2d_poses_abs_spite = projs_2d_poses_abs.view(nviews,batch_size,-1,self.num_joints,class_dims).transpose(0,1)
        confidences_spite = confidences.view(nviews,batch_size,-1,self.num_joints).transpose(0,1)

        new_refined_2d_poses_absolute = refined_2d_poses_absolute_spite[batch_ids_rev,:,query_ids_rev,...]
        new_projs_2d_poses_abs = projs_2d_poses_abs_spite[batch_ids_rev,:,query_ids_rev,...]
        confidences_filter = confidences_spite[batch_ids_rev,:,query_ids_rev,...]
        if self.open_bayesian_update:
            bayesian_conf_tensor_aver_spite = bayesian_conf_tensor_aver
            bayesian_conf_tensor_aver_spite_filter = bayesian_conf_tensor_aver_spite[batch_ids_rev,query_ids_rev,...]
        
        #! Filter meta with batch_ids_rev 
        meta_batch = []
        for mt in meta:
            mt_batch = {}
            for k, v in mt.items():
                if isinstance(v,torch.Tensor):
                    mt_batch[k] = v[batch_ids_rev]
                elif k =='image':
                    mt_batch[k] = [v[batch_id] for batch_id in batch_ids_rev]
                elif k =='camera':
                    mt_batch[k] = {}
                    for ck,cv in v.items():
                        mt_batch[k][ck] = cv.to(device)[batch_ids_rev]
                else:
                    print(f'{k} is missing')
            meta_batch.append(mt_batch)
        
        time3 = time.time()
        time_padding.update(time3-time2)
        
        ## TODO: check refined_2d_poses, for those outside of the image plane, just use the origin one without undistort
        new_reference_points_triangu = self.learnable_triangulate(new_refined_2d_poses_absolute,new_projs_2d_poses_abs,meta_batch,
            batch_size,nviews,device,confidence=confidences_filter,meta_origin=meta,indices_stgt=indices_stgt)

        time4 = time.time()
        time_triangulate.update(time4-time3)
        
        # Debug Code
        if self.open_bayesian_update:
            reference_points_prior = reference_points.view(batch_size, -1, self.num_joints, 3)[batch_ids_rev,query_ids_rev,...] # TODO: filter valid as new_ref_points
            # weights # which one to trust most? depend on the 2d confidence
            new_reference_points = self.bayesian_update(new_reference_points_triangu, reference_points_prior, bayesian_conf_tensor_aver_spite_filter)
        else:
            new_reference_points = new_reference_points_triangu
            
        # 4. Visualization
        # Visualization for different batch 
        config_jump_save_image = self.visualization_jump_num
        open_visualization = (config_jump_save_image >= 0)
        if open_visualization:
            for batch_id in range(batch_size):
                # select views, attn_feature_views and poses for 'batch_id'
                query_index = range(batch_id*nviews,(batch_id+1)*nviews)
                vis_rgb_views = [rgb_view[batch_id] for rgb_view in rgb_views]
                vis_src_views = [src_view[query_index] for src_view in src_views]
                vis_attn_feature_views = [attn_feature_view[batch_id] for attn_feature_view in attn_feature_views]
                vis_refined_2d_poses_absolute = refined_2d_poses_absolute_spite[batch_id] # 40, 105, 2 -> batch_size x views, 7x15, 2
                vis_projs_2d_poses_abs = projs_2d_poses_abs_spite[batch_id]
                vis_output_dir = f"{output_dir}/batch_{batch_id}"
                self.visualize_func(device, vis_output_dir, nviews, vis_rgb_views,
                                vis_src_views,
                                vis_attn_feature_views,
                                vis_refined_2d_poses_absolute,
                                vis_projs_2d_poses_abs,
                                frame_id)

        # For both training and val
        # Expand the size of reference_points, refined_2d_poses_absolute, projs_2d_poses_abs 
        #   to [batch_size,(nviews),num_max_query*num_joints,...]
        num_joints = self.num_joints
        num_query = num_all_points // num_joints
        new_reference_points_expand = torch.zeros((batch_size,num_query,num_joints,3), dtype=new_reference_points.dtype, device=new_reference_points.device)
        new_refined_2d_poses_absolute_expand = torch.zeros((batch_size,nviews,num_query,num_joints,2), dtype=new_reference_points.dtype, device=new_reference_points.device)
        new_projs_2d_poses_abs_expand = torch.zeros((batch_size,nviews,num_query,num_joints,2), dtype=new_reference_points.dtype, device=new_reference_points.device)
        # rev_len = batch_ids_rev.shape[0]
        # batch_ids = batch_ids[:rev_len]
        # query_ids = query_ids[:rev_len]
        batch_ids_valid = batch_ids.view(batch_size, -1)[batch_ids_rev, query_ids_rev]
        query_ids_valid = query_ids.view(batch_size, -1)[batch_ids_rev, query_ids_rev]
        
        new_reference_points_expand[batch_ids_valid, query_ids_valid, ...] = new_reference_points
        new_reference_points = new_reference_points_expand.flatten(1,2)
        
        new_refined_2d_poses_absolute_expand[batch_ids_valid,:, query_ids_valid, ...] = new_refined_2d_poses_absolute
        new_refined_2d_poses_absolute = new_refined_2d_poses_absolute_expand.flatten(2,3)
        
        new_projs_2d_poses_abs_expand[batch_ids_valid,:, query_ids_valid, ...] = new_projs_2d_poses_abs
        new_projs_2d_poses_abs = new_projs_2d_poses_abs_expand.flatten(2,3)

        time5 = time.time()
        time_finalprocess.update(time5-time4)
        
        # N_time_output = 100
        # N_time_output = 10000
        # if time_features.count % N_time_output == 0:
        #     print('* time_genearte_feat: ', time_genearte_feat.avg)
        #     print('* time_features_update:', time_features.avg)
        #     print('* time_prob:', time_prob.avg)
        #     print('* time_padding:', time_padding.avg)
        #     print('* time_triangulate:', time_triangulate.avg)
        #     print('* time_finalprocess:', time_finalprocess.avg)
        
        
        return tgt_update, new_reference_points, new_refined_2d_poses_absolute, new_projs_2d_poses_abs, outputs_class_prob

    def bayesian_update(self, new_reference_points, reference_points_prior, confidences_filter):
        # weight of prior: 1 - conf
        # weight of obs: conf
        confidences_filter_unsq = confidences_filter.unsqueeze(-1)
        output = confidences_filter_unsq * new_reference_points + (1- confidences_filter_unsq) * reference_points_prior
        
        return output
    
    def match_ref_points_to_gt(self, reference_points, batch_ids, query_ids, meta):
        '''
        OUTPUT: 
            [batch 0: query_ids, gt_ids.
            batch 1...
            ...
            batch N...]
        '''
        
        num_points = len(reference_points)
        gt_3d_points = meta[0]['joints_3d']
        gt_persons = meta[0]['num_person']
        
        # output_batch = []
        # output_ind = []
        
        batch_matches = []
        if batch_ids.numel() > 0:
            for i in range(batch_ids.max()+1):
                batch_matches.append([[], []])
            for n in range(num_points):
                pts = reference_points[n]
                
                corresponding_batch = batch_ids[n]
                corresponding_gt_pts = gt_3d_points[corresponding_batch][:gt_persons[corresponding_batch]]
                
                # match!
                dist = (corresponding_gt_pts-pts).square().sum(dim=-1).sum(dim=-1)
                gt_id = torch.argmin(dist)
                
                # if not corresponding_batch in batch_matches:
                    # batch_matches[corresponding_batch] = [[], []]  # Init
                
                query_id = query_ids[n]
                batch_matches[corresponding_batch][0].append(query_id.item())
                batch_matches[corresponding_batch][1].append(gt_id.item())
        
        return batch_matches
            
        
        # match for each valid ref points.

class DQDecoder(MvPDecoder):
    def __init__(self, cfg, decoder_layer,
                 num_layers, return_intermediate=False):
        super().__init__(cfg, decoder_layer,
                 num_layers, return_intermediate)

    def forward(self, tgt, reference_points, src_views,
                meta, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, rgb_views=None, 
                output_dir='./', frame_id=None, indices=None, threshold = 0.5, indices_all=None):
        '''
            A decoder forward function, that contains multiple decoder layers.
            Most of the the important stuff is written in DQDecoderLayer.

            Args:
                tgt: query feature tensor, (batch, num_points, feature_dims)
                reference_points: 3D reference points, (batch, num_points, 3)
                src_views: list of source views
                meta: meta information of cameras
                src_spatial_shapes: spatial shapes of source views
                src_level_start_index: start index of each level
                src_valid_ratios: valid ratios of source views
                query_pos: query position embedding tensor
        '''

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_points_2d = []
        intermediate_reference_points_2d_projs = []
        outputs_class_list = []

        query_pos_in = query_pos
        for lid, layer in enumerate(self.layers):
            output_dir_in = output_dir + f'/{frame_id}/layer-{lid}/'
            reference_points_input = reference_points[:, :, None]
            
            # Count query num
            batch, query_num_mul_joints = output.shape[:2]
            query_num = round(query_num_mul_joints / 15)
            count_query_nums[lid].update(batch * query_num)
            
            output, reference_points, ref_points_2d, projs_2d_absolute, outputs_class = layer(output, query_pos_in, reference_points_input,
                           src_views,
                           src_spatial_shapes,
                           src_level_start_index, meta, src_padding_mask,
                           rgb_views=rgb_views,output_dir=output_dir_in,
                           frame_id=frame_id,
                           indices=indices,
                           threshold=threshold,
                           indices_all=indices_all)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_points_2d.append(ref_points_2d)
                intermediate_reference_points_2d_projs.append(projs_2d_absolute)
                outputs_class_list.append(outputs_class)

        # if count_query_nums[0].count % 100 == 0:
        #     for layer_i, counter in enumerate(count_query_nums):
        #         print(f'query-layer-{layer_i}:', counter.avg)

        if self.return_intermediate:
            return torch.stack(intermediate), \
                   torch.stack(intermediate_reference_points), \
                   torch.stack(intermediate_reference_points_2d), \
                   torch.stack(intermediate_reference_points_2d_projs), \
                    outputs_class_list

        return output, reference_points, ref_points_2d

def retrieve_valid(output, batch_ids, query_ids, num_joints, input_list=False):
    # considering output is a list
    if not input_list:
        output = [output]
    data_list = []
    for data in output:
        shape = list(output[0].shape)
        shape[1] = -1
        # split nums into query_num * num_joints

        # directly select the second dims!
        # (num_joints, )  + (num_queries, ) => (num_joints * num_queries, )
        dim_index = ( query_ids.unsqueeze(1) * num_joints + torch.arange(num_joints).cuda() ).reshape(-1)
        batch_ids_expand = batch_ids.repeat_interleave(num_joints)
        # data = data.view(batch, -1, num_joints, dims)
        data = data[batch_ids_expand, dim_index, ...]
        # resize back
        data = data.view(shape)
        # output = output.unsqueeze(0)
        data_list.append(data)

    if not input_list:
        return data_list[0]
    else:
        return data_list


def filter_with_prob(output, reference_points, query_pos_in, outputs_class, num_joints):
    _, preds = outputs_class.topk(1)
    batch_ids, query_ids, _ = torch.where(preds)

    output = retrieve_valid(output, batch_ids, query_ids, num_joints)
    reference_points = retrieve_valid(reference_points, batch_ids, query_ids, num_joints)
    query_pos_in = retrieve_valid(query_pos_in, batch_ids, query_ids, num_joints)

    return output, reference_points, query_pos_in


def generate_batch_query_ids(indices):
    batch_ids = []
    query_ids = []
    for batch_id,querys in enumerate(indices):
        batch_ids.extend([batch_id]*len(querys))
        query_ids.extend(querys)
    batch_ids = torch.tensor(batch_ids,dtype=indices[0].dtype,device = indices[0].device)
    query_ids = torch.tensor(query_ids,dtype=batch_ids.dtype,device = batch_ids.device)
    return batch_ids, query_ids
