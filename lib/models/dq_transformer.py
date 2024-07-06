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
# and Deformable Detr (https://github.com/fundamentalvision/Deformable-DETR)
# ----------------------------------------------------------------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nis import cat

import torch
import torch.nn as nn
from lib.core.function import time_synchronized

from models import pose_resnet

from models.dq_decoder import DQDecoder, DQDecoderLayer
from models.ops.modules import ProjAttn
from torch.nn.init import xavier_uniform_, constant_, normal_
from models.position_encoding import PositionEmbeddingSine, \
    get_rays_new, get_2d_coords
from models.util.misc import (
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import copy
import torch.nn.functional as F
from models.matcher import HungarianMatcher
from core.loss import PerJointL1Loss, PerBoneL1Loss, PerProjectionL1Loss

import math

from models.multi_view_pose_transformer import MultiviewPosetransformer,MLP,_get_clones
from lib.utils.vis import *
import os

import time

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

time_backbone = AverageMeter()
time_preprocess = AverageMeter()
time_init_ref = AverageMeter()
time_decoder_layers = AverageMeter()
time_final_process = AverageMeter()

def construct_output_from_origin(init_reference, device, calculate_2d=False):
    output = {}
    num_classes = 2
    num_people = init_reference.shape[0]
    num_queries_m_joints = init_reference.shape[1]
    num_queries = round(num_queries_m_joints / 15)
    output['pred_logits'] = torch.ones((num_people,num_queries,num_classes)).to(device)

    output['pred_poses'] = {}
    output['pred_poses']['outputs_coord'] = init_reference.to(device)

    if calculate_2d:
        ref_points_expand = self.project_ref_points(reference_points,
            meta[n], 
            nviews_this,
            batch_size,
            nbins,
            device,
            nfeat_level,
            src_spatial_shapes)
        
        output['pred_poses_2d']['outputs_coord_2d']

    return output

class DyanmicQueryTransformer(MultiviewPosetransformer):
    """
    Multi-view Pose Transformer Module with Dynamic Queries
    Args:
        cfg: the config file
    """
    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        
        decoder_layer = DQDecoderLayer(cfg.MULTI_PERSON.SPACE_SIZE,
                                        cfg.MULTI_PERSON.SPACE_CENTER,
                                        cfg.NETWORK.IMAGE_SIZE,
                                        cfg.DECODER.pose_embed_layer,
                                        cfg.DECODER.d_model,
                                        cfg.DECODER.dim_feedforward,
                                        cfg.DECODER.dropout,
                                        cfg.DECODER.activation,
                                        cfg.DECODER.num_feature_levels,
                                        cfg.DECODER.nhead,
                                        cfg.DECODER.dec_n_points,
                                        cfg.DECODER.
                                        detach_refpoints_cameraprj_firstlayer,
                                        cfg.DECODER.fuse_view_feats,
                                        cfg.DATASET.CAMERA_NUM,
                                        cfg.DECODER.projattn_posembed_mode,
                                        cfg.DECODER.feature_update_method,
                                        cfg.DECODER.init_self_attention,
                                        cfg.DECODER.open_forward_ffn,
                                        cfg.DECODER.query_filter_method,
                                        visualization_jump_num=cfg.DEBUG.VISUALIZATION_JUMP_NUM,
                                        bayesian_update=cfg.DECODER.bayesian_update,
                                        triangulation_method=cfg.DECODER.triangulation_method,
                                        filter_query=cfg.DECODER.filter_query
                                        )
        self.decoder = DQDecoder(cfg, decoder_layer,
                                  cfg.DECODER.num_decoder_layers,
                                  cfg.DECODER.return_intermediate_dec)

        # an MLP to estimate 2d offsets
        self.pose_embed = MLP(cfg.DECODER.d_model, cfg.DECODER.d_model,
                              2, cfg.DECODER.pose_embed_layer)
        nn.init.constant_(self.pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.pose_embed.layers[-1].bias.data, 0)

        num_pred = self.decoder.num_layers
        if cfg.DECODER.with_pose_refine: ## config true
            self.pose_embed = _get_clones(self.pose_embed, num_pred)
            self.decoder.pose_embed = self.pose_embed
        else:
            nn.init.constant_(self.pose_embed.layers[-1].bias.data[2:], -2.0)
            self.pose_embed = nn.ModuleList([self.pose_embed
                                             for _ in range(num_pred)])
            self.decoder.pose_embed = None

        # Load init T-pose
        # t_pose_dir = './t_pose.pt'
        directory = os.getcwd()
        print('current dir:', directory)

        t_pose_dir = cfg.DECODER.t_pose_dir
        print('load t_pose from:', t_pose_dir)
        self.t_pose_origin = torch.load(t_pose_dir)
        # print('T pose size:', self.t_pose_origin.shape)

        self._reset_parameters()

        # for getting association indices
        self.matcher = self.criterion.matcher

        self.loss_for_each_layers = True

        # init for decoder
        self.decoder.class_embed = self.class_embed
        self.decoder.num_instance = self.num_instance
        self.decoder.num_joints = self.num_joints

        self.log_val_loss = cfg.DEBUG.LOG_VAL_LOSS

        self.visualization_jump_num = cfg.DEBUG.VISUALIZATION_JUMP_NUM

        self.init_ref_method = cfg.DECODER.init_ref_method
        self.init_ref_method_value = cfg.DECODER.init_ref_method_value

        if self.init_ref_method == 'query_adapt' or self.init_ref_method == 'query_adapt_center':
            # init sample network
            self.reference_feats = nn.Linear(
                cfg.DECODER.d_model * len(cfg.DECODER.use_feat_level)
                * cfg.DATASET.CAMERA_NUM,
                cfg.DECODER.d_model)  # 256*feat_level*num_views
            # self.reference_points = nn.Linear(cfg.DECODER.d_model, 3) # Initialized in multi_view_pose_transformer.py, for the default param reset function.

        self.gt_match = cfg.DECODER.gt_match

        self.close_pose_embedding = cfg.DECODER.close_pose_embedding
        
        self.decay_method = cfg.DECODER.decay_method
        
        self.gt_match_test = cfg.DECODER.gt_match_test
        
        self.match_method = cfg.DECODER.match_method
        
        self.open_loss_init = (cfg.DECODER.loss_weight_init > 0)

    def generate_T_pose(self, roots):
        '''
        roots: (query_num, 3)
        '''
        # for each root we add a normal trans to generate T-pose
        # 15 joints
        T_pose_joints_origin = self.t_pose_origin.to(roots.device)
        joint_num = T_pose_joints_origin.shape[0]
        roots = roots.unsqueeze(1).repeat(1,joint_num,1)
        person_joints = roots + T_pose_joints_origin
        return person_joints

    def generate_T_pose_batch(self, roots):
        '''
        roots: (batch, query_num, 3)
        '''
        # for each root we add a normal trans to generate T-pose
        # 15 joints
        T_pose_joints_origin = self.t_pose_origin.to(roots.device)
        joint_num = T_pose_joints_origin.shape[0]
        roots = roots.unsqueeze(2).repeat(1,1,joint_num,1)
        person_joints = roots + T_pose_joints_origin
        return person_joints

    def initialize_reference_points(self, tgt, meta, method="gt_noise", value=None):
        """Averagely sample 3D queries coordinates to cover all the space as coarse initialization"""

        batch_size = meta[0]['num_person'].shape[0]
        device = meta[0]['num_person'].device

        # Groundtruth information is used by gt_noise methods only, which samples poses by adding noise to the 
        # groundtruth poses.
        gt_ref_points = meta[0]['joints_3d']
        # pred_result from voxelpose 
        vp_ref_points = meta[0]['joints_3d_voxelpose_pred']
        vp_ref_vis = vp_ref_points[...,3]
        vp_ref_score = vp_ref_points[...,4]
        vp_ref_points = vp_ref_points[...,:3]
        # batchsize, query_num, joints_num, 3
        shape = gt_ref_points.shape
        joint_num = shape[2]
        gt_human_num = shape[1]
        query_num = tgt.shape[1] // joint_num
        device = gt_ref_points.device

        ## Debug method: queries are generated by adding noises to groundtruth pose
        #! have not check for batching
        if method == "gt_noise":
            if value is not None and value >= 0:
                noise_std = float(value)  
            else:
                noise_std = 100.0

            # for each data in each epoch, the noise is different
            # random_seed = gt_ref_points.reshape(-1)[0]
            # torch.manual_seed(random_seed) # each time use this, it refreshes
            rand_noise = torch.normal(torch.zeros(shape),noise_std*torch.ones(shape)).to(device)
            noise_ref_points = gt_ref_points + rand_noise

            # if query num is larger than max_gt_human num, padding zeros
            if query_num >= gt_human_num:
                shape_padding = list(shape)
                shape_padding[1] = query_num - gt_human_num
                padding_zeros = torch.zeros(shape_padding).to(device)
                noise_ref_points = torch.cat([noise_ref_points, padding_zeros], dim=1)
            else:
                # not implemented when query_num smaller than gt_num
                raise(ValueError, "query_num should larger than max people num.")

            noise_ref_points = noise_ref_points.view(batch_size, -1, 3)
            noise_ref_points = noise_ref_points.float()

        elif method == "sample_space":
            # generate center from 0-1, for the whole 3d space
            # Assume the shape is  N^3, first get N
            N = math.ceil(pow(query_num, 1/2.0))
            x_ = torch.linspace(0., 1., N).to(device)
            
            # we do not need to sample around z
            z_ = torch.zeros(N,N).to(device) + 0.5

            x, y = torch.meshgrid(x_, x_)      
            root_coordinates = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z_.unsqueeze(-1)], dim=-1)
            root_coordinates = root_coordinates.view(-1, 3)
            # ref_points_root_norm = zip(x,y,z)
            # ref_points_root_norm = torch.Tensor(ref_points_root_norm, device=device)

            # Only get the first QUERY_NUM coordinates.
            root_coordinates = root_coordinates[:query_num, ...]

            # norm2absolute
            ref_points_root_abs = self.norm2absolute(root_coordinates)

            # generate T-pose from root (no rotation, so only add offsets)
            ref_points_abs = self.generate_T_pose(ref_points_root_abs)

            noise_ref_points = ref_points_abs.expand(batch_size,-1,-1,-1).view(batch_size, -1, 3)
            noise_ref_points = noise_ref_points.float()
        
        elif method == "voxcel_pose_base":
            noise_ref_points = vp_ref_points.view(batch_size, -1, 3)
            noise_ref_points = noise_ref_points.float()

        ## output: batch_size, query_num * 15, 3
        return noise_ref_points

    def forward(self, views=None, meta=None, output_dir='./', frame_id=None, threshold=0.5):
        '''
        Forward functions for Transformer Network

        Input:
            @ views: a list of images with shape [batch_size, 3, H, W]
            @ meta: a list of meta information for each image; store camera info, e.g., R, T, Intri
            @ output_dir: output directory for visualization
            @ frame_id: frame id for visualization
            @ threshold: threshold for filtering queries
        '''

        time_0 = time.time()

        # 1. Extract image features through backbone
        if views is not None:
            ## use_feat_level: currently we only use one level;
            all_feats = self.backbone(torch.cat(views, dim=0),
                                      self.use_feat_level)
            all_feats = all_feats[::-1] # reverse
        
        time_1 = time.time()
        time_backbone.update(time_1-time_0)
        batch, _, imageh, imagew = views[0].shape
        nview = len(views)

        src_flatten_views = [] 
        mask_flatten_views = [] 
        spatial_shapes_views = [] # store feature map size for each feature level

        for lvl, src in enumerate(all_feats):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            mask = src.new_zeros(bs, h, w).bool()  # an empty mask
            mask_flatten_views.append(mask)
            mask = mask.flatten(1) # get vector feature for each batch
            src_flatten_views.append(src)

        spatial_shapes_views = \
            torch.as_tensor(spatial_shapes_views,
                            dtype=torch.long,
                            device=mask.device)
        level_start_index_views = \
            torch.cat((mask.new_zeros((1, ), dtype=torch.long),
                       torch.as_tensor(spatial_shapes_views,
                                       dtype=torch.long,
                                       device=mask.device)
                       .prod(1).cumsum(0)[:-1]))
        valid_ratios_views = torch.stack([self.get_valid_ratio(m)
                                          for m in mask_flatten_views], 1)
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]

        num_instance = self.num_instance
        num_joints = self.num_joints

        # 2. Get queries and pose embeddings according to the settings
        #   Position embbedings, and query features are split from the self.query_embed.weight
        #   channels: 512 = 256x2
        if self.query_embed_type == 'person_joint':  ## used by baseline
            # person embedding + joint embedding
            joint_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (joint_embeds + instance_embeds).flatten(0, 1) ## flatten: start at 0, end at 1
        elif self.query_embed_type == 'per_joint':
            # per joint embedding
            query_embeds = self.query_embed.weight
        elif self.query_embed_type == 'human_joint':
            # human shares the same joints
            joint_embeds = self.joint_embedding.weight
            instance_embeds = self.instance_embedding.weight
            query_embeds = (joint_embeds + instance_embeds).unsqueeze(0) ## flatten: start at 0, end at 1
            query_embeds = query_embeds.expand(num_instance, -1, -1) # copy N times
            query_embeds = query_embeds.flatten(0, 1)
        elif self.query_embed_type == 'human_joint2':
            joint_embeds = self.joint_embedding.weight
            instance_embeds = self.instance_embedding.weight
            query_embeds = (joint_embeds + instance_embeds).unsqueeze(0) ## flatten: start at 0, end at 1
            query_embeds = query_embeds.expand(num_instance, num_joints, -1) # copy N times
            query_embeds = query_embeds.flatten(0, 1)
        elif self.query_embed_type == 'per_instance':
            joint_embeds = self.joint_embedding.weight
            instance_embeds = self.instance_embedding.weight
            query_embeds = (joint_embeds + instance_embeds).unsqueeze(1) ## flatten: start at 0, end at 1
            query_embeds = query_embeds.expand(-1, num_joints, -1) # copy N times
            query_embeds = query_embeds.flatten(0, 1)
            
        ## c: query feature channels 256
        query_embed, tgt = torch.split(query_embeds, c, dim=1)  # query_embeds: 960x512, c 256 -> 960x256 each

        ## query and position embeddings are the same for each batch
        ## unsqueeze & expand: copy multiple times
        init_method = self.init_ref_method
        if self.close_pose_embedding:
            query_embed = None
        else:
            query_embed = query_embed.unsqueeze(0).expand(batch, -1, -1)
        tgt = tgt.unsqueeze(0).expand(batch, -1, -1)

        time_2 = time.time()
        time_preprocess.update(time_2 - time_1)

        # 3. Initialize reference points, the initialized 3D query poses
        # query adaptation method uses neural network to predict initialized poses;
        # otherwise, we initialize queries by averagely sampling in the space.
        if init_method == 'query_adapt' or init_method == 'query_adapt_center':
            assert(len(all_feats) == 3 and "Please use all 3 feature layers.")
            feats_0 = F.adaptive_avg_pool2d(all_feats[0], (1, 1))  # 40,256,128,240 -> 40,256,1,1
            feats_1 = F.adaptive_avg_pool2d(all_feats[1], (1, 1))  # 40,256,64,120 -> 40,256,1,1
            feats_2 = F.adaptive_avg_pool2d(all_feats[2], (1, 1))  # 40,256,32,60  -> 40,256,1,1
            feats = torch.cat((feats_0, feats_1, feats_2),
                              dim=1).squeeze().view(nview, batch, 3, c)
            # put batch dim to the 1st
            feats_batch = feats.transpose(0,1).reshape(batch, -1)
            ref_feats = self.reference_feats(feats_batch).unsqueeze(1)

            # if close embedding, use query; or, use embedding to init
            if init_method == 'query_adapt':
                if query_embed is None:
                    reference_points = self.reference_points(
                        tgt + ref_feats)
                else:
                    reference_points = self.reference_points(
                        query_embed + ref_feats)
            elif init_method == 'query_adapt_center':
                '''
                Only regress center points. And add T-Pose to generate full init pose.
                '''
                # Average the dim of joints (15), and only get one 3D center point
                if query_embed is None:
                    tgt_ins = tgt.view(batch, num_instance, num_joints, c).mean(dim=2)
                    reference_points_center = self.reference_points(
                        tgt_ins + ref_feats)
                else:
                    query_embed_ins = query_embed.view(batch, num_instance, num_joints, c).mean(dim=2)
                    reference_points_center = self.reference_points(
                        query_embed_ins + ref_feats)  
                # From center to full points with T-pose
                reference_points = self.generate_T_pose_batch(reference_points_center)
                reference_points = reference_points.view(batch, -1, 3).float()
                
        else:
            # Initialize queries poses by averagely sampling in the space
            reference_points = self.initialize_reference_points(tgt, meta, method=self.init_ref_method, value=self.init_ref_method_value)
        
        reference_points_origin = reference_points

        ## Temp saving for debug
        # temp_count_batch = 1
        # final_output_dir = './debug/{}'.format(temp_count_batch)
        # os.makedirs(final_output_dir, exist_ok=True)
        # prefix_joint = os.path.join(final_output_dir, 'joints_init')
        # ## visualize batch 0, layer 0,
        # save_ref_points_with_gt(reference_points[0], meta[0], prefix_joint, color='b', transparency=0.1)

        ## [filter_valid_queries]: Not used. Check initial ref points with groundtruth (meta) to find valid ones.
        ## So we can ignore the invalid queries during training to save time
        ## Attention: when in inference mode, all of the queries have to go through decoder
        # filter_valid_queries = False

        # Associate the initialized queries with groundtruth,
        # and keep the valid ones in reference_points_filtered
        if 'joints_3d' in meta[0] \
            and 'joints_3d_vis' in meta[0]:
            meta[0]['roots_3d_norm'] = \
                self.absolute2norm(meta[0]['roots_3d'].float())
            meta[0]['joints_3d_norm'] = \
                self.absolute2norm(meta[0]['joints_3d'].float())
            origin = construct_output_from_origin(reference_points, device=mask.device)
            indices = self.matcher(origin, meta) 
            indices_est = [item[0] for item in indices]  # batch,(est id, gt id) #! padding to max len query
            
            # Add a reorder process to make indices align the gt meta.
            # This is gauranteed for Structural Triangulation Bone Length; Not Used. 
            open_reorder = False
            if open_reorder:
                for i,indice in enumerate(indices_est):
                    order = indices[i][1]
                    order_ind = torch.sort(order)[1]
                    
                    # change ind order w.r.t order
                    ind_arranged = indice[order_ind]
                    indices_est[i] = ind_arranged

            # add a flag gt_match to open or close for training
            # new case: if open gt_match_test, we still use gt match for testing.
            if self.gt_match and (self.training or self.gt_match_test):
                indices_train =[item.cuda() for item in indices_est] 
                indices_all = indices
            else:
                indices_train = None
                indices_all = None
            
            # update 
            reference_points_filtered = [reference_point.view(-1,15,3)[indices_est[index],...].view(-1,3) for index,reference_point in enumerate(reference_points)] 

            # Debug Code: Only keep the valid queries
            #! 'filter_valid_queries = True' has not checked for batching
            # if filter_valid_queries and self.training:
            #     reference_points = reference_points_filtered
            #     vec_dim = query_embed.shape[-1]
            #     query_embed = query_embed.view(batch,-1,15,vec_dim)[:,indices_est,...].view(batch,-1,vec_dim)
            #     tgt = tgt.view(batch,-1,15,vec_dim)[:,indices_est,...].view(batch,-1,vec_dim)

            #     num_instance = len(indices_est)

        init_reference = reference_points  # Batch x 150 x 3, 150: instance * keypoints
        
        time_3 = time.time()
        time_init_ref.update(time_3-time_2)

        # 4. Go through the decoder
        ## See DQDecoder, DQDecoderLayer in dq_decoder.py
        ## The decoder contains multiple decoder layers for query pose and feature update;
        ## Each layer will project the coarse query pose to images, extract features with attention, 
        ## and estimate offsets for triangulation;
        ## Then triangulation recovers an updated 3D query pose;
        ## Query features are also updated by self-attention and image features.

        # begin_decoder = time_synchronized()
        hs, inter_references, inter_references_2d, inter_references_2d_projs, outputs_classes = \
            self.decoder(tgt, reference_points, src_flatten_views,
                         meta=meta, src_spatial_shapes=spatial_shapes_views,
                         src_level_start_index=level_start_index_views,
                         src_valid_ratios=valid_ratios_views,
                         query_pos=query_embed,
                         src_padding_mask=mask_flatten_views,
                         rgb_views=views,
                         output_dir=output_dir,
                         frame_id=frame_id,
                         indices = indices_train,
                         threshold = threshold,
                         indices_all = indices_all)  # for vis only
        # print('decoder layer time:', time_synchronized()-begin_decoder)

        time_4 = time.time()
        time_decoder_layers.update(time_4-time_3)

        # change class to logits
        output_logits = [inverse_sigmoid(out) for out in outputs_classes]

        ## store class, poses, and calculate loss for each layers.
        outputs_coords = []
        outputs_coords_2d = []
        outputs_coords_2d_proj = []
        num_layers = hs.shape[0]
        for lvl in range(num_layers):
            outputs_coord = inter_references[lvl]

            # when using other datasets: convert panoptic joints to shelf/campus
            if self.convert_joint_format_indices is not None:
                outputs_coord = \
                    outputs_coord.view(batch,
                                       num_instance,
                                       self.num_joints, -1)
                outputs_coord \
                    = outputs_coord[..., self.convert_joint_format_indices, :]
                outputs_coord = outputs_coord.flatten(1, 2)

            outputs_coords.append({'outputs_coord': outputs_coord})
            outputs_coord_2d = inter_references_2d[lvl]
            outputs_coord_2d_proj = inter_references_2d_projs[lvl]
            outputs_coords_2d.append({'outputs_coord_2d': outputs_coord_2d})
            outputs_coords_2d_proj.append({'outputs_coord_2d_proj': outputs_coord_2d_proj})

        # Final layer output
        out = {'pred_logits': output_logits[-1],
               'pred_poses': outputs_coords[-1],
               'pred_poses_2d': outputs_coords_2d[-1],
               'pred_poses_2d_proj': outputs_coords_2d_proj[-1]}

        # Count how many valid observations in each layers
        # count_valid_queries(outputs_coords, num_joints)

        if self.aux_loss:
            out['aux_outputs'] = \
                self._set_aux_loss(output_logits, outputs_coords)

        # Visualization
        config_jump_save_image = self.visualization_jump_num
        open_visualization = (config_jump_save_image >= 0)
        if open_visualization:
            if frame_id % config_jump_save_image == 0 and frame_id > 0:
                final_output_dir = '{}/{}'.format(output_dir,frame_id)
                os.makedirs(final_output_dir, exist_ok=True)
                prefix_joint = os.path.join(final_output_dir, 'joints_init')

                save_ref_points_with_gt(reference_points_origin, meta[0], prefix_joint, color='b', transparency=0.3)

                prefix_joint = os.path.join(final_output_dir, 'joints_init_filtered')
                save_ref_points_with_gt(reference_points_filtered, meta[0], prefix_joint, color='b', transparency=0.3)

                prefix2 = os.path.join(final_output_dir, 'features')
                for batch_index in range(batch): 
                    save_multiple_images(hs[:,batch_index:batch_index+1], os.path.join(prefix2, f'batch_{batch_index}'))

                for layer_id in range(num_layers):
                    ## store inter_references with ground truth
                    prefix2 = os.path.join(final_output_dir, f'joints_l{layer_id}')
                    ## visualize batch 0, layer id,
                    save_ref_points_with_gt(inter_references[layer_id], meta[0], prefix2)

                    prefix2 = os.path.join(final_output_dir, f'joints_l{layer_id}_filtered')
                    ## visualize batch 0, layer 0,
                    # indices will not change in different layers.
                    save_ref_points_with_gt(inter_references[layer_id], meta[0], prefix2, indices=indices)

                    ## save with limited range
                    # prefix2 = os.path.join(final_output_dir, f'joints_range_l{layer_id}')
                    # range_in = np.array(
                    #     [[-4000,4000],
                    #     [-4000,4000],
                    #     [-1000,2500]]
                    # )
                    # ## visualize batch 0, layer 0,
                    # save_ref_points_with_gt(inter_references[layer_id][0], meta[0], prefix2, range_in=range_in, color='b', transparency=0.1)

                print('save for batch:', final_output_dir)

        # Construct losses for training
        # Update: validation also outputs loss
        log_val_loss = self.log_val_loss
        if (self.training or log_val_loss) and 'joints_3d' in meta[0] \
                and 'joints_3d_vis' in meta[0]:
            meta[0]['roots_3d_norm'] = \
                self.absolute2norm(meta[0]['roots_3d'].float())
            meta[0]['joints_3d_norm'] = \
                self.absolute2norm(meta[0]['joints_3d'].float())
            
            # use initialized pose for data association
            if self.gt_match:
                origin = construct_output_from_origin(init_reference, device=mask.device)
            else:
                origin = None
            
            # if open each layer loss, calculate and sum them for each layers!
            if self.loss_for_each_layers:
                loss_dict_lids = []
                num_layers = self.decoder.num_layers
                for lid in range(num_layers):
                    out_lid = {'pred_logits': output_logits[lid],
                        'pred_poses': outputs_coords[lid],
                        'pred_poses_2d': outputs_coords_2d[lid],
                        'pred_poses_2d_proj': outputs_coords_2d_proj[lid]}
                    loss_dict_lid, indices = self.criterion(out_lid, meta, origin)
                    loss_dict_lids.append(loss_dict_lid)
                # sum all the loss in dict
                keys = loss_dict_lids[0].keys()
                # go over all the keys, and sum their values
                loss_dict_sum = {}
                for key in keys:
                    key_value_list = [loss_dict[key] for loss_dict in loss_dict_lids]
                    # sum_all_key_value = torch.Tensor(key_value_list).sum()
                    mean_name_list = ['class_error', 'class_recall', 'class_precision', 'cardinality_error']
                    if key in mean_name_list:
                        sum_all_key_value = torch.cat([a.view(-1,) for a in key_value_list]).mean()
                    else:
                        # If open decay summing, change here
                        num_layers = len(key_value_list)
                        if self.decay_method == 'none':
                            weights = torch.ones((num_layers,))
                        elif self.decay_method == 'linear':
                            # weights = torch.Tensor([0.25, 0.5, 0.75, 1.0])
                            weights = torch.linspace(0,1,num_layers+1)[1:]
                        elif self.decay_method == 'exp':
                            # weights = torch.Tensor([0.125, 0.25, 0.5, 1.0])
                            weights_ = 2**torch.arange(1,num_layers+1)
                            weights = weights_ / weights_[-1]
                        elif self.decay_method == 'last':
                            # only last layer has loss
                            weights = torch.zeros((num_layers,))
                            weights[-1] = 1
                        weights = weights.to(mask.device)
                        tensor_all_key_value = torch.cat([a.view(-1,) for a in key_value_list])
                        sum_all_key_value = (weights * tensor_all_key_value).sum()
                    loss_dict_sum[key] = sum_all_key_value

                loss_dict = loss_dict_sum

                # store loss for each layers into loss_dict, and output for visualization
                loss_dict['dict_losses_layers'] = loss_dict_lids
            else:
                loss_dict, indices = self.criterion(out, meta, origin)

            # Update: if open init loss, add init loss to loss_dict
            loss_dict['loss_init'] = torch.Tensor([0.0]).cuda()
            if not self.gt_match:
                # Open DETR Loss
                if self.open_loss_init:
                    # and add loss for initialization
                    origin_output = construct_output_from_origin(init_reference, device=mask.device)
                    indices_origin_output = self.criterion.matcher(origin_output, meta)
                    loss_init_criterion = self.criterion.loss_poses(outputs=origin_output['pred_poses']['outputs_coord'], outputs_2d=None, \
                        meta=meta, indices=indices_origin_output, num_samples=None, output_abs_coord=True)  # use pose_loss only and ignore classification
                    loss_dict['loss_init'] = loss_init_criterion['loss_pose_perjoint']
                    
            return out, loss_dict

        # Save time and output for every N times
        time_5 = time.time()
        time_final_process.update(time_5-time_4)
        
        # N_time_output = 100
        N_time_output = 10000
        if time_backbone.count % N_time_output == 0:
            print('time_backbone:', time_backbone.avg)
            print('time_preprocess:', time_preprocess.avg)
            print('time_init_ref:', time_init_ref.avg)
            print('time_decoder_layers:', time_decoder_layers.avg)
            print('time_final_process:', time_final_process.avg)
        
        
        # time_backbone = AverageMeter()
        # time_preprocess = AverageMeter()
        # time_init_ref = AverageMeter()
        # time_decoder_layers = AverageMeter()
        # time_final_process = AverageMeter()

        
        return out

def get_mvp(cfg, is_train=True, fix_backbone=True):
    if cfg.BACKBONE_MODEL: ## used
        backbone = eval(
            cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None

    if fix_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print(' * Fix backbone: True')
    else:
        print(' * Fix backbone: False')

    model = DyanmicQueryTransformer(backbone, cfg)
    return model

def count_valid_queries(outputs_coords, num_joints):
    for lid, data in enumerate(outputs_coords):
        coords = data['outputs_coord']
        batch = coords.shape[0]
        coords = coords.view(batch,
                                       -1,
                                       num_joints, 3)
        # batch, queries, joints, 3
        # count in the query dims
        coords_check_zero = coords.flatten(2,3).sum(-1)
        valid_queries_batch = (coords_check_zero.abs() > 0).sum(-1)

        # print(f'layer-{lid}:', valid_queries_batch.sum().item(), "Each:", valid_queries_batch.cpu().numpy())
