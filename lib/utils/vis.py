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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pickle
from mpl_toolkits.mplot3d import Axes3D

import torch

# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
           [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# # h36m
# LIMBS17 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
#          [8, 9], [9, 10], [8, 14], [14, 15], [15, 16],
#          [8, 11], [11, 12], [12, 13]]
# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5],
           [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13],
           [13, 15], [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
           [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

#####################################

def normalize_image(im, to_rgb=False):
    image = im.clone()
    min = float(image.min())
    max = float(image.max())

    image.add_(-min).div_(max - min + 1e-5)
    image=image*255.0

    # change channel rgb
    if len(image.shape) == 4:
        image = image.squeeze()
    assert( len(image.shape) == 2 or len(image.shape) == 3 )
    im = image.permute(1,2,0).detach().cpu().numpy()

    if to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def visualize_proj_attention(
            nviews,
            rgb_views,
            src_views,
            attn_feature_views,
            projs_2d_abs,
            offsets_2d_abs,
            refined_2d_poses_abs,
            save_dir='./',
            config_vis_proj = True,
            config_vis_offset = True,
            config_vis_refined = True,
            draw_line = True,
            config_vis_human = True
        ):
    os.makedirs(save_dir, exist_ok=True)
    colors = [
        (.7, .3, .3, 1.),       # red
        (.7, .5, .3, 1.),       # Yellow
        (.5, .55, .3, 1.),      # green
        (.3, .3, .7, 1.),       # dark purple
        (.3, .5, .55, 1.),      # light blue
        (.5, .5, .7, 1.),       # light purple
    ]

    # visualize view by view
    for view_id in range(nviews):
        save_dir_cur = save_dir + "/v{}.png".format(view_id)
        rgb = rgb_views[view_id]
        rgb_vis = normalize_image(rgb, to_rgb=True)
        # rgb_vis = cv2.circle(rgb_vis, (int(960/2),int(512/2)), 6, (255,0,0), 2)

        # expand image to a larger pixels.
        # rgb_vis = cv2.copyMakeBorder(rgb_vis, 0, 2500, 0, 2500, cv2.BORDER_CONSTANT, None, value = 0)
                
        if config_vis_proj:
            joint = projs_2d_abs[view_id]  # 1,150,2
            joint = joint.view(-1,15,2) # people num

            if offsets_2d_abs is not None:
                offset = offsets_2d_abs[view_id].view(-1,15,2)
            else:
                config_vis_offset = False
            num_people = joint.shape[0]
            for n in range(num_people):
                color = np.array(colors[n%6])[:3]*255
                joint_people = joint[n]
                for joint_id in range(15):
                    p = joint_people[joint_id]
                    if (p[0] < rgb.shape[-2:][1] and 
                        p[1] < rgb.shape[-2:][0] and 
                        p[0] > 0 and
                        p[1] > 0):
                        pt = (int(p[0]), int(p[1]))
                        rgb_vis = cv2.circle(rgb_vis, pt, 2,
                                        color, 2)

                        if config_vis_offset:
                            off = offset[n][joint_id]
                            dist = p + off
                            dist_pt = (int(dist[0]), int(dist[1]))
                            rgb_vis = cv2.line(rgb_vis, pt, dist_pt, (0,255,0), 2)
                            # rgb_vis = cv2.circle(rgb_vis, (int(dist[0]), int(dist[1])), 2,
                                        # color, 2)

                # visualize a whole people
                if config_vis_human:
                    jt = joint_people
                    for k in eval("LIMBS{}".format(len(jt))):
                        x1 = jt[k[0]].int().tolist()
                        x2 = jt[k[1]].int().tolist()
                        # x1 = (jt[k[0]][0].item(), jt[k[0]][1].item())
                        # x2 = (jt[k[1]][0].item(), jt[k[1]][1].item())
                        rgb_vis = cv2.line(rgb_vis, x1, x2, color=(128,128,128))

                ## Only for the first people!
                # if n >= 3:
                    # break 

        # render proj, offsets, refined points
        if config_vis_refined:
            joint = refined_2d_poses_abs[view_id]  # 1,150,2
            joint = joint.view(-1,15,2) # people num
            num_people = joint.shape[0]
            for n in range(num_people):
                color = np.array(colors[n%6])[:3]*255
                joint_people = joint[n]
                for joint_id in range(15):
                    p = joint_people[joint_id]
                    if (p[0] < rgb.shape[-2:][1] and 
                        p[1] < rgb.shape[-2:][0] and 
                        p[0] > 0 and
                        p[1] > 0):
                        pt = (int(p[0]), int(p[1]))
                        rgb_vis = cv2.circle(rgb_vis, pt, 2,
                                            (0,255,0), 2)
                        if draw_line:
                            p_origin = projs_2d_abs[view_id].view(-1,15,2)[n][joint_id]
                            pt_origin = (int(p_origin[0]), int(p_origin[1]))
                            rgb_vis = cv2.line(rgb_vis, pt, pt_origin, (0,0,255), 1)

                if config_vis_human:
                    jt = joint_people
                    for k in eval("LIMBS{}".format(len(jt))):
                        x1 = jt[k[0]].int().tolist()
                        x2 = jt[k[1]].int().tolist()
                        rgb_vis = cv2.line(rgb_vis, x1, x2, color=color, thickness=3)
                        
        # save pic
        cv2.imwrite(save_dir_cur, rgb_vis)

    print('finish saving', save_dir)
    return

# color: secify color for predictions (often used for initialization)
#  * 'b', 'g', 'c', 'y', 'm', 'orange',
                #   'pink', 'royalblue', 'lightgreen', 'gold'
# transparency: (0.,1.)
# range: 3x2 xlim, ylim, zlim
# indices: batch_size, (est, gt)
def save_ref_points_with_gt(preds, meta, file_name, color=None, transparency=1, range_in=None,
    indices=None):
    batch_size = len(preds)
    joints_num = 15

    if isinstance(preds, torch.Tensor):
        preds = preds.view(batch_size,-1,joints_num,3)
    else:
        preds = [pred.view(-1,joints_num,3) for pred in preds]

    if indices is not None:
        # select some !  # indices: batchsize, preds_ids, gt_ids
        valid_pred_ids = [ind[0].to(preds.device) for ind in indices]
        preds_valid = []
        for i in range(batch_size):
            pred_i = preds[i][valid_pred_ids[i]]
            preds_valid.append(pred_i)
        preds = preds_valid

    # batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        for n in range(num_person):
            ## plot groundtruth
            joint = joints_3d[n]
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', lw=2, marker='o',
                            markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=2,
                            marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange',
                  'pink', 'royalblue', 'lightgreen', 'gold']
        if color is not None:
            for ic in range(len(colors)):
                colors[ic] = color
        if preds is not None:
            pred = preds[i]
            ###################
            # pred = preds
            for n in range(len(pred)):
                joint = pred[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c=colors[int(n % 10)],
                            lw=1.5, marker='o',
                            markerfacecolor='w', markersize=2,
                            markeredgewidth=1,
                            alpha=transparency)
    
    if range_in is not None:
        ax.set_xlim(range_in[0,:])
        ax.set_ylim(range_in[1,:])
        ax.set_zlim(range_in[2,:])

    plt.savefig(file_name)
    plt.close(0)

def save_3d_images_direct(config, meta, preds, file_name):
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)

    # num_person = meta['num_person'][i]
    # joints_3d = meta['joints_3d'][i]
    # joints_3d_vis = meta['joints_3d_vis'][i]
    i= 0
    ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
    colors = [
        (.7, .3, .3, 1.),       # red
        (.7, .5, .3, 1.),       # Yellow
        (.5, .55, .3, 1.),      # green
        (.3, .3, .7, 1.),       # dark purple
        (.3, .5, .55, 1.),      # light blue
        (.5, .5, .7, 1.),       # light purple
    ]

    if preds is not None:
        pred = preds
        col_idx = 0
        for n in range(len(pred)):
            joint = pred[n]
            for k in eval("LIMBS{}".format(len(joint))):
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                ax.plot(x, y, z, c=colors[col_idx % 6],
                        lw=2.5, marker='o',
                        markerfacecolor='w', markersize=3,
                        markeredgewidth=2)
            col_idx += 1

    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    fig.savefig(file_name, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(0)

def save_multiple_images(images, prefix, normalize=True):
    os.makedirs(prefix, exist_ok=True)

    for i,image in enumerate(images):
        if normalize:
            image = normalize_image(image)

        save_dir = prefix + "/{}.png".format(i)
        cv2.imwrite(save_dir, image)


def save_batch_image_with_joints_multi(batch_image,
                                       batch_joints,
                                       batch_joints_vis,
                                       num_person,
                                       file_name,
                                       nrow=8,
                                       padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_person, num_joints, 3],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    '''
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            for n in range(num_person[k]):
                joints = batch_joints[k, n]
                joints_vis = batch_joints_vis[k, n]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                                   [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps_multi(batch_image,
                              batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    batch_image = batch_image.flip(1)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images_multi(config, input, meta, target, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'image_with_joints')
    dirname2 = os.path.join(dirname, 'batch_heatmaps')

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints_multi(
            input, meta['joints'],
            meta['joints_vis'],
            meta['num_person'], '{}_gt.jpg'.format(prefix1))
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps_multi(
            input, target, '{}_hm_gt.jpg'.format(prefix2))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps_multi(
            input, output, '{}_hm_pred.jpg'.format(prefix2))


def save_3d_images(config, meta, preds, file_name):
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        # num_person = meta['num_person'][i]
        # joints_3d = meta['joints_3d'][i]
        # joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        colors = [
            (.7, .3, .3, 1.),       # red
            (.7, .5, .3, 1.),       # Yellow
            (.5, .55, .3, 1.),      # green
            (.3, .3, .7, 1.),       # dark purple
            (.3, .5, .55, 1.),      # light blue
            (.5, .5, .7, 1.),       # light purple
        ]

        if preds is not None:
            pred = preds[i]
            col_idx = 0
            for n in range(len(pred)):
                joint = pred[n]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[col_idx % 6],
                                lw=2.5, marker='o',
                                markerfacecolor='w', markersize=3,
                                markeredgewidth=2)
                    col_idx += 1

    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    fig.savefig(file_name, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(0)


def save_3d_images_novel_view(config, meta, preds, file_name):
    batch_size = meta['num_person'].shape[0]
    xplot = min(2, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        # num_person = meta['num_person'][i]
        # joints_3d = meta['joints_3d'][i]
        # joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        ax.view_init(elev=28., azim=210)
        colors = [
            (.7, .3, .3, 1.),       # red
            (.7, .5, .3, 1.),       # Yellow
            (.5, .55, .3, 1.),      # green
            (.3, .3, .7, 1.),       # dark purple
            (.3, .5, .55, 1.),      # light blue
            (.5, .5, .7, 1.),       # light purple
        ]
        if preds is not None:
            pred = preds[i]
            col_idx = 0
            for n in range(len(pred)):
                joint = pred[n]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[col_idx % 6], lw=2.5,
                                marker='o', markerfacecolor='w', markersize=3,
                                markeredgewidth=2)
                    col_idx += 1
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    fig.savefig(file_name, transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.close(0)


def save_debug_2d_images_seq(batch_image,gt,prefix,index=None):


    padding = 2
    nrow = 3
    prefix = f'{prefix}/2d_gt_{index}.jpg'
    batch_image = torch.stack(batch_image)
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= len(gt['joints_2d']):
                break
            joints = gt['joints_2d'][k]
            joints_vis = gt['joints_2d_vis'][k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                                [0, 255, 255], 2)
            k = k + 1
    cv2.imwrite(prefix, ndarr)


def save_debug_3d_images_seq(pd_list,gt_list,prefix,index=None):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    if index is not None:
        file_name = f"{prefix}_3d_{index}.png"
    else:
        file_name = f"{prefix}_3d.png"

    xplot = min(4, len(gt_list))
    yplot = int(math.ceil(float(len(gt_list)) / xplot))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)    
    for i,gt in enumerate(gt_list):
        joint = gt['joints_3d']
        joint_vis = gt['joints_3d_vis']
        ax = plt.subplot(yplot, xplot, i+1, projection='3d')
        for k in eval("LIMBS{}".format(len(joint))):
            if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                ax.plot(x, y, z, c='r', lw=1.5, marker='o',
                        markerfacecolor='w', markersize=2,
                        markeredgewidth=1)
            else:
                x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                ax.plot(x, y, z, c='r', ls='--', lw=1.5,
                        marker='o', markerfacecolor='w', markersize=2,
                        markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange',
                  'pink', 'royalblue', 'lightgreen', 'gold']

        for j,pred in enumerate(pd_list):
            if pred['gt_id'] != gt['gt_id']:
                continue
            joint = pred['joints_3d']
            if joint[0, 3] >= 0:
                for k in eval("LIMBS{}".format(len(joint))):
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c=colors[int(j % 10)],
                            lw=1.5, marker='o',
                            markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
    plt.savefig(file_name)
    plt.clf()
    plt.close(0)

def save_debug_3d_images(config, meta, preds, prefix, show_id=False):
    # if not config.DEBUG.DEBUG:
    #     return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    # width = 4.0 * xplot
    # height = 4.0 * yplot
    # fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        for n in range(num_person):
            joint = joints_3d[n]
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o',
                            markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5,
                            marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                if show_id:
                    # show ID text here
                    id = k[0]
                    px,py,pz = joint[k[0], 0], joint[k[0], 1], joint[k[0], 2]
                    ax.text(px, py, pz, str(id), color='red')
                    id = k[1]
                    px,py,pz = joint[id, 0], joint[id, 1], joint[id, 2]
                    ax.text(px, py, pz, str(id), color='red')
                    
        colors = ['b', 'g', 'c', 'y', 'm', 'orange',
                  'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n]
                if joint.shape[1]==3 or joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)],
                                lw=1.5, marker='o',
                                markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    plt.savefig(file_name)
    plt.close(0)


def save_debug_3d_cubes(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'root_cubes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]
    # root_id = config.DATASET.ROOTIDX

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    # width = 6.0 * xplot
    # height = 4.0 * yplot
    # fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        roots_gt = meta['roots_3d'][i]
        num_person = meta['num_person'][i]
        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')

        x = roots_gt[:num_person, 0].cpu()
        y = roots_gt[:num_person, 1].cpu()
        z = roots_gt[:num_person, 2].cpu()
        ax.scatter(x, y, z, c='r')

        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c='b')

        space_size = config.MULTI_PERSON.SPACE_SIZE
        space_center = config.MULTI_PERSON.SPACE_CENTER
        ax.set_xlim(space_center[0] - space_size[0] / 2,
                    space_center[0] + space_size[0] / 2)
        ax.set_ylim(space_center[1] - space_size[1] / 2,
                    space_center[1] + space_size[1] / 2)
        ax.set_zlim(space_center[2] - space_size[2] / 2,
                    space_center[2] + space_size[2] / 2)

    plt.savefig(file_name)
    plt.close(0)


def save_debug_epipolar(inputs, meta, targets_2d, out, prefix):
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'epipolar')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)
    outputs = {}
    for view, image in enumerate(inputs):
        outputs['view{}_img'.format(view)] = image.cpu().numpy()
        outputs['view{}_target_2d'.format(view)] \
            = targets_2d[view].cpu().numpy()
        outputs['view{}_joints_2d'.format(view)] \
            = meta[view]['joints'][:, :meta[view]['num_person']].cpu().numpy()
        outputs['view{}_joints_vis'.format(view)] \
            = meta[view][
                'joints_vis'][:, :meta[view]['num_person']].cpu().numpy()
    if 'epipolar_line_sampled_points' in outputs:
        outputs['epipolar_line_sampled_points'] \
            = out['epipolar_line_sampled_points'].cpu().numpy()
        outputs['epipolar_line_ref_points'] \
            = out['epipolar_line_ref_points'].cpu().numpy()
    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_epipolar.pkl"
    with open(file_name, 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
