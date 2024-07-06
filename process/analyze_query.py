# ------------------------------------------------------------------------------
# Copyright
#
# This file is part of the repository of the CVPR'24 paper:
# "Multiple View Geometry Transformers for 3D Human Pose Estimation"
# https://github.com/XunshanMan/MVGFormer
#
# Please follow the LICENSE detail in the main repository.
# ------------------------------------------------------------------------------

import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

import pickle

query_score = torch.load('query_score.pt')
print('shape:', query_score.shape)

# plot to a hist
counts, bins = np.histogram(query_score)
plt.hist(bins[:-1], bins, weights=counts)
plt.savefig('./query_score.png')

# eval list
eval_list = pickle.load(open('/home/X/eval_list', 'rb'))
# get the score, mpjpe
outlier_thresh = 500
score_list = [eval['score'] for eval in eval_list if eval['mpjpe'] < outlier_thresh]
mpjpe_list = [eval['mpjpe'] for eval in eval_list if eval['mpjpe'] < outlier_thresh]
plt.figure()
plt.title(f'Score v.s. MPJPE of Estimated Poses on all datasets (mpjpe < {outlier_thresh})')
plt.scatter(score_list, mpjpe_list)
plt.xlabel('score')
plt.ylabel('mpjpe')
plt.savefig(f'./score_mpjpe_th{outlier_thresh}.png')

# Only one frame
gt_id_list=[0,1,2,3]
score_list = [eval['score'] for eval in eval_list if eval['gt_id'] in gt_id_list and eval['mpjpe'] < outlier_thresh]
mpjpe_list = [eval['mpjpe'] for eval in eval_list if eval['gt_id'] in gt_id_list and eval['mpjpe'] < outlier_thresh]
gt_id_for_color_list = [eval['gt_id'] for eval in eval_list if eval['gt_id'] in gt_id_list and eval['mpjpe'] < outlier_thresh]
plt.figure()
plt.title(f'Frame 1 with 4 Persons (mpjpe < {outlier_thresh})')
plt.scatter(score_list, mpjpe_list, c=gt_id_for_color_list)
plt.xlabel('score')
plt.ylabel('mpjpe')
plt.savefig(f'./score_mpjpe_frame1_th{outlier_thresh}.png')
