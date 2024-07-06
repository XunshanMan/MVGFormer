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

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import pprint

import _init_paths

from core.config import config
from core.config import update_config, update_config_dynamic_input
from core.function import validate_3d
from core.nms import nearby_joints_nms
from utils.utils import create_logger
import lib.utils.misc as utils
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler
from models.util.misc import is_main_process, collect_results

import dataset
import models

import models.dq_transformer

import numpy as np
from datetime import datetime
from prettytable import PrettyTable

import wandb

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
    parser.add_argument('--model_path', default=None, type=str,
                        help='pass model path for evaluation')
    parser.add_argument('--exp_name', '-n', default='exp', type=str)
    parser.add_argument('--frame_id', default=None, type=int,
                        help='which frame to process')

    # args, rest = parser.parse_known_args()
    # update_config(args.cfg)

    # if set then update the config.
    args, unknown = parser.parse_known_args()
    update_config(args.cfg)

    update_config_dynamic_input(unknown)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'validate')
    device = torch.device(args.device)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if is_main_process():
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

        if config.DEBUG.WANDB_KEY:
            wandb.login(key=config.DEBUG.WANDB_KEY)
        if config.DEBUG.WANDB_NAME:
            wandb.init(project="mvp-val",name=config.DEBUG.WANDB_NAME)
        else:
            # close wandb
            pass

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        rank, world_size = get_dist_info()
        sampler_val = DistributedSampler(test_dataset, world_size, rank,
                                         shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
        # print('Notice : random sampler is used.')
        # sampler_val = torch.utils.data.RandomSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)

    num_views = test_dataset.num_views

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + 'dq_transformer' + '.get_mvp')(
        config, is_train=True)
    model.to(device)
    # with torch.no_grad():
    #     model = torch.nn.DataParallel(model, device_ids=0).cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.model_path is not None:
        logger.info('=> load models state {}'.format(args.model_path))
        if args.distributed:
            model.module.load_state_dict(torch.load(args.model_path), strict=False)
        else:
            model.load_state_dict(torch.load(args.model_path), strict=False)
    elif os.path.isfile(
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)):
        test_model_file = \
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file),strict=False)
    else:
        test_model_file = \
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)
        raise ValueError(f'Check the model file for test file:{test_model_file}!')

    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    exp_name = args.exp_name

    os.makedirs(final_output_dir+'/validate-{}-{}/'.format(exp_name,now_str), exist_ok=True)

    conf_thr_tb = PrettyTable()
    mpjpe_threshold = np.arange(25, 155, 25)
    conf_thr_tb.field_names = \
        ["inference_conf_thr"] + \
        [f'AP{i}' for i in mpjpe_threshold] + \
        [f'Recall{i}' for i in mpjpe_threshold] + \
        ['Recall500','MPJPE']

    process_frame_id = args.frame_id
    for thr in config.DECODER.inference_conf_thr:
        pred_path = os.path.join(final_output_dir, f"{config.TEST.PRED_FILE}-{thr}.npy")
        if config.TEST.PRED_FILE and os.path.isfile(pred_path):
            preds = np.load(pred_path)
            logger.info(f"=> load pred_file from {pred_path}")
        else:
            preds_single, meta_image_files_single = validate_3d(
                config, model, test_loader, final_output_dir, thr,
                num_views=num_views, device=device, frame_id = process_frame_id)
            preds = collect_results(preds_single, len(test_dataset))
            np.save(final_output_dir+'/{}-{}-{}.npy'.format(exp_name,now_str,thr),preds)
            logger.info(f"=> save pred_file with TEST.PRED_FILE={exp_name}-{now_str}")

        if is_main_process():
            precision = None

            if 'panoptic' in config.DATASET.TEST_DATASET \
                    or 'h36m' in config.DATASET.TEST_DATASET:
                if config.DATASET.NMS_DETAIL:
                    
                    nms_tb = PrettyTable()
                    nms_tb.field_names = \
                        ["dist_thr","num_nearby_joints_thr"] + \
                        [f'AP{i}' for i in mpjpe_threshold] + \
                        [f'Recall{i}' for i in mpjpe_threshold] + \
                        ['Recall500','MPJPE']

                    if config.DATASET.NMS_DETAIL_ALL: 
                        dist_thrs = [0.01,0.03,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.8]
                        num_nearby_joints_thrs = [3,4,5,6,7,8,9,10,13]
                    else: # default
                        dist_thrs = [0.3]
                        num_nearby_joints_thrs = [7]

                    for dist_thr in dist_thrs:
                        for num_nearby_joints_thr in num_nearby_joints_thrs:
                            preds_nms = []
                            preds_nms_num = 0
                            for pred in preds:
                                pred = pred[pred[:, 0, 3] >= 0]
                                indices = nearby_joints_nms(pred,dist_thr,num_nearby_joints_thr)
                                pred_nms = pred[indices]
                                preds_nms_num = preds_nms_num + len(indices)
                                preds_nms.append(pred_nms.copy())
                            aps, recs, mpjpe, recall500 = test_loader.dataset.evaluate(preds_nms)
                            nms_tb.add_row( 
                                [dist_thr , num_nearby_joints_thr] + 
                                [f'{ap * 100:.2f}' for ap in aps] +
                                [f'{re * 100:.2f}' for re in recs] +
                                [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
                            )
                            # logger.info(nms_tb) #! DEBUG 
                    # logger.info(nms_tb)

                # nomral evo w/o NMS
                # aps, recs, mpjpe, recall500 = \
                #     test_loader.dataset.evaluate(preds)
                # conf_thr_tb.add_row( 
                #     [thr] + 
                #     [f'{ap * 100:.2f}' for ap in aps] +
                #     [f'{re * 100:.2f}' for re in recs] +
                #     [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
                # )

                # upper bound
                output_upper_bound = False 
                if output_upper_bound:
                    aps_upper, recs_upper, mpjpe_upper, recall500_upper = \
                        test_loader.dataset.evaluate(preds, method='mpjpe_sort')
                    conf_thr_tb.add_row( 
                        ['upper_bound (debug)'] + 
                        [f'{ap * 100:.2f}' for ap in aps_upper] +
                        [f'{re * 100:.2f}' for re in recs_upper] +
                        [f'{recall500_upper * 100:.2f}',f'{mpjpe_upper:.2f}']
                    )    

                # print table values
                logger.info(nms_tb.get_string(fields=nms_tb.field_names[2:]))

            elif 'campus' in config.DATASET.TEST_DATASET \
                    or 'shelf' in config.DATASET.TEST_DATASET:
                actor_pcp, avg_pcp, _, recall = \
                    test_loader.dataset.evaluate(preds)
                msg = '     | Actor 1 | Actor 2 | Actor 3 | Average | \n' \
                    ' PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  ' \
                    '|  {pcp_3:.2f}  |  {pcp_avg:.2f}  |' \
                    '\t Recall@500mm: {recall:.4f}'\
                    .format(pcp_1=actor_pcp[0] * 100,
                            pcp_2=actor_pcp[1] * 100,
                            pcp_3=actor_pcp[2] * 100,
                            pcp_avg=avg_pcp * 100,
                            recall=recall)
                logger.info(msg)
                precision = np.mean(avg_pcp)

if __name__ == '__main__':
    main()
