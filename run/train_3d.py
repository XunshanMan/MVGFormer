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

from datetime import datetime

import sys

import wandb
from core.nms import nearby_joints_nms

##########################################
# torch.autograd.set_detect_anomaly(True)
# print('Open anomaly detection!')
#########################################

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
    parser.add_argument('--exp_name', '-n', default='exp', type=str)

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

def save_config(config, config_wdb):
    # copy all attributes in config to the other
    config_wdb.batch_size = config.TRAIN.BATCH_SIZE

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

        if config.DEBUG.WANDB_KEY:
            wandb.login(key=config.DEBUG.WANDB_KEY)
        if config.DEBUG.WANDB_NAME:
            wandb.init(project="mvgformer",name=config.DEBUG.WANDB_NAME)
        else:
            wandb.init(project="mvgformer")

        wandb.define_metric("AP25", summary="max")
        wandb.define_metric("Recall25", summary="max")
        wandb.define_metric("MPJPE-mm", summary="min")
        save_config(config, wandb.config)

        now = datetime.now()
        now_str = now.strftime("%Y%m%d-%H%M%S")
        exp_name = args.exp_name
        final_output_dir += '/{}-{}/'.format(exp_name,now_str)
        os.makedirs(final_output_dir, exist_ok=True)

        # save all print to files!
        if config.DEBUG.PRINT_TO_FILE:
            f = open(final_output_dir+'/log.txt', 'w')
            sys.stdout = f # Change the standard output to the file we created.

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    num_views = train_dataset.num_views

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
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

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    if hasattr(config, 'TRANSFORMER'):
        model = eval('models.' + config.TRANSFORMER + '.get_mvp')(
            config, is_train=True)
    else:
        model = eval('models.' + 'multi_view_pose_transformer' + '.get_mvp')(
            config, is_train=True)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.\
            DistributedDataParallel(model, device_ids=[args.gpu],
                                    find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = get_optimizer(model_without_ddp,
                              args.weight_decay, config.DECODER.optimizer)

    end_epoch = config.TRAIN.END_EPOCH

    if config.NETWORK.PRETRAINED_BACKBONE:
        _ = load_backbone_panoptic(model_without_ddp,
                                   config.NETWORK.PRETRAINED_BACKBONE)
    if config.TRAIN.FINETUNE_MODEL is not None:
        start_epoch, _, optimizer, best_precision = load_checkpoint_best(
            model_without_ddp, optimizer,
            './models', config.TRAIN.FINETUNE_MODEL)

    if config.TRAIN.RESUME:
        start_epoch, _, checkpoint, optimizer,  best_precision \
            = load_checkpoint(model_without_ddp, optimizer, final_output_dir)
    else:
        start_epoch, checkpoint, best_precision = 0, None, 0

    # list for step decay
    if isinstance(config.DECODER.lr_decay_epoch, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.DECODER.lr_decay_epoch, gamma=0.1)
        if checkpoint is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # int for cosine decay
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.DECODER.lr_decay_epoch, eta_min=1e-5)
        if checkpoint is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print(get_host_info())
    print('=> Training...')

    # n_parameters = sum(p.numel() for p in model.parameters()
    #                    if p.requires_grad)
    # print('number of params:', n_parameters)

    open_evo = True
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))
        print('current lr {}'.format(optimizer.param_groups[0]["lr"]))
        final_output_dir_in = final_output_dir + f'/epoch-{epoch}/'
        train_3d(config, model, optimizer, train_loader, epoch,
                 final_output_dir_in, num_views=num_views)

        lr_scheduler.step()

        if open_evo:
            inference_conf_thr = config.DECODER.inference_conf_thr  # [0.1,] in config
            for thr in inference_conf_thr:
                preds_single, meta_image_files_single = validate_3d(
                    config, model, test_loader,
                    final_output_dir_in+'/validation/', thr, num_views=num_views, device=device, epoch=epoch)
                preds = collect_results(preds_single, len(test_dataset))

                if is_main_process():

                    precision = None

                    if 'panoptic' in config.DATASET.TEST_DATASET \
                            or 'h36m' in config.DATASET.TEST_DATASET:
                        # TODO: Update with new outputs. and with NMS.
                        mpjpe_threshold = np.arange(25, 155, 25)
                        if config.DATASET.NMS_DETAIL:
                            nms_tb = PrettyTable()
                            nms_tb.field_names = \
                                ["dist_thr","num_nearby_joints_thr"] + \
                                [f'AP{i}' for i in mpjpe_threshold] + \
                                [f'Recall{i}' for i in mpjpe_threshold] + \
                                ['Recall500','MPJPE']
                                
                            # Only use one 
                            dist_thr = 0.3
                            num_nearby_joints_thr = 7
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
                            logger.info(nms_tb)
                            
                            precision_nms = np.mean(aps[0])

                        tb = PrettyTable()
                        aps, recs, mpjpe, recall500 = \
                            test_loader.dataset.evaluate(preds)
                        tb.field_names = ['Threshold/mm'] + \
                                        [f'{i}' for i in mpjpe_threshold]
                        tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
                        tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
                        tb.add_row(['recall@500mm'] +
                                [f'{recall500 * 100:.2f}' for re in recs])
                        logger.info(tb)
                        logger.info(f'MPJPE: {mpjpe:.2f}mm')
                        precision_no_nms = np.mean(aps[0])

                        # wandb log mAP25 etc. to log history
                        log_dict = {}
                        range_num = len(mpjpe_threshold)
                        for i in range(range_num):
                            th = mpjpe_threshold[i]
                            log_dict[f'AP{th}'] = aps[i] * 100
                            log_dict[f'Recall{th}'] = recs[i] * 100
                        log_dict['Recall@500mm'] = recall500 * 100
                        log_dict['MPJPE-mm'] = mpjpe
                        wandb.log(log_dict)
                        
                        # Check if use nms, we use precision_nms
                        if config.DATASET.NMS_DETAIL:
                            precision = precision_nms
                        else:
                            precision = precision_no_nms

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

                    if precision > best_precision:
                        best_precision = precision
                        best_model = True
                    else:
                        best_model = False
                    if isinstance(config.DECODER.lr_decay_epoch, list):  #
                        logger.info('=> saving checkpoint to {} (Best: {})'.
                                    format(final_output_dir, best_model))
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.module.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'precision': best_precision,
                            'optimizer': optimizer.state_dict(),
                        }, best_model, final_output_dir)
                    else:
                        logger.info('=> saving checkpoint to {} (Best: {})'.
                                    format(final_output_dir, best_model))
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.module.state_dict(),
                            'precision': best_precision,
                            'optimizer': optimizer.state_dict(),
                        }, best_model, final_output_dir)
                dist.barrier()

    if is_main_process():
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
