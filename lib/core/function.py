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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import torch

from utils.vis import save_debug_3d_images

from models.util.misc import get_total_grad_norm, is_main_process


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


logger = logging.getLogger(__name__)

import wandb

def wandb_log_io(data):
    CLOSE_WANDB = True

    if CLOSE_WANDB:
        return

    wandb.log(data)

def train_3d(config, model, optimizer, loader, epoch,
             output_dir, device=torch.device('cuda'), num_views=5):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    batch_model_time = AverageMeter()
    batch_backward_time = AverageMeter()
    batch_calculoss_time = AverageMeter()

    loss_ce = AverageMeter()
    class_error = AverageMeter()
    class_recall = AverageMeter()
    class_precision = AverageMeter()
    loss_pose_perjoint = AverageMeter()
    loss_init = AverageMeter()
    loss_pose_perbone = AverageMeter()
    loss_pose_perprojection = AverageMeter()
    loss_pose_perprojection_2d = AverageMeter()
    cardinality_error = AverageMeter()
    losses_all = AverageMeter()

    key_value_layers = {}

    model.train()

    is_distributed = isinstance(model,torch.nn.parallel.DistributedDataParallel)

    if is_distributed :
        if model.module.backbone is not None:
            # Comment out this line if you want to train 2D backbone jointly
            model.module.backbone.eval()
            threshold = model.module.pred_conf_threshold

        weight_dict =  model.module.criterion.weight_dict
    else:
        if model.backbone is not None:
            # Comment out this line if you want to train 2D backbone jointly
            model.backbone.eval()
            threshold = model.pred_conf_threshold
        weight_dict =  model.criterion.weight_dict

    

    end = time.time()

    # begin going through all the data
    for frame_id, (inputs, meta) in enumerate(loader):
        assert len(inputs) == num_views
        inputs = [i.to(device) for i in inputs]
        meta = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()} for t in meta]
        data_time.update(time_synchronized() - end)
        end = time_synchronized()

        out, loss_dict = model(views=inputs, meta=meta, output_dir=output_dir,frame_id=frame_id)
        end_model = time_synchronized()
        batch_model_time.update(end_model - end)

        gt_3d = meta[0]['joints_3d'].float()
        num_joints = gt_3d.shape[2]
        bs, num_queries = out["pred_logits"].shape[:2]

        src_poses = out['pred_poses']['outputs_coord'].\
            view(bs, num_queries, num_joints, 3)
        # src_poses = model.norm2absolute(src_poses) 
        score = out['pred_logits'][:, :, 1:2].sigmoid() # Choose the 2nd label: positive prob
        score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
        temp = (score > threshold).float() - 1

        pred = torch.cat([src_poses, temp, score], dim=-1)

        # sum the losses together. They are calculated in the model
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
        losses_all.update(losses.sum().item())

        loss_ce.update(loss_dict['loss_ce'].sum().item())
        class_error.update(loss_dict['class_error'].sum().item())
        class_recall.update(loss_dict['class_recall'].sum().item())
        class_precision.update(loss_dict['class_precision'].sum().item())

        loss_pose_perjoint.update(loss_dict['loss_pose_perjoint'].sum().item())
        loss_init.update(loss_dict['loss_init'].sum().item())
        if 'loss_pose_perbone' in loss_dict:
            loss_pose_perbone.update(
                loss_dict['loss_pose_perbone'].sum().item())
        if 'loss_pose_perprojection' in loss_dict:
            loss_pose_perprojection.update(
                loss_dict['loss_pose_perprojection'].sum().item())
        if 'loss_pose_perprojection_2d' in loss_dict:
            loss_pose_perprojection_2d.update(
                loss_dict['loss_pose_perprojection_2d'].sum().item())
        if 'cardinality_error' in loss_dict:
            cardinality_error.update(
                loss_dict['cardinality_error'].sum().item())

        if 'dict_losses_layers' in loss_dict:
            # calculate losses for each layer!
            num_layers = len(loss_dict['dict_losses_layers'])
            key_list = ['loss_pose_perjoint'] # output those losses for each layers
            for layer_id in range(num_layers):
                for key in key_list:
                    key_value = loss_dict['dict_losses_layers'][layer_id][key].item()
                    # print(f'{key}_l{layer_id}: {key_value}')
                    if not key in key_value_layers:
                        # key_value_layers[key] = []
                        key_value_layers[key] = [AverageMeter() for p in range(num_layers)]
                    key_value_layers[key][layer_id].update(key_value)

        end_calculoss = time_synchronized()
        batch_calculoss_time.update(end_calculoss - end_model)

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()

            if config.TRAIN.clip_max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.clip_max_norm)
            else:
                grad_total_norm = get_total_grad_norm(
                    model.parameters(), config.TRAIN.clip_max_norm)

            optimizer.step()

            end_backward = time_synchronized()
            batch_backward_time.update(end_backward - end_calculoss)
        else:
            print('Please check loss status:', losses)
            grad_total_norm = torch.Tensor([0])

        batch_time.update(time_synchronized() - end)
        end = time_synchronized()

        # print info
        if is_main_process() and ((frame_id % config.PRINT_FREQ == 0 and frame_id > 0) or (frame_id == len(loader) - 1)):
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = \
                'Epoch: [{0}][{1}/{2}]\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ' '({data_time.avg:.3f}s)\t' \
                'losses_all: {losses_all.val:.7f} ' '({losses_all.avg:.7f})\t' \
                'loss_ce: {loss_ce.val:.7f} ' '({loss_ce.avg:.7f})\t' \
                'class_recall: {class_recall.val:.7f} ' \
                '({class_recall.avg:.7f})\t' \
                'class_precision: {class_precision.val:.7f} ' \
                '({class_precision.avg:.7f})\t' \
                'class_error: {class_error.val:.7f} ' \
                '({class_error.avg:.7f})\t' \
                'loss_pose_perjoint: {loss_pose_perjoint.val:.6f} ' \
                '({loss_pose_perjoint.avg:.6f})\t' \
                'loss_pose_perbone: {loss_pose_perbone.val:.6f} ' \
                '({loss_pose_perbone.avg:.6f})\t' \
                'loss_pose_perprojection: {loss_pose_perprojection.val:.6f} ' \
                '({loss_pose_perprojection.avg:.6f})\t' \
                'loss_pose_perprojection_2d: {loss_pose_perprojection_2d.val:.6f} ' \
                '({loss_pose_perprojection_2d.avg:.6f})\t' \
                'cardinality_error: {cardinality_error.val:.6f} ' \
                '({cardinality_error.avg:.6f})\t' \
                'Memory {memory:.1f}\t'\
                'gradnorm {gradnorm:.2f}'.format(
                  epoch, frame_id, len(loader),
                  batch_time=batch_time,
                  speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                  data_time=data_time,
                  losses_all=losses_all,
                  loss_ce=loss_ce,
                  class_error=class_error,
                  class_recall=class_recall,
                  class_precision=class_precision,
                  loss_pose_perjoint=loss_pose_perjoint,
                  loss_pose_perbone=loss_pose_perbone,
                  loss_pose_perprojection=loss_pose_perprojection,
                  loss_pose_perprojection_2d=loss_pose_perprojection_2d,
                  cardinality_error=cardinality_error,
                  memory=gpu_memory_usage,
                  gradnorm=grad_total_norm)
            logger.info(msg)
            msg_time = \
                'batch_all: {batch_all.val:.3f}s ({batch_all.avg:.3f}s)\t' \
                'batch_model: {batch_model.val:.3f}s ({batch_model.avg:.3f}s)\t' \
                'batch_calculoss: {batch_calculoss.val:.3f}s ({batch_calculoss.avg:.3f}s)\t' \
                'batch_backward: {batch_backward.val:.3f}s ({batch_backward.avg:.3f}s)\t'.format(
                    batch_all=batch_time,
                    batch_model=batch_model_time,
                    batch_calculoss=batch_calculoss_time,
                    batch_backward=batch_backward_time
                )
            logger.info(msg_time)

            wandb_log_dict_layers = {}
            if 'dict_losses_layers' in loss_dict:
                # loss per layers
                # calculate losses for each layer!
                num_layers = len(loss_dict['dict_losses_layers'])
                key_list = ['loss_pose_perjoint']
                for layer_id in range(num_layers):
                    for key in key_list:
                        key_value = key_value_layers[key][layer_id].val
                        # print(f'{key}_l{layer_id}: {key_value}')
                        wandb_log_dict_layers[f'{key}_l{layer_id}'] = key_value
                        # wandb_log_io({
                            # f'{key}_l{layer_id}' : key_value
                        # })

                if (frame_id == len(loader) - 1):
                    for layer_id in range(num_layers):
                        for key in key_list:
                            key_value_avg = key_value_layers[key][layer_id].avg
                            # print(f'epoch_{key}_l{layer_id}: {key_value_avg}')
                            wandb_log_io({
                                f'epoch-{key}_l{layer_id}' : key_value_avg
                            })

            wandb_log_dict_per_iter = {
                "epoch" : epoch, 
                "frame_id" : frame_id, 
                "all_frames": len(loader),
                "batch_time" : batch_time.val,
                "speed" : len(inputs) * inputs[0].size(0) / batch_time.val,
                "data_time" : data_time.val,
                "losses_all" : losses_all.val,
                "loss_ce" : loss_ce.val,
                "class_error" : class_error.val,
                "class_recall" : class_recall.val,
                "class_precision" : class_precision.val,
                "loss_pose_perjoint" : loss_pose_perjoint.val,
                "loss_init" : loss_init.val,
                "loss_pose_perbone" : loss_pose_perbone.val,
                "loss_pose_perprojection" : loss_pose_perprojection.val,
                "loss_pose_perprojection_2d" : loss_pose_perprojection_2d.val,
                "cardinality_error" : cardinality_error.val,
                "memory" : gpu_memory_usage,
                "gradnorm" : grad_total_norm.item()
            }

            # log for the epoch
            wandb_log_dict_per_epoch = {}
            if (frame_id == len(loader) - 1):
                # wandb_log_io({
                wandb_log_dict_per_epoch = {
                    "epoch-batch_time" : batch_time.avg,
                    "epoch-speed" : len(inputs) * inputs[0].size(0) / batch_time.avg,
                    "epoch-data_time" : data_time.avg,
                    "epoch-losses_all" : losses_all.avg,
                    "epoch-loss_ce" : loss_ce.avg,
                    "epoch-class_error" : class_error.avg,
                    "epoch-class_recall" : class_recall.avg,
                    "epoch-class_precision" : class_precision.avg,
                    "epoch-loss_pose_perjoint" : loss_pose_perjoint.avg,
                    "epoch-loss_init" : loss_init.avg,
                    "epoch-loss_pose_perbone" : loss_pose_perbone.avg,
                    "epoch-loss_pose_perprojection" : loss_pose_perprojection.avg,
                    "epoch-loss_pose_perprojection_2d" : loss_pose_perprojection_2d.avg,
                    "epoch-cardinality_error" : cardinality_error.avg,
                }
            
            # cat all and log
            wandb_log_dict = {}
            wandb_log_dict.update(wandb_log_dict_layers)
            wandb_log_dict.update(wandb_log_dict_per_iter)
            wandb_log_dict.update(wandb_log_dict_per_epoch)
            wandb_log_io(wandb_log_dict)
                
            # prefix2 = '{}_{:08}'.format(
            #     os.path.join(output_dir, 'train'), frame_id)
            # save_debug_3d_images(config, meta[0], pred, prefix2)
            
        # Only Train Once
        # print('ONLY TRAIN ONCE.')
        # break


def validate_3d(config, model, loader, output_dir, threshold, num_views=5, device="cpu", epoch=0, frame_id=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    log_val_loss = config.DEBUG.LOG_VAL_LOSS

    if log_val_loss:
        loss_ce = AverageMeter()
        class_error = AverageMeter()
        class_recall = AverageMeter()
        class_precision = AverageMeter()
        loss_pose_perjoint = AverageMeter()
        loss_init = AverageMeter()
        loss_pose_perbone = AverageMeter()
        loss_pose_perprojection = AverageMeter()
        loss_pose_perprojection_2d = AverageMeter()
        cardinality_error = AverageMeter()
        losses_all = AverageMeter()
        key_value_layers = {}

    model.eval()

    preds = []
    meta_image_files = []
    
    # if frame_id is None:
    loader_all_frames = enumerate(loader)
    # else:
    #     # meta also needs batch version
    #     (inputs, meta) = loader.dataset.__getitem__(frame_id)
    #     inputs = [i.unsqueeze(0) for i in inputs]
    #     meta = [n.unsqueeze(0) for n in meta]
    #     loader_all_frames = enumerate([(inputs, meta)])
        
    with torch.no_grad():
        end = time.time()
        for i, (inputs, meta) in loader_all_frames:
            if frame_id is not None:
                if i != frame_id:
                    continue
            
            data_time.update(time.time() - end)
            assert len(inputs) == num_views

            inputs = [i.to(device) for i in inputs]
            meta = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in meta]
            if log_val_loss:
                output, loss_dict = model(views=inputs, meta=meta,output_dir=output_dir,frame_id=i, threshold=threshold)
            else:
                output = model(views=inputs, meta=meta,output_dir=output_dir,frame_id=i, threshold=threshold)

            meta_image_files.append(meta[0]['image'])
            gt_3d = meta[0]['joints_3d'].float()
            num_joints = gt_3d.shape[2]
            bs, num_queries = output["pred_logits"].shape[:2]

            src_poses = output['pred_poses']['outputs_coord'].\
                view(bs, num_queries, num_joints, 3)
            # src_poses = model.norm2absolute(src_poses)
            score = output['pred_logits'][:, :, 1:2].sigmoid() # attention: score only uses 2nd dim of classification
            score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
            temp = (score > threshold).float() - 1

            pred = torch.cat([src_poses, temp, score], dim=-1)
            pred = pred.detach().cpu().numpy() # x,y,z,1/0,score
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            batch_time.update(time.time() - end)
            end = time.time()
            if ((i % config.PRINT_FREQ == 0 and i > 0) or i == len(loader) - 1) \
                    and is_main_process():
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

            if log_val_loss:
                is_distributed = isinstance(model,torch.nn.parallel.DistributedDataParallel)
                if is_distributed :
                    weight_dict =  model.module.criterion.weight_dict
                else:
                    weight_dict =  model.criterion.weight_dict

                # sum the losses together. They are calculated in the model
                losses = sum(loss_dict[k] * weight_dict[k]
                            for k in loss_dict.keys() if k in weight_dict)
                losses_all.update(losses.sum().item())

                loss_ce.update(loss_dict['loss_ce'].sum().item())
                class_error.update(loss_dict['class_error'].sum().item())
                class_recall.update(loss_dict['class_recall'].sum().item())
                class_precision.update(loss_dict['class_precision'].sum().item())

                loss_pose_perjoint.update(loss_dict['loss_pose_perjoint'].sum().item())
                loss_init.update(loss_dict['loss_init'].sum().item())
                if 'loss_pose_perbone' in loss_dict:
                    loss_pose_perbone.update(
                        loss_dict['loss_pose_perbone'].sum().item())
                if 'loss_pose_perprojection' in loss_dict:
                    loss_pose_perprojection.update(
                        loss_dict['loss_pose_perprojection'].sum().item())
                if 'loss_pose_perprojection_2d' in loss_dict:
                    loss_pose_perprojection_2d.update(
                        loss_dict['loss_pose_perprojection_2d'].sum().item())
                if 'cardinality_error' in loss_dict:
                    cardinality_error.update(
                        loss_dict['cardinality_error'].sum().item())

                if 'dict_losses_layers' in loss_dict:
                    # calculate losses for each layer!
                    num_layers = len(loss_dict['dict_losses_layers'])
                    key_list = ['loss_pose_perjoint'] # output those losses for each layers
                    for layer_id in range(num_layers):
                        for key in key_list:
                            key_value = loss_dict['dict_losses_layers'][layer_id][key].item()
                            # print(f'{key}_l{layer_id}: {key_value}')
                            if not key in key_value_layers:
                                # key_value_layers[key] = []
                                key_value_layers[key] = [AverageMeter() for p in range(num_layers)]
                            key_value_layers[key][layer_id].update(key_value)

                if is_main_process() and ((i % config.PRINT_FREQ == 0 and i > 0 ) or (i == len(loader) - 1)):
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed: {speed:.1f} samples/s\t' \
                        'Data: {data_time.val:.3f}s ' '({data_time.avg:.3f}s)\t' \
                        'losses_all: {losses_all.val:.7f} ' '({losses_all.avg:.7f})\t' \
                        'loss_ce: {loss_ce.val:.7f} ' '({loss_ce.avg:.7f})\t' \
                        'class_recall: {class_recall.val:.7f} ' \
                        '({class_recall.avg:.7f})\t' \
                        'class_precision: {class_precision.val:.7f} ' \
                        '({class_precision.avg:.7f})\t' \
                        'class_error: {class_error.val:.7f} ' \
                        '({class_error.avg:.7f})\t' \
                        'loss_pose_perjoint: {loss_pose_perjoint.val:.6f} ' \
                        '({loss_pose_perjoint.avg:.6f})\t' \
                        'loss_init: {loss_init.val:.6f} ' \
                        '({loss_init.avg:.6f})\t' \
                        'loss_pose_perbone: {loss_pose_perbone.val:.6f} ' \
                        '({loss_pose_perbone.avg:.6f})\t' \
                        'loss_pose_perprojection: {loss_pose_perprojection.val:.6f} ' \
                        '({loss_pose_perprojection.avg:.6f})\t' \
                        'loss_pose_perprojection_2d: {loss_pose_perprojection_2d.val:.6f} ' \
                        '({loss_pose_perprojection_2d.avg:.6f})\t' \
                        'cardinality_error: {cardinality_error.val:.6f} ' \
                        '({cardinality_error.avg:.6f})\t' \
                        'Memory {memory:.1f}\t'.format(
                        i, len(loader),
                        batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time,
                        losses_all=losses_all,
                        loss_ce=loss_ce,
                        class_error=class_error,
                        class_recall=class_recall,
                        class_precision=class_precision,
                        loss_pose_perjoint=loss_pose_perjoint,
                        loss_init=loss_init,
                        loss_pose_perbone=loss_pose_perbone,
                        loss_pose_perprojection=loss_pose_perprojection,
                        loss_pose_perprojection_2d=loss_pose_perprojection_2d,
                        cardinality_error=cardinality_error,
                        memory=gpu_memory_usage)
                    logger.info(msg)

                    wandb_log_dict_layers = {}
                    if 'dict_losses_layers' in loss_dict:
                        # loss per layers
                        # calculate losses for each layer!
                        num_layers = len(loss_dict['dict_losses_layers'])
                        key_list = ['loss_pose_perjoint']
                        for layer_id in range(num_layers):
                            for key in key_list:
                                key_value = key_value_layers[key][layer_id].val
                                # print(f'{key}_l{layer_id}: {key_value}')
                                # wandb_log_io({
                                    # f'val-{key}_l{layer_id}' : key_value
                                # })
                                wandb_log_dict_layers[f'val-{key}_l{layer_id}'] = key_value
                        
                        if (i == len(loader) - 1):
                            for layer_id in range(num_layers):
                                for key in key_list:
                                    key_value_avg = key_value_layers[key][layer_id].avg
                                    # print(f'epoch-val-{key}_l{layer_id}: {key_value_avg}')
                                    wandb_log_io({
                                        f'epoch-val-{key}_l{layer_id}' : key_value_avg
                                    })

                    wandb_log_dict_per_iter = {
                    # wandb_log_io({
                        "val-epoch" : epoch,
                        "val-frame_id" : i, 
                        "val-all_frames": len(loader),
                        "val-batch_time" : batch_time.val,
                        "val-speed" : len(inputs) * inputs[0].size(0) / batch_time.val,
                        "val-data_time" : data_time.val,
                        "val-losses_all" : losses_all.val,
                        "val-loss_ce" : loss_ce.val,
                        "val-class_error" : class_error.val,
                        "val-class_recall" : class_recall.val,
                        "val-class_precision" : class_precision.val,
                        "val-loss_pose_perjoint" : loss_pose_perjoint.val,
                        "val-loss_init" : loss_init.val,
                        "val-loss_pose_perbone" : loss_pose_perbone.val,
                        "val-loss_pose_perprojection" : loss_pose_perprojection.val,
                        "val-loss_pose_perprojection_2d" : loss_pose_perprojection_2d.val,
                        "val-cardinality_error" : cardinality_error.val,
                        "val-memory" : gpu_memory_usage
                    # })
                    }

                    wandb_log_dict_per_epoch = {}
                    if (i == len(loader) - 1):
                        # wandb_log_io({
                        wandb_log_dict_per_epoch = {
                            "epoch-val-epoch" : epoch,
                            "epoch-val-batch_time" : batch_time.avg,
                            "epoch-val-speed" : len(inputs) * inputs[0].size(0) / batch_time.avg,
                            "epoch-val-data_time" : data_time.avg,
                            "epoch-val-losses_all" : losses_all.avg,
                            "epoch-val-loss_ce" : loss_ce.avg,
                            "epoch-val-class_error" : class_error.avg,
                            "epoch-val-class_recall" : class_recall.avg,
                            "epoch-val-class_precision" : class_precision.avg,
                            "epoch-val-loss_pose_perjoint" : loss_pose_perjoint.avg,
                            "epoch-val-loss_init" : loss_init.avg,
                            "epoch-val-loss_pose_perbone" : loss_pose_perbone.avg,
                            "epoch-val-loss_pose_perprojection" : loss_pose_perprojection.avg,
                            "epoch-val-loss_pose_perprojection_2d" : loss_pose_perprojection_2d.avg,
                            "epoch-val-cardinality_error" : cardinality_error.avg,
                        }
                    ###########################################

                    wandb_log_dict = {}
                    wandb_log_dict.update(wandb_log_dict_layers)
                    wandb_log_dict.update(wandb_log_dict_per_iter)
                    wandb_log_dict.update(wandb_log_dict_per_epoch)
                    wandb_log_io(wandb_log_dict)

                # prefix2 = '{}_{:08}'.format(
                #     os.path.join(output_dir, 'validation'), i)
                # save_debug_3d_images(config, meta[0], pred, prefix2)
            if frame_id is not None:
                print('finish processing frame_id:', frame_id)
                break

    return preds, meta_image_files


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
