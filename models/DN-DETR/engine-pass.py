# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import cv2
import numpy as np
from PIL import Image
from util.visualizer import COCOVisualizer
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
                if args.contrastive is not False:
                    dn_args += (args.contrastive,)

                outputs, mask_dict = model(samples, dn_args=dn_args)
                loss_dict = criterion(outputs, targets, mask_dict)
            else:
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("BREAK!"*5)
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    # 初始化基础参数
    try:
        need_tgt_for_training = args.use_dn
        score_threshold = 0.2  # person置信度阈值（简单固定，避免复杂参数）
    except:
        need_tgt_for_training = False
        score_threshold = 0.2

    model.eval()
    criterion.eval()

    # 初始化日志与图片保存目录
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 初始化图片保存目录（仅person类可视化）
    save_img_dir = None
    if args and args.save_results:
        save_img_dir = os.path.join(output_dir, 'results_person_only')
        os.makedirs(save_img_dir, exist_ok=True)
        print(f"仅person类检测框图片保存至：{save_img_dir}")

    # 初始化COCO评估器（仅评估person类）
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # 初始化全景分割评估器（若有）
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # 保存仅person类的数值结果
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        # 模型推理
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, _ = model(samples, dn_args=args.num_patterns)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # 损失日志更新
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # --------------------------
        # 核心：仅保留person类（COCO ID=1）检测结果
        # --------------------------
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        raw_results = postprocessors['bbox'](outputs, orig_target_sizes)  # 原始全类别结果
        
        # 过滤：只保留 label=1（person）且置信度>阈值的结果
        person_results = []
        for res in raw_results:
            person_mask = (res['labels'] == 1) & (res['scores'] > score_threshold)
            filtered_res = {
                'boxes': res['boxes'][person_mask],    # 已过滤的person框（x1,y1,x2,y2）
                'scores': res['scores'][person_mask],  # 已过滤的person置信度
                'labels': res['labels'][person_mask]   # 已过滤的person标签（全为1）
            }
            person_results.append(filtered_res)
        results = person_results  # 替换为仅含person的结果
        # --------------------------

        # 分割结果处理（若有，同样仅保留person相关）
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # 评估器更新（仅基于person类结果）
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # 全景分割评估（若有）
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)
        
        # --------------------------
        # 修复：保存仅person类的数值结果（用已过滤的res['boxes']，避免outbbox维度错误）
        # --------------------------
        if args.save_results:
            # 遍历每个样本的结果（已过滤为person）
            for i, (tgt, res) in enumerate(zip(targets, results)):  # 去掉outbbox参数
                # 1. 过滤GT中的非person类
                gt_person_mask = tgt['labels'] == 1
                gt_bbox = tgt['boxes'][gt_person_mask]
                gt_label = tgt['labels'][gt_person_mask]
                if len(gt_bbox) == 0:
                    gt_info = torch.tensor([]).reshape(0, 5).to(device)
                else:
                    gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # 2. 直接用已过滤的res['boxes']（无需再索引outbbox）
                _res_bbox = res['boxes']  # 重点修复：改用已过滤的person框
                _res_prob = res['scores']
                _res_label = res['labels']
                if len(_res_bbox) == 0:
                    res_info = torch.tensor([]).reshape(0, 6).to(device)
                else:
                    res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)

                # 3. 存入结果字典
                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # --------------------------
            # 绘制：仅person类的检测框（无改动，已正确）
            # --------------------------
            for i, (tgt, res) in enumerate(zip(targets, results)):
                # 获取图片信息
                image_id = tgt['image_id'].item()
                img_info = base_ds.loadImgs(image_id)[0]
                img_filename = img_info['file_name']
                img_path = os.path.join(args.coco_path, 'test2017', img_filename)

                # 加载原图（处理异常）
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except FileNotFoundError:
                    print(f"警告：图片 {img_path} 不存在，跳过保存")
                    continue

                # 绘制person检测框（已过滤，无其他类别）
                person_boxes = res['boxes'].cpu().numpy()
                person_scores = res['scores'].cpu().numpy()
                for box, score in zip(person_boxes, person_scores):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"person: {score:.2f}"
                    text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img_cv, (x1, y1 - text_h - 5), (x1 + text_w, y1 - 5), (0, 255, 0), -1)
                    cv2.putText(img_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 保存图片
                save_img_path = os.path.join(save_img_dir, img_filename)
                cv2.imwrite(save_img_path, img_cv)
                print(f"已保存person检测图：{save_img_path}")

        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("BREAK!"*5)
            break

    # 保存仅person类的pkl数值结果
    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results_person_only-{}.pkl'.format(utils.get_rank()))
        print(f"保存仅person类数值结果至：{savepath}")
        torch.save(output_state_dict, savepath)

    # 评估结果汇总
    metric_logger.synchronize_between_processes()
    print("Averaged stats (仅person类):", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
        panoptic_res = panoptic_evaluator.summarize()
    else:
        panoptic_res = None

    # 整理返回 stats
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None and 'bbox' in postprocessors.keys():
        stats['coco_eval_person_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_person_all'] = panoptic_res["All"]
        stats['PQ_person_things'] = panoptic_res["Things"]

    return stats, coco_evaluator