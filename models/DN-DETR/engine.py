"""
Train and eval functions used in main.py
"""

import math
import os
import sys
import json
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
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                dn_args = (targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
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
            print(f"Loss is {loss_value}, stopping training")
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
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
        score_threshold = 0.2  # person置信度阈值
    except:
        need_tgt_for_training = False
        score_threshold = 0.2

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # 输出目录（保留版本一的JSON保存功能）
    save_img_dir = None
    if args and args.save_results:
        save_img_dir = os.path.join(output_dir, "res")
        os.makedirs(save_img_dir, exist_ok=True)
        print(f"仅person类检测框图片保存至：{save_img_dir}")

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}
    json_results_all = []  # 保留版本一的JSON汇总功能

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        # 推理（核心修复：恢复版本二的dn_args传递逻辑 + 修复autocast警告）
        with torch.amp.autocast('cuda', enabled=args.amp):
            if need_tgt_for_training:
                outputs, _ = model(samples, dn_args=args.num_patterns)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)

        # 后续的损失计算、person类过滤、JSON保存、图片绘制等逻辑均保留版本一的代码，无需修改
        weight_dict = criterion.weight_dict
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # ===== 以下均为版本一的原有代码，无需修改 =====
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        raw_results = postprocessors['bbox'](outputs, orig_target_sizes)
        person_results = []
        for res in raw_results:
            mask = (res['labels'] == 1) & (res['scores'] > score_threshold)
            filtered_res = {
                'boxes': res['boxes'][mask],
                'scores': res['scores'][mask],
                'labels': res['labels'][mask],
            }
            person_results.append(filtered_res)
        results = person_results

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if args.save_results:
            for i, (tgt, res) in enumerate(zip(targets, results)):
                image_id = tgt["image_id"].item()
                img_info = base_ds.loadImgs(image_id)[0]
                img_filename = img_info["file_name"]

                gt_mask = tgt["labels"] == 1
                gt_bbox = tgt["boxes"][gt_mask]
                gt_label = tgt["labels"][gt_mask]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1) if len(gt_bbox) else torch.empty((0, 5))

                res_bbox = res["boxes"]
                res_prob = res["scores"]
                res_label = res["labels"]
                res_info = torch.cat((res_bbox, res_prob.unsqueeze(-1), res_label.unsqueeze(-1)), 1) if len(res_bbox) else torch.empty((0, 6))

                output_state_dict.setdefault("gt_info", []).append(gt_info.cpu())
                output_state_dict.setdefault("res_info", []).append(res_info.cpu())

                # 保留JSON汇总功能
                for box, score, label in zip(res_bbox.cpu().numpy(), res_prob.cpu().numpy(), res_label.cpu().numpy()):
                    x1, y1, x2, y2 = box.tolist()
                    w, h = x2 - x1, y2 - y1
                    json_results_all.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

                # 保留图片绘制功能
                img_path = os.path.join(args.coco_path, 'test2017', img_filename)
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except FileNotFoundError:
                    continue

                for box, score in zip(res_bbox.cpu().numpy(), res_prob.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"person: {score:.2f}"
                    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img_cv, (x1, y1 - th - 5), (x1 + tw, y1 - 5), (0, 255, 0), -1)
                    cv2.putText(img_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                save_img_path = os.path.join(save_img_dir, img_filename)
                cv2.imwrite(save_img_path, img_cv)

        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            break

    # 保留版本一的PKL和JSON保存功能
    if args.save_results:
        import os.path as osp
        pkl_path = osp.join(args.output_dir, f"results_person_only-{utils.get_rank()}.pkl")
        torch.save(output_state_dict, pkl_path)
        print(f"✅ 保存仅person类数值结果至: {pkl_path}")

        json_path = osp.join(args.output_dir, "res.json")
        with open(json_path, "w") as f:
            json.dump(json_results_all, f, indent=2)
        print(f"✅ 保存person检测框汇总 JSON 至: {json_path}  (共 {len(json_results_all)} 条记录)")

    # 评估汇总（保留原逻辑）
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

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None and 'bbox' in postprocessors.keys():
        stats['coco_eval_person_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_person_all'] = panoptic_res["All"]
        stats['PQ_person_things'] = panoptic_res["Things"]

    return stats, coco_evaluator



