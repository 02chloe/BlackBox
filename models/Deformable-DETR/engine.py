# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import json
from pathlib import Path
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    """
    评估阶段：
    - 继续正常的 COCO 指标计算；
    - 同时在迭代每个 batch 时，把boxes/labels/scores 直接写入 JSONL（流式，避免中途状态丢失）；
    - 循环结束后，再把 JSONL 合并为标准 JSON / PTH（res.*）。
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # --- 文件准备
    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "detections_stream.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()  # 清空旧文件

    # 内存缓冲（最终合并用）
    detections_buffer = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # === DEBUG 输出（观察每个 batch 的产出规模）===
        num_per_img = [int(o["scores"].numel()) for o in results]
        print(f"[DEBUG][engine] batch detections per image = {num_per_img}, batch_total={sum(num_per_img)}")

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if res:
            _fid = list(res.keys())[0]
            _o = res[_fid]
            print(f"[DEBUG][engine] first image_id={_fid}, keys={list(_o.keys())}, n_scores={_o['scores'].numel()}")
        else:
            print(f"[DEBUG][engine] res is EMPTY dict")

        # === 实时落盘（每个 batch 写一段 JSONL，避免中途 evaluator 丢状态）===
        batch_dets = []
        for target, output in zip(targets, results):
            image_id = int(target['image_id'].item())
            boxes  = output['boxes'].cpu().tolist()   # xywh
            scores = output['scores'].cpu().tolist()
            labels = output['labels'].cpu().tolist()
            for b, s, c in zip(boxes, scores, labels):
                x, y, w, h = b
                det = {
                    "image_id": image_id,
                    "category_id": int(c),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(s)
                }
                batch_dets.append(det)

        if batch_dets:
            # 追加写入
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for det in batch_dets:
                    f.write(json.dumps(det, ensure_ascii=False) + "\n")
            # 放入内存缓冲（最终合并）
            detections_buffer.extend(batch_dets)
            print(f"[DEBUG][export] wrote {len(batch_dets)} detections for batch to {jsonl_path}")

        # --- COCO evaluator 正常更新
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            # 若启用了 panoptic，但未启用 segm，这里 target_sizes 可能未定义。按需自行处理。
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # === 合并导出：把 JSONL/内存缓冲输出为最终 JSON / PTH ===
    try:
        total = len(detections_buffer)
        if total == 0 and jsonl_path.exists():
            # 极端情况下，从 JSONL 读回
            with open(jsonl_path, "r", encoding="utf-8") as f:
                detections_buffer = [json.loads(line) for line in f]
            total = len(detections_buffer)

        if total > 0:
            final_json = out_dir / "res.json"
            final_pth  = out_dir / "res.pth"
            with open(final_json, "w", encoding="utf-8") as f:
                json.dump(detections_buffer, f, indent=2, ensure_ascii=False)
            torch.save(detections_buffer, final_pth)
            print(f"✅ Exported {total} detections to:")
            print(f"   JSON: {final_json}")
            print(f"   PTH : {final_pth}")
        else:
            print("⚠️ No detections collected. Nothing to export.")
    except Exception as e:
        print(f"❌ Export error: {e}")

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
