# /opt/data/private/BlackBox/train_t13_.py  (train_t13-kl.py)
# 核心修改：
#   - 通过 --version 控制保存目录（原有）
#   - 新增命令行参数 --beta 控制 KL 一致性损失权重
#   - 新增命令行参数 --p_base 控制 TMM 的基础屏蔽概率

import os
import random
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.ops import box_convert, nms
from torch.nn.functional import interpolate

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch.optim.lr_scheduler as lr_scheduler

from inria_dataloader import get_inria_dataloader
from utils.load_model import load_detr_r50
from tmm import TransformerMaskingMatrix, NestedTensor
from gse import GradientSelfEnsemble
from loss_kl_ import BlackBoxLoss

# -----------------------
# 固定不变的根与数据根
# -----------------------
ROOT = "/opt/data/private/BlackBox"
DATA_ROOT = os.path.join(ROOT, "data", "INRIAPerson")

# 这些路径将由 main() 在解析 --version 后设置为全局变量
SAVE_DIR = None
FINAL_PATCH_DIR = None
LOG_PATH = None
VISUAL_DIR = None

# matplotlib config (no-GUI)
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

logger = logging.getLogger(__name__)

# training params
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 4
INIT_LR = 0.005
DECAY_EPOCH = int(NUM_EPOCHS * 0.5)
DECAY_FACTOR = 0.1

# patch params
PATCH_SIDE = 300
PATCH_RATIO = 0.5
PATCH_INIT_STD = 0.1
MIN_PATCH_PX = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model / detection params
MODEL_INPUT_H, MODEL_INPUT_W = 640, 640
TARGET_CLASS_IDX = 1
SCORE_THRESH = 0.8
FALLBACK_TO_TOP = True
FALLBACK_SCORE_THRESH = 0.2
IOU_NMS_THRESH = 0.3
MIN_BOX_SIDE = 20

# loss weights
DETECTION_WEIGHT = 1.0
TV_WEIGHT = 1e-3
NPS_WEIGHT = 0.0  # 当前未使用

# EoT params
USE_EOT = False
EOT_NUM_SAMPLES = 5

# transformation ranges
TRANS_ROT_ANGLE = (-5.0, 5.0)
TRANS_BRIGHTNESS = (0.9, 1.1)
TRANS_SCALE = (0.9, 1.1)

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Helpers
# -----------------------
def detach_cpu(img: torch.Tensor):
    return img.detach().cpu().clamp(0, 1)

def draw_boxes_on_tensor(img_tensor: torch.Tensor, boxes_xyxy_cpu: torch.Tensor):
    if boxes_xyxy_cpu is None or boxes_xyxy_cpu.numel() == 0:
        return img_tensor
    img_uint8 = (img_tensor * 255).byte()
    boxes = boxes_xyxy_cpu.clone()
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if boxes.shape[0] == 0:
        return img_tensor
    boxes_int = boxes.to(torch.int64)
    img_boxes = draw_bounding_boxes(img_uint8, boxes=boxes_int, colors="red", width=2)
    return img_boxes.float() / 255.0

def detr_boxes_to_xyxy_pixel(pred_boxes):
    pb = pred_boxes.clone()
    if pb.max() <= 1.01:
        pb[:, 0] = pb[:, 0] * MODEL_INPUT_W
        pb[:, 1] = pb[:, 1] * MODEL_INPUT_H
        pb[:, 2] = pb[:, 2] * MODEL_INPUT_W
        pb[:, 3] = pb[:, 3] * MODEL_INPUT_H
    return box_convert(pb, in_fmt='cxcywh', out_fmt='xyxy').cpu()

def paste_patch_via_mask(base_img: torch.Tensor, patch_tensor: torch.Tensor, patch_mask: torch.Tensor, center_xy: tuple):
    if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
        p = patch_tensor[0]
    elif patch_tensor.dim() == 3:
        p = patch_tensor
    else:
        raise ValueError("Patch形状无效：需为[1,3,ph,pw]或[3,ph,pw]")

    if patch_mask.dim() == 4 and patch_mask.shape[0] == 1:
        m = patch_mask[0]
    elif patch_mask.dim() == 3 and patch_mask.shape[0] == 1:
        m = patch_mask
    else:
        raise ValueError("掩码形状无效：需为[1,1,ph,pw]或[1,ph,pw]")

    ph, pw = p.shape[1], p.shape[2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = cx - pw // 2
    y0 = cy - ph // 2

    H, W = base_img.shape[1], base_img.shape[2]
    src_x0, src_y0 = 0, 0
    dst_x0, dst_y0 = x0, y0
    dst_x1, dst_y1 = x0 + pw, y0 + ph

    if dst_x0 < 0:
        src_x0 = -dst_x0
        dst_x0 = 0
    if dst_y0 < 0:
        src_y0 = -dst_y0
        dst_y0 = 0
    if dst_x1 > W:
        dst_x1 = W
    if dst_y1 > H:
        dst_y1 = H

    out_w = dst_x1 - dst_x0
    out_h = dst_y1 - dst_y0
    if out_w <= 0 or out_h <= 0:
        return base_img.clone()

    src_x1 = src_x0 + out_w
    src_y1 = src_y0 + out_h
    p_cropped = p[:, src_y0:src_y1, src_x0:src_x1]
    m_cropped = m[:, src_y0:src_y1, src_x0:src_x1]

    mask = torch.zeros_like(base_img)
    mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = m_cropped
    padded_patch = torch.zeros_like(base_img)
    padded_patch[:, dst_y0:dst_y1, dst_x0:dst_x1] = p_cropped
    return base_img * (1.0 - mask) + padded_patch * mask

# -----------------------
# Patch Transformation
# -----------------------
def transform_single_patch(patch_tensor: torch.Tensor):
    own_batch_dim = False
    if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
        p = patch_tensor[0]
        own_batch_dim = True
    elif patch_tensor.dim() == 3:
        p = patch_tensor
    else:
        raise ValueError("patch_tensor must be [1,3,H,W] or [3,H,W]")

    p = p.to(device=patch_tensor.device, dtype=patch_tensor.dtype)
    orig_h, orig_w = p.shape[-2], p.shape[-1]

    angle = random.uniform(*TRANS_ROT_ANGLE)
    p_rotated = TF.rotate(p, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=True)
    mask_orig = torch.ones(1, orig_h, orig_w, device=p.device, dtype=p.dtype)
    mask_rotated = TF.rotate(mask_orig, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=True)
    p = interpolate(p_rotated.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)[0]
    mask = interpolate(mask_rotated.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)[0]
    mask = (mask > 0.5).float()

    bright = random.uniform(*TRANS_BRIGHTNESS)
    p = TF.adjust_brightness(p, bright)

    scale = random.uniform(*TRANS_SCALE)
    new_h = max(1, int(round(orig_h * scale)))
    new_w = max(1, int(round(orig_w * scale)))
    if new_h != orig_h or new_w != orig_w:
        p = interpolate(p.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        p = interpolate(p.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)[0]
        mask = interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        mask = interpolate(mask.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False)[0]
        mask = (mask > 0.5).float()

    if own_batch_dim:
        return p.unsqueeze(0), mask.unsqueeze(0)
    else:
        return p, mask

def eot_transform_patch_once(patch_tensor: torch.Tensor):
    if not USE_EOT or EOT_NUM_SAMPLES <= 1:
        return transform_single_patch(patch_tensor)
    else:
        patched_list = []
        mask_list = []
        for _ in range(EOT_NUM_SAMPLES):
            p, m = transform_single_patch(patch_tensor)
            patched_list.append(p)
            mask_list.append(m)
        stacked_p = torch.stack(patched_list, dim=0)
        stacked_m = torch.stack(mask_list, dim=0)
        mean_p = torch.mean(stacked_p, dim=0)
        mean_m = torch.mean(stacked_m, dim=0)
        mean_m = (mean_m > 0.5).float()
        return mean_p, mean_m

# -----------------------
# Plotting helper
# -----------------------
def plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, save_dir, dataloader):
    assert len(step_history) == len(loss_history), \
        f"数据长度不匹配！step_history={len(step_history)}, loss_history={len(loss_history)}"
    valid_data = []
    valid_steps = []
    for step, loss in zip(step_history, loss_history):
        if len(loss) < 3:
            logger.warning(f"跳过无效loss数据：{loss}")
            continue
        total_loss, det_loss, tv_loss = loss[:3]
        if not (np.isfinite(total_loss) and np.isfinite(det_loss) and np.isfinite(tv_loss)):
            logger.warning(f"跳过无效数值：total_loss={total_loss}")
            continue
        valid_steps.append(step)
        valid_data.append(loss)
    if not valid_steps:
        logger.error("无有效训练数据，无法绘制曲线")
        return

    plt.subplot(4, 1, 1)
    plt.plot(valid_steps, [l[0] for l in valid_data],
             label='Total Loss', linewidth=3, alpha=0.8)
    plt.xlabel('Global Step'); plt.ylabel('Total Loss Value')
    plt.title('Training Total Loss Curve (Exclusive Plot)')
    plt.legend(loc='upper right'); plt.grid(alpha=0.5); plt.tight_layout()

    plt.subplot(4, 1, 2)
    plt.plot(valid_steps, [l[1] for l in valid_data], label='Detection Loss', linewidth=2)
    plt.plot(valid_steps, [l[2] for l in valid_data], label='TV Loss', linewidth=2)
    if len(valid_data[0]) >= 4:
        plt.plot(valid_steps, [l[3] for l in valid_data], label='Consistency Loss (KL)', linestyle='--', linewidth=2)
    plt.xlabel('Global Step'); plt.ylabel('Loss Value')
    plt.title('Training Detection / TV / Consistency Loss Curves')
    plt.legend(); plt.grid(alpha=0.5); plt.tight_layout()

    plt.subplot(4, 1, 3)
    valid_grad = [(s, g) for s, g in zip(step_history, grad_norm_history) if g is not None and np.isfinite(g)]
    if valid_grad:
        s_valid, g_valid = zip(*valid_grad)
        plt.plot(s_valid, g_valid, linewidth=2)
    plt.xlabel('Global Step'); plt.ylabel('Gradient Norm')
    plt.title('Patch Gradient Norm Curve')
    plt.grid(alpha=0.5); plt.tight_layout()

    plt.subplot(4, 1, 4)
    plt.plot(step_history, lr_history, linewidth=2)
    if step_history and DECAY_EPOCH <= NUM_EPOCHS:
        decay_step = DECAY_EPOCH * len(dataloader)
        if decay_step <= max(step_history):
            plt.axvline(x=decay_step, color='red', linestyle='--', alpha=0.7, label=f'Decay (epoch={DECAY_EPOCH})')
            plt.legend()
    plt.xlabel('Global Step'); plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (BlackBox Paper)')
    plt.grid(alpha=0.3); plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练可视化图表已保存至: {save_path} | 有效数据点：{len(valid_steps)}")

# -----------------------
# main
# -----------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        required=True,
        help="版本号（用于目录名），例如：kl0.0008 -> 路径使用 save-kl0.0008"
    )
    parser.add_argument(
        "--root",
        default=ROOT,
        help="项目根目录（默认 /opt/data/private/BlackBox）"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.001,
        help="KL 一致性损失权重 beta（默认 0.001）"
    )
    parser.add_argument(
        "--p_base",
        type=float,
        default=0.1,
        help="TMM 模块基础屏蔽概率 p_base（默认 0.1）"
    )
    args = parser.parse_args()

    # 根据 --version 组装保存目录前缀：save-<version>
    save_tag = f"save-{args.version}"
    root = args.root

    # 把路径注入到全局变量
    global SAVE_DIR, FINAL_PATCH_DIR, LOG_PATH, VISUAL_DIR
    SAVE_DIR = os.path.join(root, save_tag, "demo")
    FINAL_PATCH_DIR = os.path.join(root, save_tag, "final_patch")
    LOG_PATH = os.path.join(root, save_tag, "train.log")
    VISUAL_DIR = os.path.join(root, save_tag, "visual")

    # 确保目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FINAL_PATCH_DIR, exist_ok=True)
    os.makedirs(VISUAL_DIR, exist_ok=True)

    # 重新配置 logging
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )

    logger.info(f"ROOT={root}")
    logger.info(f"VERSION={args.version} -> save_tag={save_tag}")
    logger.info(f"SAVE_DIR={SAVE_DIR}")
    logger.info(f"FINAL_PATCH_DIR={FINAL_PATCH_DIR}")
    logger.info(f"VISUAL_DIR={VISUAL_DIR}")
    logger.info(f"LOG_PATH={LOG_PATH}")
    logger.info(f"超参数: KL beta={args.beta}, TMM p_base={args.p_base}")

    # dataloader
    dataloader = get_inria_dataloader(
        DATA_ROOT, split="Train", batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, disable_random_aug=True
    )
    logger.info(f"训练数据集大小: {len(dataloader.dataset)}")
    logger.info(f"学习率策略：初始LR={INIT_LR}，第{DECAY_EPOCH}个epoch后衰减至{INIT_LR*DECAY_FACTOR}（BlackBox论文）")
    logger.info(f"Patch transforms: rotate{TRANS_ROT_ANGLE} brightness{TRANS_BRIGHTNESS} scale{TRANS_SCALE} USE_EOT={USE_EOT}")

    # model
    model = load_detr_r50(device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # TMM
    tmm = TransformerMaskingMatrix(
        num_enc_layers=6, num_dec_layers=6,
        p_base=args.p_base,
        sampling_strategy='categorical',
        device=DEVICE
    )
    tmm.register_hooks(model)
    tmm.reset_grad_history()

    # GSE + loss
    gse = GradientSelfEnsemble(model=model, device=DEVICE)
    loss_fn = BlackBoxLoss(
        gse=gse,
        target_class=TARGET_CLASS_IDX,
        detection_weight=DETECTION_WEIGHT,
        tv_weight=TV_WEIGHT,
        l2_weight=0.0,
        layer_aggregation='per_layer_loss',
        use_sigmoid_for_binary=False,
        device=DEVICE,
        kl_beta=args.beta,   # ✅ 外部可控的 KL 权重
        kl_tau=2.0           # 如需外部控制，可以再加一个命令行参数
    )

    # patch init
    patch = torch.randn(1, 3, PATCH_SIDE, PATCH_SIDE, device=DEVICE) * PATCH_INIT_STD + 0.5
    patch = patch.clamp(0.0, 1.0)
    patch.requires_grad_(True)

    # optimizer / scheduler
    optimizer = torch.optim.Adam([patch], lr=INIT_LR)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: DECAY_FACTOR if epoch >= DECAY_EPOCH else 1.0
    )

    # logging lists
    step_history, loss_history, grad_norm_history, lr_history = [], [], [], []

    logger.info(f"开始训练（PATCH_RATIO={PATCH_RATIO}），结果保存至: {SAVE_DIR}")
    logger.info(f"Final patch将单独保存至: {FINAL_PATCH_DIR}")

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.eval()
        tmm.reset_grad_history()

        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(DEVICE).clamp(0, 1)
            B = imgs.shape[0]

            # 1. Clean detection
            tmm.remove_hooks()
            with torch.no_grad():
                try:
                    det_out = model(imgs)
                except Exception:
                    det_out = model(NestedTensor(imgs))
            tmm.register_hooks(model)

            # 2. Select boxes per image
            batch_boxes_all = []
            for bi in range(B):
                logits = det_out['pred_logits'][bi]
                boxes = det_out['pred_boxes'][bi]
                probs = torch.softmax(logits, dim=-1)
                cls_scores = probs[..., TARGET_CLASS_IDX]

                keep_idx = (cls_scores > SCORE_THRESH).nonzero(as_tuple=False).squeeze(1) \
                    if (cls_scores > SCORE_THRESH).any() else torch.tensor([], dtype=torch.long, device=cls_scores.device)
                if keep_idx.numel() == 0 and FALLBACK_TO_TOP:
                    top_score, top_idx = torch.max(cls_scores, dim=0)
                    if top_score.item() >= FALLBACK_SCORE_THRESH:
                        keep_idx = top_idx.unsqueeze(0)
                    else:
                        keep_idx = torch.tensor([], dtype=torch.long, device=cls_scores.device)

                if keep_idx.numel() == 0:
                    batch_boxes_all.append(torch.empty((0, 4), dtype=torch.float32))
                    continue

                sel_boxes = boxes[keep_idx]
                sel_xyxy = detr_boxes_to_xyxy_pixel(sel_boxes.detach().cpu())
                widths = sel_xyxy[:, 2] - sel_xyxy[:, 0]
                heights = sel_xyxy[:, 3] - sel_xyxy[:, 1]
                large_mask = (widths >= MIN_BOX_SIDE) & (heights >= MIN_BOX_SIDE)
                if large_mask.sum() == 0:
                    batch_boxes_all.append(torch.empty((0, 4), dtype=torch.float32))
                    continue
                sel_xyxy = sel_xyxy[large_mask]
                sel_scores_cpu = cls_scores[keep_idx].detach().cpu()[large_mask]
                keep_nms = nms(sel_xyxy, sel_scores_cpu, IOU_NMS_THRESH)
                sel_xyxy_nms = sel_xyxy[keep_nms]
                batch_boxes_all.append(sel_xyxy_nms)

            # 3. Build patched images
            patched = imgs.clone()
            patch_sizes = []
            for bi in range(B):
                sel_boxes_cpu = batch_boxes_all[bi]
                if sel_boxes_cpu.numel() == 0:
                    continue
                sel_boxes_dev = sel_boxes_cpu.to(DEVICE)
                for box in sel_boxes_dev:
                    xmin, ymin, xmax, ymax = box.tolist()
                    box_w = max(int(xmax - xmin), 1)
                    box_h = max(int(ymax - ymin), 1)
                    short = min(box_w, box_h)
                    side = max(MIN_PATCH_PX, int(round(short * PATCH_RATIO)))
                    patch_sizes.append(side)

                    transformed_patch, patch_mask = eot_transform_patch_once(patch)
                    patch_resized = interpolate(transformed_patch, size=(side, side),
                                               mode='bilinear', align_corners=False)
                    mask_resized = interpolate(patch_mask, size=(side, side),
                                               mode='bilinear', align_corners=False)
                    mask_resized = (mask_resized > 0.5).float()

                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    patched[bi] = paste_patch_via_mask(patched[bi], patch_resized, mask_resized, (cx, cy))

            # 4. loss & backward
            loss_dict = loss_fn(imgs, patched, patch_tensor=patch)
            total_loss = loss_dict['total_loss']
            optimizer.zero_grad()
            total_loss.backward()

            grad_norm = patch.grad.detach().cpu().norm().item() if patch.grad is not None else None
            if grad_norm is None:
                logger.warning(f"[epoch {epoch+1} batch {batch_idx}] Patch梯度为None！")

            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            # 5. logging & visualization bookkeeping
            if batch_idx % 10 == 0:
                det_loss_v = loss_dict.get('det_loss', 0.0)
                det_loss_v = det_loss_v.item() if isinstance(det_loss_v, torch.Tensor) else float(det_loss_v)
                tv_loss_v = loss_dict.get('tv_loss', 0.0)
                tv_loss_v = tv_loss_v.item() if isinstance(tv_loss_v, torch.Tensor) else float(tv_loss_v)
                cons_loss_v = loss_dict.get('consistency_loss', 0.0)
                cons_loss_v = cons_loss_v.item() if isinstance(cons_loss_v, torch.Tensor) else float(cons_loss_v)
                current_lr = optimizer.param_groups[0]['lr']

                step_history.append(global_step)
                loss_history.append((total_loss.item(), det_loss_v, tv_loss_v, cons_loss_v))
                grad_norm_history.append(grad_norm)
                lr_history.append(current_lr)

            # 6. info log
            det_loss_v = loss_dict.get('det_loss', 0.0)
            det_loss_v = det_loss_v.item() if isinstance(det_loss_v, torch.Tensor) else float(det_loss_v)
            tv_loss_v = loss_dict.get('tv_loss', 0.0)
            tv_loss_v = tv_loss_v.item() if isinstance(tv_loss_v, torch.Tensor) else float(tv_loss_v)
            cons_loss_v = loss_dict.get('consistency_loss', 0.0)
            cons_loss_v = cons_loss_v.item() if isinstance(cons_loss_v, torch.Tensor) else float(cons_loss_v)

            avg_patch_side = np.mean(patch_sizes) if patch_sizes else 0
            logger.info(
                f"[epoch {epoch+1}/{NUM_EPOCHS} batch {batch_idx}] "
                f"total_loss={total_loss.item():.6f} | "
                f"det_loss={det_loss_v:.6f} | "
                f"tv_loss={tv_loss_v:.6f} | "
                f"consistency_loss={cons_loss_v:.6f} | "
                f"grad_norm={(grad_norm if grad_norm is not None else float('nan')):.4f} | "
                f"目标框数量={[b.shape[0] for b in batch_boxes_all]} | "
                f"平均Patch尺寸={avg_patch_side:.1f} | "
                f"当前学习率={optimizer.param_groups[0]['lr']:.6f}"
            )

            # 7. save visual examples
            if global_step % 20 == 0:
                orig_with_boxes = draw_boxes_on_tensor(detach_cpu(imgs[0]), batch_boxes_all[0])
                patched_with_boxes = draw_boxes_on_tensor(detach_cpu(patched[0]), batch_boxes_all[0])
                save_image(orig_with_boxes, os.path.join(SAVE_DIR, f"step_{global_step}_orig_with_boxes.png"))
                save_image(patched_with_boxes, os.path.join(SAVE_DIR, f"step_{global_step}_patched_with_boxes.png"))
                save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"step_{global_step}_patch.png"))
            global_step += 1

        # epoch end
        scheduler.step()
        save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.png"))
        torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.pt"))
        logger.info(
            f"Epoch {epoch+1} 保存Patch快照 | 当前学习率={optimizer.param_groups[0]['lr']:.6f} | "
            f"距离学习率衰减剩余epoch: {max(0, DECAY_EPOCH - epoch)}"
        )

    # final: 清理与最终patch保存
    tmm.remove_hooks()

    save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.pt"))

    save_image(patch[0].detach().cpu(), os.path.join(FINAL_PATCH_DIR, "final_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(FINAL_PATCH_DIR, "final_patch.pt"))

    logger.info(f"训练完成！最终Patch保存至: {SAVE_DIR}")
    logger.info(f"Final patch已单独保存至专用目录: {FINAL_PATCH_DIR}")

if __name__ == "__main__":
    main()
