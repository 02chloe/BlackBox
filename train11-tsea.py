# /opt/data/private/BlackBox/train13-tsea_v2.py
# 集成 T-SEA 三大机制：CDA + PatchCutout + ShakeDrop
# 保留 BlackBox 原始逻辑、可视化、保存与日志结构

import os
import math
import random
import torch
import logging
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
from loss import BlackBoxLoss

# === 导入 T-SEA 模块 ===
from augment.cda_transformer import DataTransformer
from augment.patch_cutout import PatchCutout
from augment.shakedrop import ShakeDrop

# -----------------------
# Config
# -----------------------
ROOT = "/opt/data/private/BlackBox"
DATA_ROOT = os.path.join(ROOT, "data", "INRIAPerson")
SAVE_DIR = os.path.join(ROOT, "save-tt", "demo")
FINAL_PATCH_DIR = os.path.join(ROOT, "save-tt", "final_patch")
LOG_PATH = os.path.join(ROOT, "save-tt", "train.log")
VISUAL_DIR = os.path.join(ROOT, "save-tt", "visual")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_PATCH_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------
# Training Params
# -----------------------
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 4
INIT_LR = 0.005
DECAY_EPOCH = int(NUM_EPOCHS * 0.5)
DECAY_FACTOR = 0.1

PATCH_SIDE = 300
PATCH_RATIO = 0.3
PATCH_INIT_STD = 0.1
MIN_PATCH_PX = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_INPUT_H, MODEL_INPUT_W = 640, 640
TARGET_CLASS_IDX = 1
SCORE_THRESH = 0.8
FALLBACK_TO_TOP = True
FALLBACK_SCORE_THRESH = 0.2
IOU_NMS_THRESH = 0.3
MIN_BOX_SIDE = 20

DETECTION_WEIGHT = 1.0
TV_WEIGHT = 1e-3

USE_EOT = False
EOT_NUM_SAMPLES = 5
TRANS_ROT_ANGLE = (-5.0, 5.0)
TRANS_BRIGHTNESS = (0.9, 1.1)
TRANS_SCALE = (0.9, 1.1)

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

def paste_patch_via_mask(base_img, patch_tensor, patch_mask, center_xy):
    if patch_tensor.dim() == 4: p = patch_tensor[0]
    else: p = patch_tensor
    if patch_mask.dim() == 4: m = patch_mask[0]
    else: m = patch_mask
    ph, pw = p.shape[1], p.shape[2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0, y0 = cx - pw // 2, cy - ph // 2
    H, W = base_img.shape[1], base_img.shape[2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
    p_cropped = p[:, :y1 - y0, :x1 - x0]
    m_cropped = m[:, :y1 - y0, :x1 - x0]
    base_img[:, y0:y1, x0:x1] = base_img[:, y0:y1, x0:x1] * (1 - m_cropped) + p_cropped * m_cropped
    return base_img

def transform_single_patch(patch_tensor):
    p = patch_tensor[0] if patch_tensor.dim() == 4 else patch_tensor
    orig_h, orig_w = p.shape[-2], p.shape[-1]
    angle = random.uniform(*TRANS_ROT_ANGLE)
    p = TF.rotate(p, angle, interpolation=InterpolationMode.BILINEAR)
    bright = random.uniform(*TRANS_BRIGHTNESS)
    p = TF.adjust_brightness(p, bright)
    scale = random.uniform(*TRANS_SCALE)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    p = interpolate(p.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
    mask = torch.ones(1, new_h, new_w, device=p.device)
    return p.unsqueeze(0), mask.unsqueeze(0)

def eot_transform_patch_once(patch_tensor):
    if not USE_EOT or EOT_NUM_SAMPLES <= 1:
        return transform_single_patch(patch_tensor)
    else:
        p_list, m_list = [], []
        for _ in range(EOT_NUM_SAMPLES):
            p, m = transform_single_patch(patch_tensor)
            p_list.append(p); m_list.append(m)
        return torch.mean(torch.stack(p_list), 0), (torch.mean(torch.stack(m_list), 0) > 0.5).float()

# -----------------------
# ShakeDrop 注入函数
# -----------------------
def inject_shakedrop_to_backbone(model, p_max=0.1):
    """
    正确注入ShakeDrop：仅在残差Block输出（F(x)+x）之后，
    不改变Conv/BN/ReLU层内部结构。
    """
    backbone = model.backbone[0]
    layers = [
        backbone.body['layer1'],
        backbone.body['layer2'],
        backbone.body['layer3'],
        backbone.body['layer4']
    ]
    total_blocks = sum(len(layer) for layer in layers)
    block_id = 0

    for layer in layers:
        for name, block in layer.named_children():
            block_id += 1
            p_drop = ((block_id / total_blocks) ** 2) * p_max  # 平滑增长
            shake = ShakeDrop(p_drop=p_drop)
            # ✅ 在整个ResidualBlock后添加ShakeDrop
            block.add_module("shake", shake)
    return model


# -----------------------
# Plotting Helper (原版保持)
# -----------------------
def plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, save_dir, dataloader):
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(step_history, [l[0] for l in loss_history], label='Total Loss', color='blue')
    plt.xlabel('Step'); plt.ylabel('Total Loss'); plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(step_history, [l[1] for l in loss_history], label='Detection Loss', color='orange')
    plt.plot(step_history, [l[2] for l in loss_history], label='TV Loss', color='green')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(step_history, grad_norm_history, label='Grad Norm', color='red')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(step_history, lr_history, label='Learning Rate', color='purple')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存：{save_path}")

# -----------------------
# Main
# -----------------------
def main():
    dataloader = get_inria_dataloader(DATA_ROOT, split="Train", batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORKERS, disable_random_aug=True)
    logger.info(f"训练数据集大小: {len(dataloader.dataset)}")

    # === 加载 DETR 模型并注入 ShakeDrop ===
    model = load_detr_r50(device=DEVICE)
    model = inject_shakedrop_to_backbone(model, p_max=0.1)
    logger.info("✅ ShakeDrop 已注入 DETR backbone (p_max=0.1)")

    model.eval()
    for m in model.modules():
        if isinstance(m, ShakeDrop):
            m.eval()    # 禁用随机丢弃，固定为恒等映射
    for p in model.parameters(): p.requires_grad = False

    tmm = TransformerMaskingMatrix(num_enc_layers=6, num_dec_layers=6, p_base=0.2,
                                   sampling_strategy='categorical', device=DEVICE)
    tmm.register_hooks(model)
    gse = GradientSelfEnsemble(model=model, device=DEVICE)
    loss_fn = BlackBoxLoss(gse=gse, target_class=TARGET_CLASS_IDX,
                           detection_weight=DETECTION_WEIGHT, tv_weight=TV_WEIGHT, device=DEVICE)

    patch = torch.randn(1, 3, PATCH_SIDE, PATCH_SIDE, device=DEVICE) * PATCH_INIT_STD + 0.5
    patch = patch.clamp(0, 1).requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=INIT_LR)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: DECAY_FACTOR if e >= DECAY_EPOCH else 1.0)

    cda = DataTransformer(device=DEVICE)
    cutout = PatchCutout(phi_c=0.3)
    CDA_PROB = 0.5

    step_history, loss_history, grad_norm_history, lr_history = [], [], [], []
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.eval(); tmm.reset_grad_history()
        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(DEVICE).clamp(0, 1)
            imgs = cda(imgs, p_aug=CDA_PROB)

            tmm.remove_hooks()
            with torch.no_grad(): det_out = model(NestedTensor(imgs))
            tmm.register_hooks(model)

            # === 框选取 ===
            batch_boxes_all = []
            for bi in range(imgs.shape[0]):
                logits = det_out['pred_logits'][bi]; boxes = det_out['pred_boxes'][bi]
                probs = torch.softmax(logits, dim=-1); cls_scores = probs[..., TARGET_CLASS_IDX]
                keep_idx = (cls_scores > SCORE_THRESH).nonzero(as_tuple=False).squeeze(1)
                if keep_idx.numel() == 0 and FALLBACK_TO_TOP:
                    top_score, top_idx = torch.max(cls_scores, dim=0)
                    if top_score.item() >= FALLBACK_SCORE_THRESH: keep_idx = top_idx.unsqueeze(0)
                if keep_idx.numel() == 0: batch_boxes_all.append(torch.empty((0, 4))); continue
                sel_boxes = boxes[keep_idx]
                sel_xyxy = detr_boxes_to_xyxy_pixel(sel_boxes.detach().cpu())
                widths, heights = sel_xyxy[:, 2]-sel_xyxy[:, 0], sel_xyxy[:, 3]-sel_xyxy[:, 1]
                large_mask = (widths >= MIN_BOX_SIDE) & (heights >= MIN_BOX_SIDE)
                if large_mask.sum() == 0: batch_boxes_all.append(torch.empty((0, 4))); continue
                sel_xyxy = sel_xyxy[large_mask]
                keep_nms = nms(sel_xyxy, cls_scores[keep_idx].detach().cpu()[large_mask], IOU_NMS_THRESH)
                batch_boxes_all.append(sel_xyxy[keep_nms])

            # === 构建带补丁图像 ===
            patched = imgs.clone()
            for bi in range(imgs.shape[0]):
                sel_boxes_cpu = batch_boxes_all[bi]
                if sel_boxes_cpu.numel() == 0: continue
                for box in sel_boxes_cpu:
                    xmin, ymin, xmax, ymax = box.tolist()
                    side = max(MIN_PATCH_PX, int(round(min(xmax-xmin, ymax-ymin) * PATCH_RATIO)))
                    transformed_patch, patch_mask = eot_transform_patch_once(patch)
                    patch_resized = interpolate(transformed_patch, size=(side, side), mode='bilinear', align_corners=False)
                    mask_resized = interpolate(patch_mask, size=(side, side), mode='bilinear', align_corners=False)
                    patched[bi] = paste_patch_via_mask(patched[bi], patch_resized, (mask_resized > 0.5).float(),
                                                       ((xmin+xmax)/2, (ymin+ymax)/2))

            # === Patch Cutout ===
            patch = cutout(patch.detach()); patch.requires_grad_(True)

            # === Loss & Backprop ===
            loss_dict = loss_fn(imgs, patched, patch_tensor=patch)
            total_loss = loss_dict['total_loss']
            optimizer.zero_grad(); total_loss.backward()
            grad_norm = patch.grad.detach().cpu().norm().item() if patch.grad is not None else None
            optimizer.step(); patch.data.clamp_(0, 1)

            # === 5. logging & visualization bookkeeping（严格还原老版本命名） ===
            if batch_idx % 10 == 0:
                step_history.append(global_step)
                det_loss_v = loss_dict.get('det_loss', 0.0)
                tv_loss_v  = loss_dict.get('tv_loss',  0.0)
                det_loss_v = det_loss_v.item() if isinstance(det_loss_v, torch.Tensor) else float(det_loss_v)
                tv_loss_v  = tv_loss_v.item()  if isinstance(tv_loss_v,  torch.Tensor) else float(tv_loss_v)
                loss_history.append((total_loss.item(), det_loss_v, tv_loss_v))
                grad_norm_history.append(grad_norm)
                lr_history.append(optimizer.param_groups[0]['lr'])

            # === 6. info log（保持原样式） ===
            logger.info(f"[epoch {epoch+1}/{NUM_EPOCHS} batch {batch_idx}] total_loss={total_loss.item():.6f}")

            # === 7. save visual examples（严格按老版本：*_orig_with_boxes / *_patched_with_boxes / *_patch） ===
            if global_step % 20 == 0:
                orig_with_boxes    = draw_boxes_on_tensor(detach_cpu(imgs[0]),    batch_boxes_all[0])
                patched_with_boxes = draw_boxes_on_tensor(detach_cpu(patched[0]), batch_boxes_all[0])
                save_image(orig_with_boxes,    os.path.join(SAVE_DIR, f"step_{global_step}_orig_with_boxes.png"))
                save_image(patched_with_boxes, os.path.join(SAVE_DIR, f"step_{global_step}_patched_with_boxes.png"))
                save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"step_{global_step}_patch.png"))

            global_step += 1

        # === epoch end（严格按老版本：先 scheduler.step，再 plot，再保存 epoch_*_patch） ===
        scheduler.step()
        if step_history and loss_history:
            plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, VISUAL_DIR, dataloader)

        save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.png"))
        torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.pt"))
        logger.info(
            f"Epoch {epoch+1} 保存Patch快照 | 当前学习率={optimizer.param_groups[0]['lr']:.6f} | "
            f"距离学习率衰减剩余epoch: {max(0, DECAY_EPOCH - epoch)}"
        )

    # === final: plotting & cleanup（严格按老版本：重复 plot、移除 hook、双路径保存、双日志） ===
    if step_history and loss_history:
        plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, VISUAL_DIR, dataloader)

    tmm.remove_hooks()

    # 1) 保存到 SAVE_DIR
    save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.pt"))

    # 2) 另存到 FINAL_PATCH_DIR
    save_image(patch[0].detach().cpu(), os.path.join(FINAL_PATCH_DIR, "final_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(FINAL_PATCH_DIR, "final_patch.pt"))

    logger.info(f"训练完成！最终Patch保存至: {SAVE_DIR}")
    logger.info(f"Final patch已单独保存至专用目录: {FINAL_PATCH_DIR}")

if __name__ == "__main__":
    main()
