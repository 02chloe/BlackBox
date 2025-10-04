# /opt/data/private/BlackBox/train.py
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
# 导入学习率调度器模块
import torch.optim.lr_scheduler as lr_scheduler

from inria_dataloader import get_inria_dataloader
from tmm import TransformerMaskingMatrix, load_detr_r50, NestedTensor
from gse import GradientSelfEnsemble
from loss import BlackBoxLoss


# -----------------------
# Config
# -----------------------
ROOT = "/opt/data/private/BlackBox"
DATA_ROOT = os.path.join(ROOT, "data", "INRIAPerson")
SAVE_DIR = os.path.join(ROOT, "save", "demo")
LOG_PATH = os.path.join(ROOT, "save", "train.log")
VISUAL_DIR = os.path.join(ROOT, "save", "visual")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# matplotlib配置
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 训练参数（NUM_EPOCHS=100，50%衰减即第50个epoch）
BATCH_SIZE = 8
NUM_EPOCHS = 100           
NUM_WORKERS = 4
INIT_LR = 0.005
DECAY_EPOCH = int(NUM_EPOCHS * 0.5)  # 第50个epoch后衰减
DECAY_FACTOR = 0.1

# Patch参数
PATCH_SIDE = 300
PATCH_RATIO = 0.5
PATCH_INIT_STD = 0.1
MIN_PATCH_PX = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型检测参数
MODEL_INPUT_H, MODEL_INPUT_W = 640, 640
TARGET_CLASS_IDX = 1
SCORE_THRESH = 0.5
FALLBACK_TO_TOP = True
FALLBACK_SCORE_THRESH = 0.2
IOU_NMS_THRESH = 0.5
MIN_BOX_SIDE = 5

# 损失权重
DETECTION_WEIGHT = 1.0
TV_WEIGHT = 1e-3
NPS_WEIGHT = 0.0

# 禁用EoT
USE_EOT = False

# 可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# -----------------------
# Helpers（核心修改：增加dataloader参数）
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

def eot_transform_patch(patch_tensor: torch.Tensor):
    return patch_tensor

def paste_patch_via_mask(base_img: torch.Tensor, patch_tensor: torch.Tensor, center_xy: tuple):
    if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
        p = patch_tensor[0]
    elif patch_tensor.dim() == 3:
        p = patch_tensor
    else:
        raise ValueError("Patch形状无效：需为[1,3,ph,pw]或[3,ph,pw]")

    ph, pw = p.shape[1], p.shape[2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = cx - pw // 2
    y0 = cy - ph // 2

    H, W = base_img.shape[1], base_img.shape[2]
    src_x0, src_y0 = 0, 0
    dst_x0, dst_y0 = x0, y0
    dst_x1, dst_y1 = x0 + pw, y0 + ph

    if dst_x0 < 0:
        src_x0 = -dst_x0; dst_x0 = 0
    if dst_y0 < 0:
        src_y0 = -dst_y0; dst_y0 = 0
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

    mask = torch.zeros_like(base_img)
    mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    padded_patch = torch.zeros_like(base_img)
    padded_patch[:, dst_y0:dst_y1, dst_x0:dst_x1] = p_cropped
    return base_img * (1.0 - mask) + padded_patch * mask

# 核心修改1：增加dataloader参数，用于计算decay_step
def plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, save_dir, dataloader):
    """
    绘制3类可视化图表：
    1. 损失曲线（total_loss、det_loss、tv_loss）
    2. 梯度范数曲线
    3. 学习率变化曲线（含衰减点标记）
    """
    # 1. 损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(step_history, [l[0] for l in loss_history], label='Total Loss', color='#1f77b4', linewidth=2)
    plt.plot(step_history, [l[1] for l in loss_history], label='Detection Loss', color='#ff7f0e', linewidth=2)
    plt.plot(step_history, [l[2] for l in loss_history], label='TV Loss', color='#2ca02c', linewidth=2)
    plt.xlabel('Global Step')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()

    # 2. 梯度范数曲线（过滤None值）
    plt.subplot(3, 1, 2)
    valid_grad = [(s, g) for s, g in zip(step_history, grad_norm_history) if g is not None]
    if valid_grad:
        s_valid, g_valid = zip(*valid_grad)
        plt.plot(s_valid, g_valid, color='#d62728', linewidth=2)
    plt.xlabel('Global Step')
    plt.ylabel('Gradient Norm')
    plt.title('Patch Gradient Norm Curve')
    plt.tight_layout()

    # 3. 学习率变化曲线（标记衰减点）
    plt.subplot(3, 1, 3)
    plt.plot(step_history, lr_history, color='#9467bd', linewidth=2)
    # 标记学习率衰减点（使用传入的dataloader计算len）
    if step_history and DECAY_EPOCH <= NUM_EPOCHS:
        decay_step = DECAY_EPOCH * len(dataloader)  # 现在dataloader可正常访问
        if decay_step <= max(step_history):
            plt.axvline(x=decay_step, color='red', linestyle='--', alpha=0.7, label=f'Decay (epoch={DECAY_EPOCH})')
            plt.legend()
    plt.xlabel('Global Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (BlackBox Paper)')
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(save_dir, 'training_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"训练可视化图表已保存至: {save_path}")


def main():
    # 数据加载
    dataloader = get_inria_dataloader(
        DATA_ROOT, split="Train", batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, disable_random_aug=True
    )
    logger.info(f"训练数据集大小: {len(dataloader.dataset)}")
    logger.info(f"学习率策略：初始LR={INIT_LR}，第{DECAY_EPOCH}个epoch后衰减至{INIT_LR*DECAY_FACTOR}（BlackBox论文）")

    # 模型初始化
    model = load_detr_r50().to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # TMM初始化
    tmm = TransformerMaskingMatrix(
        num_enc_layers=6, num_dec_layers=6, p_base=0.2,
        sampling_strategy='categorical', device=DEVICE
    )
    tmm.register_hooks(model)
    tmm.reset_grad_history()

    # GSE和损失函数
    gse = GradientSelfEnsemble(model=model, device=DEVICE)
    loss_fn = BlackBoxLoss(
        gse=gse, target_class=TARGET_CLASS_IDX,
        detection_weight=DETECTION_WEIGHT, tv_weight=TV_WEIGHT, l2_weight=0.0,
        layer_aggregation='per_layer_loss', use_sigmoid_for_binary=False,
        device=DEVICE
    )

    # 初始化Patch
    patch = torch.randn(1, 3, PATCH_SIDE, PATCH_SIDE, device=DEVICE) * PATCH_INIT_STD + 0.5
    patch = patch.clamp(0.0, 1.0)
    patch.requires_grad_(True)

    # 优化器与调度器（符合论文一次衰减）
    optimizer = torch.optim.Adam([patch], lr=INIT_LR)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: DECAY_FACTOR if epoch >= DECAY_EPOCH else 1.0
    )

    # 可视化记录列表
    step_history = []
    loss_history = []
    grad_norm_history = []
    lr_history = []

    logger.info(f"开始训练（EoT已禁用，PATCH_RATIO={PATCH_RATIO}），结果保存至: {SAVE_DIR}")

    # 训练循环
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.eval()
        tmm.reset_grad_history()

        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(DEVICE).clamp(0, 1)
            B = imgs.shape[0]

            # 1. 检测干净图像目标框
            tmm.remove_hooks()
            with torch.no_grad():
                try:
                    det_out = model(imgs)
                except Exception:
                    det_out = model(NestedTensor(imgs))
            tmm.register_hooks(model)

            # 2. 筛选目标框
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

            # 3. 生成带Patch的图像
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

                    patch_to_paste = patch
                    patch_resized = interpolate(patch_to_paste, size=(side, side),
                                               mode='bilinear', align_corners=False)
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    patched[bi] = paste_patch_via_mask(patched[bi], patch_resized, (cx, cy))

            # 4. 计算损失并更新Patch
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

            # 5. 记录可视化数据
            if batch_idx % 10 == 0:
                det_loss_v = loss_dict.get('det_loss', 0.0)
                det_loss_v = det_loss_v.item() if isinstance(det_loss_v, torch.Tensor) else float(det_loss_v)
                tv_loss_v = loss_dict.get('tv_loss', 0.0)
                tv_loss_v = tv_loss_v.item() if isinstance(tv_loss_v, torch.Tensor) else float(tv_loss_v)
                current_lr = optimizer.param_groups[0]['lr']

                step_history.append(global_step)
                loss_history.append((total_loss.item(), det_loss_v, tv_loss_v))
                grad_norm_history.append(grad_norm)
                lr_history.append(current_lr)

            # 6. 日志记录
            det_loss_v = loss_dict.get('det_loss', 0.0)
            det_loss_v = det_loss_v.item() if isinstance(det_loss_v, torch.Tensor) else float(det_loss_v)
            tv_loss_v = loss_dict.get('tv_loss', 0.0)
            tv_loss_v = tv_loss_v.item() if isinstance(tv_loss_v, torch.Tensor) else float(tv_loss_v)
            avg_patch_side = np.mean(patch_sizes) if patch_sizes else 0
            logger.info(
                f"[epoch {epoch+1}/{NUM_EPOCHS} batch {batch_idx}] "
                f"total_loss={total_loss.item():.6f} | "
                f"det_loss={det_loss_v:.6f} | "
                f"tv_loss={tv_loss_v:.6f} | "
                f"grad_norm={grad_norm:.4f} | "
                f"目标框数量={[b.shape[0] for b in batch_boxes_all]} | "
                f"平均Patch尺寸={avg_patch_side:.1f} | "
                f"当前学习率={optimizer.param_groups[0]['lr']:.6f}"
            )

            # 7. 保存样本可视化
            if global_step % 200 == 0:
                orig_with_boxes = draw_boxes_on_tensor(detach_cpu(imgs[0]), batch_boxes_all[0])
                patched_with_boxes = draw_boxes_on_tensor(detach_cpu(patched[0]), batch_boxes_all[0])
                save_image(orig_with_boxes, os.path.join(SAVE_DIR, f"step_{global_step}_orig_with_boxes.png"))
                save_image(patched_with_boxes, os.path.join(SAVE_DIR, f"step_{global_step}_patched_with_boxes.png"))
                save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"step_{global_step}_patch.png"))
            global_step += 1

        # 更新学习率调度器
        scheduler.step()

        # 核心修改2：调用可视化函数时传入dataloader
        if step_history and loss_history:
            plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, VISUAL_DIR, dataloader)

        # 保存epoch Patch
        save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.png"))
        torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.pt"))
        logger.info(
            f"Epoch {epoch+1} 保存Patch快照 | "
            f"当前学习率={optimizer.param_groups[0]['lr']:.6f} | "
            f"距离学习率衰减剩余epoch: {max(0, DECAY_EPOCH - epoch)}"
        )

    # 训练结束：最终可视化
    if step_history and loss_history:
        plot_training_curves(step_history, loss_history, grad_norm_history, lr_history, VISUAL_DIR, dataloader)
    tmm.remove_hooks()
    save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.pt"))
    logger.info(f"训练完成！最终Patch保存至: {SAVE_DIR} | 最终学习率={optimizer.param_groups[0]['lr']:.6f}")


if __name__ == "__main__":
    main()