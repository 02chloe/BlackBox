# attack_t01.py 只有attack，不部署其他utils
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.ops import box_convert, nms
from torch.nn.functional import interpolate
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import logging

# 自定义模块导入
from inria_dataloader import get_inria_dataloader
from tmm import load_detr_r50, NestedTensor  # NestedTensor用于DETR输入兼容

# -----------------------
# 补充缺失的辅助函数：绘制边界框
# -----------------------
def draw_boxes_on_tensor(img_tensor: torch.Tensor, boxes_xyxy_cpu: torch.Tensor):
    """
    在图像张量上绘制边界框
    img_tensor: (3, H, W) 张量（值范围0-1）
    boxes_xyxy_cpu: 边界框（xyxy格式，像素坐标，CPU张量）
    返回：带边界框的图像张量
    """
    if boxes_xyxy_cpu is None or boxes_xyxy_cpu.numel() == 0:
        return img_tensor
    # 转换为uint8格式（draw_bounding_boxes要求）
    img_uint8 = (img_tensor * 255).byte()
    boxes = boxes_xyxy_cpu.clone()
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    # 确保边界框在图像范围内
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)
    # 过滤无效框（宽高为0的）
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if boxes.shape[0] == 0:
        return img_tensor
    # 绘制边界框
    boxes_int = boxes.to(torch.int64)
    img_boxes = draw_bounding_boxes(img_uint8, boxes=boxes_int, colors="red", width=2)
    return img_boxes.float() / 255.0  # 转回0-1范围

# -----------------------
# 配置参数
# -----------------------
ROOT = "/opt/data/private/BlackBox"
DATA_ROOT = "/opt/data/private/BlackBox/T-SEA-B/data/INRIAPerson"
TEST_DATA_DIR = os.path.join(DATA_ROOT, "Test")
SAVE_DIR = os.path.join(ROOT, "save", "attack_results")
PATCH_PATH = os.path.join(ROOT, "save", "demo", "final_patch.pt")  # 训练好的Patch路径
VISUAL_DIR = os.path.join(SAVE_DIR, "visual")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# 模型与检测参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_H, MODEL_INPUT_W = 640, 640
TARGET_CLASS_IDX = 1
SCORE_THRESH = 0.5        # 检测置信度阈值（可调整）
IOU_NMS_THRESH = 0.5      # NMS去重阈值
MIN_BOX_SIDE = 20         # 最小检测框边长

# Patch参数
PATCH_RATIO = 0.5         # Patch尺寸与目标框短边的比例
MIN_PATCH_PX = 16         # Patch最小像素尺寸

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# -----------------------
# 核心函数：模块化检测（训练/攻击复用）
# -----------------------
def detect_boxes(model, imgs):
    """
    对图像进行“干净检测”（无TMM/GSE，仅原始DETR）
    返回：每张图像的检测框（xyxy格式，像素坐标）
    """
    model.eval()
    batch_boxes_all = []
    B = imgs.shape[0]
    with torch.no_grad():
        try:
            det_out = model(imgs)
        except Exception:
            det_out = model(NestedTensor(imgs))  # 兼容DETR输入格式

    for bi in range(B):
        logits = det_out['pred_logits'][bi]
        boxes = det_out['pred_boxes'][bi]
        probs = torch.softmax(logits, dim=-1)
        cls_scores = probs[..., TARGET_CLASS_IDX]

        # 筛选高置信度框
        keep_idx = (cls_scores > SCORE_THRESH).nonzero(as_tuple=False).squeeze(1) \
            if (cls_scores > SCORE_THRESH).any() else torch.tensor([], dtype=torch.long, device=cls_scores.device)

        if keep_idx.numel() == 0:
            batch_boxes_all.append(torch.empty((0, 4), dtype=torch.float32))
            continue

        # 转换为像素坐标并筛选有效框
        sel_boxes = boxes[keep_idx]
        sel_xyxy = detr_boxes_to_xyxy_pixel(sel_boxes.detach().cpu())
        widths = sel_xyxy[:, 2] - sel_xyxy[:, 0]
        heights = sel_xyxy[:, 3] - sel_xyxy[:, 1]
        large_mask = (widths >= MIN_BOX_SIDE) & (heights >= MIN_BOX_SIDE)
        
        if large_mask.sum() == 0:
            batch_boxes_all.append(torch.empty((0, 4), dtype=torch.float32))
            continue

        # NMS去重
        sel_xyxy = sel_xyxy[large_mask]
        sel_scores_cpu = cls_scores[keep_idx].detach().cpu()[large_mask]
        keep_nms = nms(sel_xyxy, sel_scores_cpu, IOU_NMS_THRESH)
        sel_xyxy_nms = sel_xyxy[keep_nms]
        batch_boxes_all.append(sel_xyxy_nms)
    return batch_boxes_all


def detr_boxes_to_xyxy_pixel(pred_boxes):
    """将DETR输出的cxcywh格式转换为xyxy像素坐标"""
    pb = pred_boxes.clone()
    if pb.max() <= 1.01:  # 若为归一化坐标，转换为像素
        pb[:, 0] = pb[:, 0] * MODEL_INPUT_W
        pb[:, 1] = pb[:, 1] * MODEL_INPUT_H
        pb[:, 2] = pb[:, 2] * MODEL_INPUT_W
        pb[:, 3] = pb[:, 3] * MODEL_INPUT_H
    return box_convert(pb, in_fmt='cxcywh', out_fmt='xyxy').cpu()


# -----------------------
# Patch粘贴函数（攻击阶段：无变换，直接粘贴）
# -----------------------
def paste_patch_attack(base_img, patch_tensor, center_xy, target_side):
    """
    攻击阶段粘贴Patch：无随机变换，直接缩放后粘贴到目标框中心
    base_img: (3, H, W) tensor
    patch_tensor: (3, PATCH_SIDE, PATCH_SIDE) tensor（训练好的Patch）
    center_xy: (cx, cy) 目标框中心像素坐标
    target_side: Patch需缩放到的边长（目标框短边 × PATCH_RATIO）
    """
    # 缩放Patch到目标尺寸
    patch_resized = interpolate(
        patch_tensor.unsqueeze(0), 
        size=(target_side, target_side),
        mode='bilinear', 
        align_corners=False
    )[0]
    ph, pw = patch_resized.shape[1], patch_resized.shape[2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = cx - pw // 2
    y0 = cy - ph // 2

    H, W = base_img.shape[1], base_img.shape[2]
    # 边界裁剪（避免Patch超出图像）
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(W, x0 + pw)
    dst_y1 = min(H, y0 + ph)

    out_w = dst_x1 - dst_x0
    out_h = dst_y1 - dst_y0
    if out_w <= 0 or out_h <= 0:
        return base_img.clone()

    # 粘贴Patch（直接替换图像对应区域）
    base_img[:, dst_y0:dst_y1, dst_x0:dst_x1] = patch_resized[:, :out_h, :out_w]
    return base_img


# -----------------------
# 主函数：黑盒攻击流程
# -----------------------
def main():
    # 1. 加载DETR模型（原始模型，无TMM/GSE）
    model = load_detr_r50().to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("DETR-R50模型加载完成（原始黑盒模型）")

    # 2. 加载训练好的Patch
    patch = torch.load(PATCH_PATH, map_location=DEVICE, weights_only=True)  # 修复torch.load警告
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)  # 确保形状为(1, 3, H, W)
    patch = patch.to(DEVICE).clamp(0.0, 1.0)
    logger.info(f"Patch加载完成，路径：{PATCH_PATH}，形状：{patch.shape}")

    # 3. 加载测试数据集
    dataloader = get_inria_dataloader(
        DATA_ROOT, 
        split="Test", 
        batch_size=1,  # 测试时单张处理，方便可视化
        num_workers=4, 
        disable_random_aug=True
    )
    logger.info(f"测试数据集大小：{len(dataloader.dataset)}")

    # 4. 评估指标初始化
    total_targets = 0       # 干净检测的总目标数
    attacked_targets = 0    # 攻击后未被检测到的目标数

    # 5. 遍历测试数据，执行攻击
    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(DEVICE).clamp(0, 1)
        B = imgs.shape[0]

        for bi in range(B):
            img = imgs[bi]
            # 5.1 干净检测：获取目标框
            clean_boxes = detect_boxes(model, img.unsqueeze(0))[0]
            if clean_boxes.numel() == 0:
                continue  # 无目标，跳过
            total_targets += len(clean_boxes)

            # 5.2 粘贴Patch到每个目标框
            attacked_img = img.clone()
            for box in clean_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                box_w = max(int(xmax - xmin), 1)
                box_h = max(int(ymax - ymin), 1)
                short_side = min(box_w, box_h)
                target_side = max(MIN_PATCH_PX, int(round(short_side * PATCH_RATIO)))

                # 计算目标框中心
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0

                # 粘贴Patch（无变换）
                attacked_img = paste_patch_attack(attacked_img, patch[0], (cx, cy), target_side)

            # 5.3 攻击后检测：判断目标是否被“攻击”（未被检测到）
            attacked_boxes = detect_boxes(model, attacked_img.unsqueeze(0))[0]

            # 5.4 统计攻击成功数（攻击后目标框消失或数量减少）
            if attacked_boxes.numel() == 0:
                attacked_targets += len(clean_boxes)
            else:
                # 若攻击后检测框数量减少，统计减少的数量
                attacked_targets += (len(clean_boxes) - len(attacked_boxes))

            # 5.5 可视化：保存原图、攻击后图、Patch
            if batch_idx % 10 == 0:
                # 原图带检测框
                orig_with_boxes = draw_boxes_on_tensor(img.detach().cpu(), clean_boxes)
                # 攻击后图带检测框
                attacked_with_boxes = draw_boxes_on_tensor(attacked_img.detach().cpu(), attacked_boxes)
                # 保存图像
                save_image(orig_with_boxes, os.path.join(VISUAL_DIR, f"test_{batch_idx}_orig.png"))
                save_image(attacked_with_boxes, os.path.join(VISUAL_DIR, f"test_{batch_idx}_attacked.png"))
                save_image(patch[0].detach().cpu(), os.path.join(VISUAL_DIR, "patch.png"))

        # 打印进度与临时指标
        if batch_idx % 50 == 0:
            success_rate = attacked_targets / max(1, total_targets) * 100
            logger.info(f"已处理 {batch_idx}/{len(dataloader)} 批次 | "
                        f"当前攻击成功率: {success_rate:.2f}%")

    # 6. 计算最终攻击指标（Recall：攻击成功的目标比例）
    final_success_rate = attacked_targets / max(1, total_targets) * 100
    logger.info(f"=== 攻击实验完成 ===")
    logger.info(f"总目标数: {total_targets}")
    logger.info(f"攻击成功数: {attacked_targets}")
    logger.info(f"最终攻击成功率 (Recall): {final_success_rate:.2f}%")

    # 7. 保存评估结果
    with open(os.path.join(SAVE_DIR, "attack_metrics.txt"), "w") as f:
        f.write(f"Total targets: {total_targets}\n")
        f.write(f"Attacked targets: {attacked_targets}\n")
        f.write(f"Attack success rate (Recall): {final_success_rate:.2f}%\n")


if __name__ == "__main__":
    main()