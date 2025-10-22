#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor-DETR 模型加载器（带仓库隔离机制）
路径: /opt/data/private/BlackBox/utils/models_loader/anchor_detr_loader.py
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# ===============================================================
# 路径配置
# ===============================================================
anchor_detr_root = Path("/opt/data/private/BlackBox/models/anchor_detr")
weight_path = "/opt/data/private/BlackBox/models/weights/AnchorDETR_r50_c5.pth"

assert anchor_detr_root.exists(), f"❌ Anchor-DETR 根目录不存在：{anchor_detr_root}"
assert (anchor_detr_root / "models").exists(), f"models 文件夹不存在：{anchor_detr_root / 'models'}"

# ===============================================================
# 仓库隔离函数（统一版本）
# ===============================================================
_SAFE_PREFIXES = (
    "sys", "builtins", "os", "types", "importlib", "pkgutil", "pkg_resources",
    "torch", "numpy", "cv2", "json", "logging", "warnings", "inspect",
)

def isolate_repo(repo_root: str, extra_keep_prefixes=()):
    """
    在导入某个模型仓库前调用，保证 'models' / 'util' 等指向该 repo：
    1) 从 sys.modules 中删除可能冲突的模块（以 'models' 或 'util' 开头的项），
       但保留 _SAFE_PREFIXES 与 extra_keep_prefixes 指定的模块。
    2) 将 repo 的路径及常用子目录插入 sys.path（在最前面）。
    """
    repo_root = os.fspath(repo_root)
    # 删除缓存模块
    keep_prefixes = tuple(_SAFE_PREFIXES) + tuple(extra_keep_prefixes)
    keys = list(sys.modules.keys())
    for k in keys:
        if any(k == p or k.startswith(p + ".") for p in keep_prefixes):
            continue
        if k == "models" or k.startswith("models.") \
           or k == "util" or k.startswith("util.") \
           or k == "ops" or k.startswith("ops.") \
           or k == "datasets" or k.startswith("datasets."):
            try:
                del sys.modules[k]
            except KeyError:
                pass

    # 插入 repo 路径
    to_add = [
        repo_root,
        os.path.join(repo_root, "models"),
        os.path.join(repo_root, "util"),
        os.path.join(repo_root, "models", "ops"),
    ]
    for p in to_add:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

# ===============================================================
# 主加载函数
# ===============================================================
def load_anchor_detr(device='cuda'):
    """加载 Anchor-DETR 模型（含权重与隔离机制）"""
    # ✅ 仓库隔离（核心一步）
    isolate_repo(anchor_detr_root)

    # 延后导入以避免命名冲突
    from models import build_model
    from util.misc import NestedTensor

    # 构造模型参数（完整对齐 main.py）
    parser = argparse.ArgumentParser('AnchorDETR配置', add_help=False)
    
    # ---- 模型结构参数 ----
    parser.add_argument('--dataset_file', default='coco', type=str, 
                        help='指定数据集类型，AnchorDETR 默认使用 coco')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', default=False, type=bool)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--num_query_position', default=300, type=int)
    parser.add_argument('--num_query_pattern', default=3, type=int)
    parser.add_argument('--spatial_prior', default='learned', type=str, choices=['learned', 'grid'])
    parser.add_argument('--attention_type', default='RCDA', type=str, choices=['RCDA', 'nn.MultiheadAttention'])
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--aux_loss', default=True, action='store_true')
    parser.add_argument('--device', default=device, type=str)
    parser.add_argument('--coco_path', default='', type=str)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # ---- 匹配器权重参数 ----
    parser.add_argument('--set_cost_class', default=2.0, type=float,
                        help="匹配成本中的类别系数")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float,
                        help="匹配成本中的 L1 框系数")
    parser.add_argument('--set_cost_giou', default=2.0, type=float,
                        help="匹配成本中的 GIoU 框系数")
    
    # ---- 损失权重参数 ----
    parser.add_argument('--cls_loss_coef', default=2.0, type=float,
                        help="分类损失系数")
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float,
                        help="边界框损失系数")
    parser.add_argument('--giou_loss_coef', default=2.0, type=float,
                        help="GIoU 损失系数")
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help="Focal Loss 的 alpha 参数")
    
    args = parser.parse_args([])

    # === 构建模型 ===
    print("🧩 构建 Anchor-DETR 模型结构中...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # === 加载预训练权重 ===
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"✅ 成功加载预训练权重：{weight_path}")
    else:
        print(f"⚠️ 未找到权重文件：{weight_path}，使用随机初始化模型")
    
    # === 冻结并设为 eval ===
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"🚀 Anchor-DETR 模型加载完成 ({device})")
    return model, postprocessors

# ===============================================================
# 测试模块
# ===============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")
    model, postprocessors = load_anchor_detr(device=device)
    print("✅ Anchor-DETR 模型加载测试通过。")
