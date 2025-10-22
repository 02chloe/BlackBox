#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deformable-DETR R50 Single-Scale 加载与测试（BlackBox 攻击用）
路径: /opt/data/private/BlackBox/utils/models_loader/deformable_detr_loader.py
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import subprocess

# ===============================================================
# 路径配置
# ===============================================================
DEFORMABLE_ROOT = Path("/opt/data/private/BlackBox/models/Deformable-DETR")
WEIGHT_PATH = "/opt/data/private/BlackBox/models/weights/r50_deformable_detr_single_scale-checkpoint.pth"

assert DEFORMABLE_ROOT.exists(), f"❌ 仓库路径不存在: {DEFORMABLE_ROOT}"
assert os.path.exists(WEIGHT_PATH), f"❌ 权重文件不存在: {WEIGHT_PATH}"

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
# 自动检测与编译 MultiScaleDeformableAttention
# ===============================================================
ops_dir = DEFORMABLE_ROOT / "models" / "ops"
if not any(f.name.startswith("MultiScaleDeformableAttention") and f.suffix == ".so"
           for f in ops_dir.glob("*.so")):
    print("⚙️ 检测到未编译的 MultiScaleDeformableAttention 扩展，正在执行 build_ext...")
    subprocess.run(["python", "setup.py", "build_ext", "--inplace"], cwd=str(ops_dir), check=True)
else:
    print("✅ MultiScaleDeformableAttention 扩展已存在。")

# ===============================================================
# 兼容性补丁（timm / torchvision）
# ===============================================================
import torchvision.ops.misc as misc
if not hasattr(misc, "_NewEmptyTensorOp"):
    class _NewEmptyTensorOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, new_shape):
            return x.new_empty(new_shape)
        @staticmethod
        def backward(ctx, grad):
            return None, None
    misc._NewEmptyTensorOp = _NewEmptyTensorOp

# ===============================================================
# 主加载函数
# ===============================================================
def load_deformable_detr(device="cuda"):
    # ✅ 关键：在导入 models 之前先隔离仓库
    isolate_repo(DEFORMABLE_ROOT)

    # 系统路径（确保优先搜索当前 repo）
    sys.path.insert(0, str(DEFORMABLE_ROOT))
    sys.path.insert(0, str(DEFORMABLE_ROOT / "models"))
    sys.path.insert(0, str(DEFORMABLE_ROOT / "util"))
    sys.path.insert(0, str(DEFORMABLE_ROOT / "models" / "ops"))

    # 构建函数定位
    def _get_build_fn():
        try:
            import models
        except Exception as e:
            raise ImportError(f"无法导入仓库内 models 包: {e}")

        candidates = [
            "build_deforamble_detr", "build_deformable_detr",
            "build", "build_model"
        ]
        for name in candidates:
            fn = getattr(models, name, None)
            if callable(fn):
                return fn

        # fallback 尝试子模块
        try:
            from models import deformable_detr as deformable_mod
            for name in ("build_deforamble_detr", "build_deformable_detr", "build"):
                fn = getattr(deformable_mod, name, None)
                if callable(fn):
                    return fn
        except Exception:
            pass
        raise ImportError("❌ 未在 Deformable-DETR 仓库中找到构建函数。")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    build_fn = _get_build_fn()

    # 导入 util 工具
    try:
        from util.misc import nested_tensor_from_tensor_list
    except Exception:
        try:
            from models.util.misc import nested_tensor_from_tensor_list
        except Exception as e:
            raise ImportError(
                "无法导入 nested_tensor_from_tensor_list，请确认 Deformable-DETR/util/misc.py 存在。"
            ) from e

    # ==== 参数构建 ====
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', default=str(device))
    parser.add_argument('--dataset_file', default='coco', type=str)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dilation', action='store_true', default=False)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--two_stage', action='store_true', default=False)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--with_box_refine', action='store_true', default=False)
    parser.add_argument('--num_classes', default=91, type=int)
    parser.add_argument('--aux_loss', action='store_true', default=False)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--frozen_weights', default=None, type=str)
    parser.add_argument('--remove_difficult', action='store_true', default=False)
    parser.add_argument('--pre_norm', action='store_true', default=False)
    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    args = parser.parse_args([])

    # === 构建模型 ===
    print("🧩 构建 Deformable-DETR 模型中...")
    built = build_fn(args)

    if isinstance(built, tuple):
        model = built[0]
        criterion = built[1] if len(built) > 1 else None
        postprocessors = built[2] if len(built) > 2 else None
    elif hasattr(built, "state_dict"):
        model = built
        criterion = None
        postprocessors = None
    elif isinstance(built, dict):
        model = built.get("model", None)
        criterion = built.get("criterion", None)
        postprocessors = built.get("postprocessors", None)
    else:
        raise RuntimeError("未知的 build_fn 返回结构。")

    assert model is not None, "构建失败：model 为 None"
    model.to(device)

    # === 加载 checkpoint ===
    print(f"📥 加载 checkpoint: {WEIGHT_PATH}")
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except Exception:
        new_state = {}
        for k, v in state_dict.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        state_dict = new_state

    print(f"✅ 权重加载完成（缺失键 {len(missing)}, 意外键 {len(unexpected)}）")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("🚀 Deformable-DETR 模型加载成功！")
    return model, nested_tensor_from_tensor_list


# ===============================================================
# 前向测试
# ===============================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {dev}")
    model, nested_fn = load_deformable_detr(device=dev)
    print("模型加载完成，可用于推理。")
