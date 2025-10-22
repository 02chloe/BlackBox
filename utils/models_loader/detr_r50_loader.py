#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETR-R50 离线加载器（带仓库隔离机制 + 本地缓存 torch.hub 加载）
路径: /opt/data/private/BlackBox/utils/models_loader/detr_r50_loader.py
"""

import os
import sys
import torch
from pathlib import Path

# ===============================================================
# 路径配置
# ===============================================================
REPO_PATH = Path("/opt/data/private/BlackBox/models/torch_cache/hub/facebookresearch_detr_main")
WEIGHT_PATH = Path("/opt/data/private/BlackBox/models/weights/detr-r50-e632da11.pth")

assert REPO_PATH.exists(), f"❌ 缓存仓库不存在: {REPO_PATH}"
assert WEIGHT_PATH.exists(), f"❌ 权重文件不存在: {WEIGHT_PATH}"

# ===============================================================
# 仓库隔离函数（与其他模型统一）
# ===============================================================
_SAFE_PREFIXES = (
    "sys", "builtins", "os", "types", "importlib", "pkgutil",
    "pkg_resources", "torch", "numpy", "cv2", "json", "logging",
    "warnings", "inspect",
)

def isolate_repo(repo_root: str, extra_keep_prefixes=()):
    """
    在导入某个模型仓库前调用，确保 models/util/ops 等模块隔离，
    防止与其他仓库冲突。
    """
    repo_root = os.fspath(repo_root)
    keep_prefixes = tuple(_SAFE_PREFIXES) + tuple(extra_keep_prefixes)
    for k in list(sys.modules.keys()):
        if any(k == p or k.startswith(p + ".") for p in keep_prefixes):
            continue
        if k in ("models", "util", "ops", "datasets") or k.startswith(("models.", "util.", "ops.", "datasets.")):
            try:
                del sys.modules[k]
            except KeyError:
                pass

    # 插入 repo 目录及其子路径
    to_add = [
        repo_root,
        os.path.join(repo_root, "models"),
        os.path.join(repo_root, "util"),
    ]
    for p in to_add:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

# ===============================================================
# 主加载函数
# ===============================================================
def load_detr_r50(device="cuda"):
    """
    离线加载官方 DETR-R50 模型（直接从本地缓存 main 文件夹加载，无需联网）
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("🚀 启动 DETR-R50 离线加载器（带仓库隔离）...")

    # ✅ 启用隔离，防止与 DN-DETR/Deformable-DETR 冲突
    isolate_repo(REPO_PATH)

    # ✅ 直接从本地仓库加载（不会联网）
    model = torch.hub.load(
        repo_or_dir=str(REPO_PATH),
        model="detr_resnet50",   # hubconf.py 注册的官方模型名
        source="local",          # 强制从本地加载
        pretrained=False          # 不自动下载
    )

    # ✅ 加载本地权重
    print(f"📦 加载权重文件: {WEIGHT_PATH}")
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    # ✅ 模型冻结 + eval
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"✅ DETR-R50 离线加载成功！（设备: {device}）")
    return model

# ===============================================================
# 测试入口
# ===============================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detr_r50(dev)
    print("🚀 模型可直接用于推理。")
