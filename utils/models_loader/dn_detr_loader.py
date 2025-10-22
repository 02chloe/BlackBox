import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Tuple, Any

# ===============================================================
# 路径配置
# ===============================================================
DN_DETR_ROOT = Path("/opt/data/private/BlackBox/models/DN-DETR")
WEIGHT_PATH = "/opt/data/private/BlackBox/models/weights/dn_detr_r50_50ep.pth"
assert DN_DETR_ROOT.exists(), f"❌ DN-DETR 根目录不存在：{DN_DETR_ROOT}"

# ===============================================================
# 仓库隔离（避免与其他仓库的 models/util 命名冲突）
# ===============================================================
_SAFE_PREFIXES = (
    "sys", "builtins", "os", "types", "importlib", "pkgutil", "pkg_resources",
    "torch", "numpy", "cv2", "json", "logging", "warnings", "inspect",
)

def isolate_repo(repo_root: str, extra_keep_prefixes=()):
    repo_root = os.fspath(repo_root)
    keep_prefixes = tuple(_SAFE_PREFIXES) + tuple(extra_keep_prefixes)

    # 1) 清理可能冲突的已缓存模块
    for k in list(sys.modules.keys()):
        if any(k == p or k.startswith(p + ".") for p in keep_prefixes):
            continue
        if k in ("models", "util", "ops", "datasets") or \
           k.startswith(("models.", "util.", "ops.", "datasets.")):
            try:
                del sys.modules[k]
            except KeyError:
                pass

    # 2) 注入目标仓库路径（优先）
    to_add = [
        repo_root,
        os.path.join(repo_root, "models"),
        os.path.join(repo_root, "util"),
        os.path.join(repo_root, "models", "dn_dab_deformable_detr", "ops"),
    ]
    for p in to_add:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

# ===============================================================
# 参数补全
# ===============================================================
def ensure_arg_defaults(args, defaults: dict):
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args

# ===============================================================
# 前向保护器：在 forward 调用期间，强制关闭 DN 相关开关
# ===============================================================
def _wrap_forward_disable_dn(model):
    if not hasattr(model, "forward"):
        return model  # 非预期，跳过

    orig_forward = model.forward

    def safe_forward(*f_args, **f_kwargs):
        # 双保险：在一次 forward 的生命周期内，强制关闭 DN
        had_use_dn = hasattr(model, "use_dn")
        had_scalar = hasattr(model, "scalar")
        old_use_dn = getattr(model, "use_dn", False)
        old_scalar = getattr(model, "scalar", 0)

        try:
            if had_use_dn:
                model.use_dn = False
            if had_scalar:
                try:
                    # 部分实现 scalar 可能是 Tensor / nn.Parameter
                    if isinstance(model.scalar, torch.Tensor):
                        model.scalar = model.scalar.detach().clone()
                        model.scalar.zero_()
                    else:
                        model.scalar = 0
                except Exception:
                    model.scalar = 0
            return orig_forward(*f_args, **f_kwargs)
        finally:
            # 恢复原状
            if had_use_dn:
                model.use_dn = old_use_dn
            if had_scalar:
                model.scalar = old_scalar

    model.forward = safe_forward
    return model

# ===============================================================
# 官方 DN-DETR 加载（禁用 DN，返回顺序：model, criterion, postprocessors）
# ===============================================================
def load_dn_detr(device: str = "cuda") -> Tuple[Any, Any, Any]:
    """
    加载 DN-DETR-R50 (COCO 50ep, 44.41 mAP) 推理模型：
    - 强制禁用 DeNoising（use_dn=False, scalar=0）
    - 返回 (model, criterion, postprocessors)
    - 仓库路径隔离，避免命名冲突
    """
    isolate_repo(str(DN_DETR_ROOT))

    # 延迟导入（在隔离后）
    from models import build_dab_deformable_detr
    # from util.misc import nested_tensor_from_tensor_list  # 仅构建期需要时才会用到

    # ---- 构建 args：显式关闭 DN ----
    parser = argparse.ArgumentParser("DN-DETR config (eval)", add_help=False)
    parser.add_argument("--device", default=device, type=str)
    parser.add_argument("--dataset_file", default="coco", type=str)
    parser.add_argument("--num_classes", default=91, type=int)
    # 关键：默认 False，并且不用 action（避免误触）
    parser.add_argument("--use_dn", type=bool, default=False)
    parser.add_argument("--num_patterns", type=int, default=10)
    parser.add_argument("--scalar", type=int, default=0)  # 关键：0 代表无 DN queries
    args = parser.parse_args([])

    # 其余必要超参（与官方默认对齐，确保能正常 build）
    args = ensure_arg_defaults(args, {
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "batch_size": 2,
        "num_workers": 2,
        "clip_max_norm": 0.1,

        "backbone": "resnet50",
        "dilation": False,
        "position_embedding": "sine",
        "enc_layers": 6,
        "dec_layers": 6,
        "dim_feedforward": 2048,
        "hidden_dim": 256,
        "nheads": 8,
        "num_queries": 300,
        "num_feature_levels": 4,
        "dropout": 0.1,
        "pre_norm": True,

        "enc_n_points": 4,
        "dec_n_points": 4,

        # DN 相关（即便存在也会被我们强制关闭）
        "label_noise_scale": 0.2,
        "box_noise_scale": 0.4,

        # 损失与匹配
        "aux_loss": True,
        "cls_loss_coef": 1,
        "bbox_loss_coef": 5,
        "giou_loss_coef": 2,
        "set_cost_class": 2,
        "set_cost_bbox": 5,
        "set_cost_giou": 2,
        "focal_alpha": 0.25,
        "eos_coef": 0.1,

        "masks": False,
        "two_stage": False,
        "random_refpoints_xy": False,
    })

    print("🧩 构建官方 DN-DETR 模型结构中...")
    print(f"   参数: use_dn={args.use_dn}, scalar={args.scalar}, device={args.device}, dataset={args.dataset_file}")

    # ---- 构建模型 ----
    model, criterion, postprocessors = build_dab_deformable_detr(args)
    model.to(device)

    # ---- 加载权重 ----
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"❌ 未找到权重文件: {WEIGHT_PATH}")
    print(f"📥 加载权重文件: {WEIGHT_PATH}")
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    try:
        from util.utils import clean_state_dict
        state_dict = clean_state_dict(state_dict)
    except Exception:
        pass
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✅ 权重加载完成（缺失键 {len(missing)}, 意外键 {len(unexpected)}）")

    # ---- Eval + 冻结 ----
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- 双保险：彻底关闭 DN ----
    if hasattr(model, "use_dn"):
        try:
            print(f"🛡️ 覆盖 model.use_dn: {getattr(model,'use_dn')} -> False")
            model.use_dn = False
        except Exception:
            pass
    if hasattr(model, "scalar"):
        try:
            if isinstance(model.scalar, torch.Tensor):
                print(f"🛡️ 覆盖 model.scalar: Tensor -> 0")
                model.scalar = model.scalar.detach().clone()
                model.scalar.zero_()
            else:
                print(f"🛡️ 覆盖 model.scalar: {getattr(model,'scalar')} -> 0")
                model.scalar = 0
        except Exception:
            pass

    # ---- 前向保护器（一次 forward 周期内强制 use_dn=False, scalar=0）----
    model = _wrap_forward_disable_dn(model)

    # 注意返回顺序：与检测脚本保持一致 (model, criterion, postprocessors)
    return model, criterion, postprocessors


# ===============================================================
# 独立测试
# ===============================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{dev}")
    m, c, pp = load_dn_detr(device=dev)
    print("🚀 官方 DN-DETR (R50, 50ep) 加载完成（推理态，无 DN）")
