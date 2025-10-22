# /opt/data/private/BlackBox/utils/sparse_detr_loader.py
import torch
import os
import sys
import inspect

# 列出不应被删除的模块前缀（保留核心库）
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
    # 1) 删除缓存模块
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

    # 2) 插入 repo 路径到 sys.path（优先）
    to_add = [
        repo_root,
        os.path.join(repo_root, "models"),
        os.path.join(repo_root, "util"),
        os.path.join(repo_root, "models", "ops"),
    ]
    for p in to_add:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

# ---------------------------------------------------------

def load_sparse_detr():
    """
    智能容错版 Sparse-DETR 加载函数
    - 自动修复环境与torchvision兼容
    - 自动补齐缺失参数
    - 离线加载缓存权重
    返回: 已加载完成的 model
    """
    print("🚀 启动 Sparse-DETR 智能加载器...")

    # === 路径设置 ===
    sparse_detr_path = "/opt/data/private/BlackBox/models/sparse_detr"

    # ✅ 关键：隔离并确保本仓库占据 'models' / 'util' 命名空间
    isolate_repo(sparse_detr_path)

    # （可选）以下两行已由 isolate_repo 插入，可删除
    # sys.path.insert(0, sparse_detr_path)
    # sys.path.insert(0, os.path.join(sparse_detr_path, "models", "ops"))

    # === torchvision兼容性修复 ===
    import torchvision.ops.misc as misc
    if not hasattr(misc, '_NewEmptyTensorOp'):
        class _NewEmptyTensorOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, shape):
                return x.new_empty(shape)
            @staticmethod
            def backward(ctx, grad):
                return None, None
        misc._NewEmptyTensorOp = _NewEmptyTensorOp

    # === 导入模型构建函数（此时一定来自 Sparse-DETR 仓库）===
    from models import build_model

    # === 定义参数类（基础字段） ===
    class Args:
        def __init__(self):
            # Backbone
            self.backbone = 'resnet50'
            self.backbone_from_scratch = False
            self.scrl_pretrained_path = None
            self.lr_backbone = 1e-5
            self.dilation = False
            self.position_embedding = 'sine'
            self.finetune_early_layers = 0

            # Transformer参数
            self.enc_layers = 6
            self.dec_layers = 6
            self.dim_feedforward = 1024
            self.hidden_dim = 256
            self.dropout = 0.1
            self.nheads = 8
            self.num_queries = 300
            self.pre_norm = False
            self.num_feature_levels = 4
            self.enc_n_points = 4
            self.dec_n_points = 4

            # 训练参数
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.masks = False
            self.eval = False
            self.two_stage = True
            self.two_stage_num_proposals = 300
            self.with_box_refine = True
            self.aux_loss = True
            self.use_enc_aux_loss = False

            # 损失权重
            self.set_cost_class = 2
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.cls_loss_coef = 2.0
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.eos_coef = 0.1

            # Sparse-DETR 扩展
            self.eff_query_init = False
            self.eff_specific_head = False
            self.rho = 1.0

            # 数据集参数
            self.dataset_file = 'coco'
            self.remove_difficult = False
            self.coco_path = None
            self.coco_panoptic_path = None

        def __contains__(self, name):
            return hasattr(self, name)

    args = Args()

    # === 自动补齐机制 ===
    def auto_fill_args(args, module):
        """
        自动扫描module源码中 args.xxx 调用
        若Args未定义该字段，则补默认值
        """
        source = inspect.getsource(module)
        lines = [line.strip() for line in source.split("\n") if "args." in line]
        added = []
        for line in lines:
            parts = line.split("args.")
            for p in parts[1:]:
                key = p.split()[0].split(")")[0].split(",")[0].split("=")[0]
                key = key.replace(":", "").replace(".", "").strip()
                if key and not hasattr(args, key):
                    setattr(args, key, False)  # 默认False
                    added.append(key)
        if added:
            print(f"⚙️ 自动补齐参数: {', '.join(sorted(set(added)))}")

    # 自动扫描构建模块中的 args 使用
    import models.deformable_detr as deformable_detr
    import models.deformable_transformer as deformable_transformer
    import models.backbone as backbone
    for m in [deformable_detr, deformable_transformer, backbone]:
        auto_fill_args(args, m)

    # === 构建模型 ===
    print("🧩 构建 Sparse-DETR 模型结构中...")
    model, criterion, postprocessors = build_model(args)

    # === 加载权重 ===
    cache_dir = "/opt/data/private/BlackBox/models/torch_cache/hub/sparse_detr_cache"
    state_dict_path = os.path.join(cache_dir, "sparse_detr_r50_state_dict.pth")
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"未找到模型缓存: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # === 设备与模式 ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"✅ Sparse-DETR 模型加载成功 ({device})")
    return model

# === 示例调用 ===
if __name__ == "__main__":
    model = load_sparse_detr()
    print("模型可直接用于推理。")
