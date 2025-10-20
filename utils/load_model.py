# /opt/data/private/BlackBox/utils/load_model.py
import torch
from torch.hub import load_state_dict_from_url
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import inspect

"""
模型加载工具类：严格对齐《BlackBox》论文需求
- 白盒模型：DETR-R50（训练阶段用）
- 黑盒模型：Deformable-DETR、Sparse-DETR、Anchor-DETR（攻击阶段用）
所有模型加载后默认冻结参数、设为eval模式，与train/attack原有逻辑一致
"""

# -----------------------
# 1. 白盒模型：DETR-R50（完全复用tmm.py原有逻辑）
# -----------------------
def load_detr_r50(device: torch.device = None) -> torch.nn.Module:
    """
    加载DETR-R50模型（《BlackBox》论文白盒模型）
    功能与原tmm.py完全一致：冻结参数、eval模式、权重从官方URL下载
    Args:
        device: 模型部署设备（默认自动识别cuda/cpu）
    Returns:
        model: 冻结参数后的DETR-R50模型（eval模式）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载DETR-R50（复用原tmm.py逻辑）
    model = torch.hub.load(
        "facebookresearch/detr:main",
        "detr_resnet50",
        pretrained=False,
        force_reload=False
    )

    # 加载官方预训练权重（与原逻辑一致）
    weight_url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    checkpoint = load_state_dict_from_url(weight_url, progress=True)
    model.load_state_dict(checkpoint["model"])

    # 冻结参数+eval模式（核心：与train/attack原有逻辑一致）
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # 仅优化Patch，冻结模型参数

    return model


# -----------------------
# 2. 黑盒模型1：Deformable-DETR（《BlackBox》论文指定）
# -----------------------
def load_deformable_detr(device: torch.device = None) -> torch.nn.Module:
    """
    加载Deformable-DETR模型（《BlackBox》论文黑盒模型）
    遵循官方加载逻辑，冻结参数、eval模式
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从官方hub加载Deformable-DETR（r50 backbone，与DETR-R50一致）
    model = torch.hub.load(
        "fundamentalvision/Deformable-DETR",
        "deformable_detr_r50",
        pretrained=False,
        num_classes=91  # COCO默认类别数，不影响Person类检测
    )

    # 加载官方预训练权重（确保与模型匹配）
    weight_url = "https://dl.fbaipublicfiles.com/deformable_detr/deformable_detr-r50-dc511b7b.pth"
    checkpoint = load_state_dict_from_url(weight_url, progress=True)
    model.load_state_dict(checkpoint["model"])

    # 冻结参数+eval模式（与DETR-R50加载逻辑统一）
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

# -----------------------
# 3. 黑盒模型2：sparse-DETR（《BlackBox》论文指定）
# -----------------------

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
    sys.path.insert(0, sparse_detr_path)
    sys.path.insert(0, os.path.join(sparse_detr_path, "models", "ops"))

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

    # === 导入模型构建函数 ===
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