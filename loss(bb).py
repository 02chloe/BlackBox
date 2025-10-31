# 文件: loss.py
# 目的：实现 BlackBox 论文风格的攻击损失（兼容 GSE + TMM）
# 兼容路径: /opt/data/private/BlackBox/gse.py, /opt/data/private/BlackBox/tmm.py
# 设计原则：
#  - 用 GSE 提供的 decoder intermediate logits 作为检测打分来源
#  - 提供多种 layer 聚合策略（论文语义通常按层 avg）
#  - 提供 TV / L2 正则化（常用于补丁平滑与幅度控制）；保留 NPS 钩子以便外部实现
#  - 训练时，应只优化补丁参数（模型参数冻结），调用 loss.backward() 会把梯度传回 patch
import torch
import torch.nn.functional as F
from typing import Optional, Callable, List, Dict
from tmm import NestedTensor

class BlackBoxLoss:
    """
    BlackBox-style loss for adversarial patch optimization.

    主要组件（可配置）：
      - detection_weight: 检测目标损失权重（论文主目标）
      - tv_weight: Total Variation 正则项权重（平滑）
      - l2_weight: L2 约束权重（限制补丁幅度）
      - target_class: 目标类别索引（例如论文目标: 行人类 index，DETR里可能是1）
      - layer_aggregation: 聚合中间层输出策略，支持:
          * 'mean_logits' - 对各层 logits 做平均后再算 softmax/sigmoid -> loss
          * 'mean_prob'   - 先对每层 logits 做 softmax/sigmoid 得到 prob，再对 prob 求均值 -> loss
          * 'per_layer_loss' - 对每层单独计算 loss，最后平均这些 loss（最贴近论文“对每层求和/均值”的做法）
      - use_sigmoid_for_binary: 对单一目标（binary 目标如 person）是否用 sigmoid；DETR通常多分类，若class数>2建议用 softmax
      - reduction: 'mean' or 'sum' for loss aggregation over batch
    """

    def __init__(
        self,
        gse,                         # GradientSelfEnsemble 实例或任意 callable that returns logits
        target_class: int = 1,
        detection_weight: float = 1.0,
        tv_weight: float = 1e-3,
        l2_weight: float = 1e-3,
        layer_aggregation: str = 'per_layer_loss',  # 'mean_logits' | 'mean_prob' | 'per_layer_loss'
        use_sigmoid_for_binary: bool = True,
        reduction: str = 'mean',
        nps_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # 非可打印性分数函数 (可选)
        device: Optional[torch.device] = None
    ):
        self.gse = gse
        self.target_class = target_class
        self.detection_weight = float(detection_weight)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)
        self.layer_aggregation = layer_aggregation
        self.use_sigmoid_for_binary = use_sigmoid_for_binary
        self.reduction = reduction
        self.nps_fn = nps_fn
        self.device = device if device is not None else next(gse.model.parameters()).device

        assert layer_aggregation in ('mean_logits', 'mean_prob', 'per_layer_loss'), \
            "layer_aggregation must be one of 'mean_logits','mean_prob','per_layer_loss'"

    # --------------------------
    # detection loss helpers
    # --------------------------
    @staticmethod
    def _softmax_probs_over_classes(logits: torch.Tensor):
        # logits shape: [..., num_classes]
        return F.softmax(logits, dim=-1)

    @staticmethod
    def _sigmoid_probs_over_classes(logits: torch.Tensor):
        return torch.sigmoid(logits)

    def _compute_detection_loss_from_logits(
        self,
        logits_all_layers: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        logits_all_layers: [L, B, Q, C]   (L = decoder layers)
        返回标量 loss（对 batch 聚合，按 reduction）
        论文实现细节：对所有层进行聚合以得到最终判分 / loss。这里提供三种聚合策略（可配置）。
        """

        if target_class is None:
            target_class = self.target_class

        L, B, Q, C = logits_all_layers.shape

        # Strategy 1: mean_logits -> average logits across layers then compute prob on class
        if self.layer_aggregation == 'mean_logits':
            mean_logits = logits_all_layers.mean(dim=0)  # [B, Q, C]
            if self.use_sigmoid_for_binary and C == 1:
                probs = self._sigmoid_probs_over_classes(mean_logits).squeeze(-1)  # [B,Q]
                # For binary sigmoid case, target class = single logit (we assume positive)
                loss_per_query = probs[..., 0] if probs.dim() == 2 else probs  # safe
            else:
                probs = self._softmax_probs_over_classes(mean_logits)  # [B,Q,C]
                loss_per_query = probs[..., target_class]  # [B,Q]

            # want to minimize target_class probability (make model less confident)
            # create loss = mean(probabilities) so gradient pushes down the prob
            if self.reduction == 'mean':
                return loss_per_query.mean() * self.detection_weight
            else:
                return loss_per_query.sum() * self.detection_weight

        # Strategy 2: mean_prob -> per-layer probs then average
        elif self.layer_aggregation == 'mean_prob':
            # compute probs per layer then mean
            if self.use_sigmoid_for_binary and C == 1:
                probs = torch.sigmoid(logits_all_layers).squeeze(-1)  # [L,B,Q]
                target_probs = probs[..., 0]
            else:
                probs = F.softmax(logits_all_layers, dim=-1)  # [L,B,Q,C]
                target_probs = probs[..., target_class]  # [L,B,Q]
            mean_target_prob = target_probs.mean(dim=0)  # [B,Q]
            if self.reduction == 'mean':
                return mean_target_prob.mean() * self.detection_weight
            else:
                return mean_target_prob.sum() * self.detection_weight

        # Strategy 3: per_layer_loss -> compute loss per layer then average
        elif self.layer_aggregation == 'per_layer_loss':
            layer_losses = []
            for li in range(L):
                logits = logits_all_layers[li]  # [B,Q,C]
                if self.use_sigmoid_for_binary and C == 1:
                    probs = torch.sigmoid(logits).squeeze(-1)  # [B,Q]
                    target_probs = probs  # assumes single logit is positive class
                else:
                    probs = F.softmax(logits, dim=-1)  # [B,Q,C]
                    target_probs = probs[..., target_class]  # [B,Q]
                if self.reduction == 'mean':
                    layer_losses.append(target_probs.mean())
                else:
                    layer_losses.append(target_probs.sum())
            # average over layers
            loss = torch.stack(layer_losses).mean() * self.detection_weight
            return loss

        else:
            raise RuntimeError("未知的 layer_aggregation 策略")

    # --------------------------
    # regularizers
    # --------------------------
    @staticmethod
    def total_variation(image: torch.Tensor):
        """
        Total variation for a batch of image patches or images.
        Accepts tensor shape [B, C, H, W] or [1, C, H, W] (for a single patch padded into image).
        Returns scalar (mean over batch).
        """
        # use anisotropic TV (sum of abs of differences)
        if image.dim() != 4:
            raise ValueError("total_variation expects [B,C,H,W]")
        dh = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        dw = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        return (dh.mean() + dw.mean())

    @staticmethod
    def l2_norm(image: torch.Tensor):
        # L2 norm averaged over batch
        return torch.mean(image.pow(2))

    # --------------------------
    # 主接口：给定原始图像、patched_image 以及补丁张量，返回总loss与各子项
    # --------------------------
    def __call__(
        self,
        imgs: torch.Tensor,              # 原始batch images [B,3,H,W] or list->stacked
        patched_imgs: torch.Tensor,      # patched images [B,3,H,W] or already NestedTensor
        patch_tensor: Optional[torch.Tensor] = None,  # 裁剪/原补丁 [1,C,ph,pw]（可选，用于 TV/L2 正则）
        reduction: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算并返回损失字典：
          {
            'total_loss': Tensor(scalar),
            'det_loss': Tensor,
            'tv_loss': Tensor,
            'l2_loss': Tensor,
            'nps_loss': Tensor
          }
    
        注意：
        - 为避免 DETR 内部在 no_grad 路径中对 view 做 inplace copy 的问题，
          我们在传入模型前将 patched_imgs 包装为 NestedTensor（若尚未包装）。
        - patched_imgs 可以是 Tensor 或预先构造好的 NestedTensor；函数会兼容两者。
        """
        if reduction is None:
            reduction = self.reduction
    
        # --- Wrap patched images into NestedTensor to avoid DETR's internal inplace copy issue
        # Ensure images are on the correct device
        if isinstance(patched_imgs, torch.Tensor):
            # move to device and ensure contiguous
            patched_imgs = patched_imgs.to(self.device)
            if not patched_imgs.is_contiguous():
                patched_imgs = patched_imgs.contiguous()
            samples = NestedTensor(tensors=patched_imgs)
        else:
            # assume caller has provided a NestedTensor or compatible object
            samples = patched_imgs
    
        # 1) use GSE to compute logits across decoder layers
        # GSE is expected to accept the same type as model.forward (NestedTensor for DETR)
        logits_all = self.gse(samples, return_all_layers=True)  # expected shape [L, B, Q, C]
        logits_all = logits_all.to(self.device)
    
        # 2) detection loss (target class prob) computed from GSE logits
        det_loss = self._compute_detection_loss_from_logits(logits_all, target_class=self.target_class)
    
        # 3) regularization losses
        tv_loss = torch.tensor(0.0, device=self.device)
        l2_loss = torch.tensor(0.0, device=self.device)
        nps_loss = torch.tensor(0.0, device=self.device)
    
        # If patch_tensor provided, compute TV and L2 on the patch content
        if patch_tensor is not None:
            p = patch_tensor.to(self.device)
            # Normalize shape to [B, C, H, W] if needed
            if p.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                p_batch = p.unsqueeze(0)
            elif p.dim() == 4 and p.shape[0] == 1:
                p_batch = p
            elif p.dim() == 4 and p.shape[0] > 1:
                p_batch = p
            else:
                # fallback: try to unsqueeze
                p_batch = p.unsqueeze(0) if p.dim() == 3 else p
    
            # compute TV on patch region; fallback to whole image TV if fails
            try:
                tv_loss = self.total_variation(p_batch)
            except Exception:
                # As a fallback compute TV on a dummy expanded patched image region if available
                # Here we compute on p_batch anyway to avoid raising
                tv_loss = self.total_variation(p_batch)
    
            l2_loss = self.l2_norm(p_batch)
    
        # optional NPS (non-printability score) if function provided
        if self.nps_fn is not None and patch_tensor is not None:
            try:
                nps_loss = self.nps_fn(patch_tensor.to(self.device))
            except Exception:
                # If nps function fails, keep as zero but do not break optimization
                nps_loss = torch.tensor(0.0, device=self.device)
    
        # 4) combine losses
        total_loss = det_loss + self.tv_weight * tv_loss + self.l2_weight * l2_loss
        if self.nps_fn is not None:
            total_loss = total_loss + nps_loss
    
        # Respect reduction if needed (det_loss and others already aggregated in helper)
        return {
            'total_loss': total_loss,
            'det_loss': det_loss,
            'tv_loss': tv_loss,
            'l2_loss': l2_loss,
            'nps_loss': nps_loss
        }
    
        
