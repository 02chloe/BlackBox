# soft_tmm.py
# 在原始 tmm.py 基础上改为“软掩码”版本：
# - 接口整体风格保持一致（类名、注册方式、调用方式一致）
# - 但参数命名与硬 TMM 做语义区分：
#     * 硬 TMM: p_base 表示“丢弃概率”
#     * Soft TMM: soft_mask_ratio 表示“最大衰减比例（最多乘到 1-soft_mask_ratio）”
# - 软掩码：不再 0/1 丢弃，而是用基于梯度的连续缩放

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Literal
from utils.load_model import load_detr_r50


class NestedTensor:
    """匹配DETR的NestedTensor属性（复数tensors）"""
    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.tensors = tensors  # 复数属性，匹配DETR调用
        self.mask = mask if mask is not None else torch.zeros(
            (tensors.shape[0], tensors.shape[2], tensors.shape[3]),
            dtype=torch.bool,
            device=tensors.device
        )

    def decompose(self):
        return self.tensors, self.mask

    @property
    def device(self):
        return self.tensors.device


class TransformerMaskingMatrix(nn.Module):
    """
    Soft-TMM：与《BlackBox》论文3.1节 TMM 模块保持相同的整体结构，但改为“软掩码”。
    
    参数对比说明：
    - num_enc_layers / num_dec_layers / sampling_strategy / device：
        * 与硬 TMM 语义一致，仅控制 hook 位置和（预留的）采样策略
    - soft_mask_ratio（⚠ 新名字）：
        * 取代硬 TMM 的 p_base，语义改为“最大衰减比例”
        * 掩码 M ∈ [1 - soft_mask_ratio, 1]
    - tau：
        * Soft-TMM 新增温度参数，控制梯度重要性映射的“陡峭程度”
    """
    def __init__(
        self,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        soft_mask_ratio: float = 0.3,  # ⚠ 新名字：最大衰减比例，取代硬 TMM 的 p_base
        sampling_strategy: Literal['categorical', 'bernoulli'] = 'categorical',
        device: Optional[torch.device] = None,
        tau: float = 1.0,  # Soft-TMM 独有：温度参数（控制重要性映射的陡峭程度）
    ):
        super().__init__()
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.soft_mask_ratio = soft_mask_ratio
        self.sampling_strategy = sampling_strategy
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.tau = tau

        if self.sampling_strategy not in ['categorical', 'bernoulli']:
            raise ValueError(f"采样策略仅支持'categorical'和'bernoulli'，当前为{self.sampling_strategy}")

        # 记录每一层的梯度 |grad|（供下一次前向时生成 soft mask）
        self.grad_history: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # 保留原函数定义（目前不在 soft 模式下使用，仅作兼容占位）
    def _categorical_mask_sampling(self, grad_abs: torch.Tensor) -> torch.Tensor:
        """
        说明：
        - 对应硬 TMM 中的“按梯度分布进行采样丢弃”的接口
        - soft_tmm 中默认不调用此函数，仅保留以方便未来扩展/兼容
        - 这里仍然使用 soft_mask_ratio 作为比例，语义为“最多衰减/丢弃比例”
        """
        grad_flat = grad_abs.flatten()
        total_grad = grad_flat.sum()
        num_elements = grad_flat.numel()

        if total_grad < 1e-8:
            prob_dist = torch.ones_like(grad_flat) / num_elements
        else:
            prob_dist = grad_flat / total_grad

        num_to_select = max(1, int(self.soft_mask_ratio * num_elements))
        indices = torch.multinomial(prob_dist, num_to_select, replacement=False)
        mask_flat = torch.ones_like(grad_flat)
        # 这里仍然是“置 0”的形式，但通常不会在 soft_tmm 流程中被调用
        mask_flat = mask_flat.scatter_(0, indices, 0.0)

        return mask_flat.view(grad_abs.shape).contiguous()

    def _apply_mask_to_input(self, input_tensor: torch.Tensor, layer_key: str) -> torch.Tensor:
        """
        Soft 掩码版本：
        - 从 grad_history 中取出该层的 |grad|（与 input_tensor 同shape）
        - 对梯度做空间/序列维度聚合 + 全局归一化
        - 通过 sigmoid(g_norm / tau) 得到重要性 p ∈ (0,1)
        - 掩码 M = 1 - soft_mask_ratio * p ∈ [1 - soft_mask_ratio, 1]
        - 返回 input_tensor * M
        """
        input_tensor = input_tensor.clone().contiguous()
        input_dim = input_tensor.dim()

        # 如果没有梯度历史（例如第一个 step），则不做缩放
        if layer_key not in self.grad_history:
            return input_tensor

        grad_abs = self.grad_history[layer_key]
        # 防止形状不匹配（比如中途修改过网络结构），直接跳过
        if grad_abs.shape != input_tensor.shape:
            return input_tensor

        # 1. 聚合梯度到“空间/序列”维度上
        #    4D: (B, C, H, W) → (B, 1, H, W)
        #    3D: (B, S, C)    → (B, S, 1)
        if input_dim == 4:
            # (B, C, H, W)
            g = grad_abs.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            # 标准化
            mean = g.mean()
            std = g.std()
            g_norm = (g - mean) / (std + 1e-6)
            # 通过温度缩放 + sigmoid 得到重要性概率 p
            p = torch.sigmoid(g_norm / self.tau)  # [B,1,H,W]
            # 扩展到通道维度
            p = p.expand(-1, input_tensor.shape[1], -1, -1)  # [B,C,H,W]
            # 软掩码：M = 1 - soft_mask_ratio * p
            M = 1.0 - self.soft_mask_ratio * p
            return input_tensor * M

        elif input_dim == 3:
            # (B, S, C)
            g = grad_abs.abs().mean(dim=-1, keepdim=True)  # [B,S,1]
            mean = g.mean()
            std = g.std()
            g_norm = (g - mean) / (std + 1e-6)
            p = torch.sigmoid(g_norm / self.tau)          # [B,S,1]
            p = p.expand(-1, -1, input_tensor.shape[-1])  # [B,S,C]
            M = 1.0 - self.soft_mask_ratio * p
            return input_tensor * M

        else:
            raise ValueError(f"不支持的输入维度：{input_dim}")

    def _register_layer_hooks(self, layers: nn.ModuleList, prefix: str):
        for layer_idx, layer in enumerate(layers):
            layer_key = f"{prefix}_{layer_idx}"

            def backward_hook(module, grad_in, grad_out, key=layer_key):
                if grad_in[0] is not None:
                    # 存储梯度时仍需 detach（不影响传播链）
                    # grad_in[0] 形状与 forward_hook 中的 input_tensor 一致
                    self.grad_history[key] = grad_in[0].abs().detach().clone().contiguous()

            def forward_hook(module, args, key=layer_key):
                input_tensor = args[0]
                # 在前向时根据上一 step 的梯度为该层生成 soft mask
                masked_input = self._apply_mask_to_input(input_tensor, key)
                return (masked_input,) + args[1:]

            self.hooks.append(layer.register_full_backward_hook(backward_hook, prepend=False))
            self.hooks.append(layer.register_forward_pre_hook(forward_hook))

    def register_hooks(self, model: nn.Module):
        """
        与硬 TMM 相同的使用方式：
        - 传入完整 DETR 模型
        - 在 transformer.encoder/decoder 的前 num_enc_layers / num_dec_layers 层注册 hook
        """
        self.remove_hooks()
        base_model = getattr(model, 'module', model)

        assert hasattr(base_model, "transformer"), "模型必须包含transformer属性"
        assert len(base_model.transformer.encoder.layers) >= self.num_enc_layers, "encoder层数不足"
        assert len(base_model.transformer.decoder.layers) >= self.num_dec_layers, "decoder层数不足"

        self._register_layer_hooks(base_model.transformer.encoder.layers, prefix="enc")
        self._register_layer_hooks(base_model.transformer.decoder.layers, prefix="dec")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_grad_history(self):
        self.grad_history.clear()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TMM通过 register_hooks() 注入掩码，无需直接调用 forward")


def run_blackbox_whitebox_demo():
    """
    简单白盒 demo：
    - 验证 soft_tmm 与 DETR 的梯度链路是通的
    - 不涉及 BlackBoxLoss / GSE，只看能否正常 backward
    """
    print("正在加载DETR-R50模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detr_r50(device=device)
    print("✅ DETR-R50模型加载完成")

    # 初始化 Soft TMM
    tmm = TransformerMaskingMatrix(
        num_enc_layers=6,
        num_dec_layers=6,
        soft_mask_ratio=0.2,        # ⚠ Soft-TMM 的“最大衰减比例”
        sampling_strategy='categorical',
        device=device,
        tau=1.0
    )
    tmm.register_hooks(model)

    # 初始化补丁（需要梯度）和优化器
    patch = torch.randn(1, 3, 300, 300, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.005)

    # 模拟输入图像（无需梯度）
    img = torch.randn(1, 3, 800, 800, device=device).clone().contiguous()
    img.requires_grad = False

    # 优化循环：只是验证 soft_tmm 正常工作
    for iter in range(5):
        optimizer.zero_grad()
        tmm.reset_grad_history()

        # 生成掩码（这里是图像层面的 patch mask，和 TMM 无关）
        mask = torch.zeros_like(img, device=device).clone().contiguous()
        mask[:, :, 100:400, 100:400] = 1.0

        # 补丁填充
        padded_patch = torch.nn.functional.pad(patch, (100, 400, 100, 400)).clone().contiguous()

        # 生成 patched_img（保留梯度传播链）
        patched_img = torch.empty_like(img, device=device)
        fusion_result = img * (1 - mask) + padded_patch * mask
        patched_img.copy_(fusion_result.clone().contiguous())
        patched_img.requires_grad_(True)

        # 构造 NestedTensor 输入模型
        nested_patched_img = NestedTensor(tensors=patched_img)
        outputs = model(nested_patched_img)

        # 计算损失（行人类别置信度）
        pred_logits = outputs['pred_logits']
        person_confidence = torch.sigmoid(pred_logits[..., 1]).mean()
        loss = person_confidence  # 目标：降低行人置信度

        loss.backward()
        optimizer.step()

        # 补丁裁剪
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, -2.1179, 2.6400)

        print(f"迭代{iter+1}/5 | 行人置信度损失: {loss.item():.4f} | 梯度历史层数: {len(tmm.grad_history)}")

    tmm.remove_hooks()
    print("\n✅ Soft-TMM 白盒实验核心流程验证完成（梯度传播正常）")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    run_blackbox_whitebox_demo()
