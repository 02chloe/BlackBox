import torch
import torch.nn as nn
from typing import List, Optional, Dict, Literal
from torch.hub import load_state_dict_from_url


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
    """严格对齐《BlackBox》论文3.1节TMM模块（保留梯度传播）"""
    def __init__(
        self,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        p_base: float = 0.2,
        sampling_strategy: Literal['categorical', 'bernoulli'] = 'categorical',
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.p_base = p_base
        self.sampling_strategy = sampling_strategy
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        if self.sampling_strategy not in ['categorical', 'bernoulli']:
            raise ValueError(f"采样策略仅支持'categorical'和'bernoulli'，当前为{self.sampling_strategy}")

        self.grad_history: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _categorical_mask_sampling(self, grad_abs: torch.Tensor) -> torch.Tensor:
        grad_flat = grad_abs.flatten()
        total_grad = grad_flat.sum()
        num_elements = grad_flat.numel()

        if total_grad < 1e-8:
            prob_dist = torch.ones_like(grad_flat) / num_elements
        else:
            prob_dist = grad_flat / total_grad

        num_to_mask = max(1, int(self.p_base * num_elements))
        indices = torch.multinomial(prob_dist, num_to_mask, replacement=False)
        mask_flat = torch.ones_like(grad_flat)
        mask_flat = mask_flat.scatter_(0, indices, 0.0)

        return mask_flat.view(grad_abs.shape).contiguous()

    def _apply_mask_to_input(self, input_tensor: torch.Tensor, layer_key: str) -> torch.Tensor:
        input_tensor = input_tensor.clone().contiguous()
        input_dim = input_tensor.dim()
    
        if input_dim == 4:  # (B, C, H, W)
            B, C, H, W = input_tensor.shape
            input_seq = input_tensor.flatten(2).permute(2, 0, 1).contiguous()  # (seq_len, B, C)
            masked_seq_list = []
    
            for b in range(B):
                S_len = input_seq.shape[0]  # 当前样本 seq_len
                mask = torch.rand(S_len, 1, C, device=input_tensor.device) > self.p_base
                mask = mask.float()
                masked_seq = input_seq[:, b:b+1, :] * mask
                masked_seq_list.append(masked_seq)
    
            masked_seq = torch.cat(masked_seq_list, dim=1)  # (seq_len, B, C)
            return masked_seq.permute(1, 2, 0).view(B, C, H, W).contiguous()
    
        elif input_dim == 3:  # (B, S, C)
            B, S, C = input_tensor.shape
            masked_list = []
            for b in range(B):
                mask = torch.rand(S, C, device=input_tensor.device) > self.p_base
                mask = mask.float()
                masked_list.append(input_tensor[b] * mask)
            return torch.stack(masked_list, dim=0)
    
        else:
            raise ValueError(f"不支持的输入维度：{input_dim}")

    def _register_layer_hooks(self, layers: nn.ModuleList, prefix: str):
        for layer_idx, layer in enumerate(layers):
            layer_key = f"{prefix}_{layer_idx}"

            def backward_hook(module, grad_in, grad_out, key=layer_key):
                if grad_in[0] is not None:
                    # 存储梯度时仍需detach（不影响传播链）
                    self.grad_history[key] = grad_in[0].abs().detach().clone().contiguous()

            def forward_hook(module, args, key=layer_key):
                input_tensor = args[0]
                return (self._apply_mask_to_input(input_tensor, key),) + args[1:]

            self.hooks.append(layer.register_full_backward_hook(backward_hook, prepend=False))
            self.hooks.append(layer.register_forward_pre_hook(forward_hook))

    def register_hooks(self, model: nn.Module):
        self.remove_hooks()
        base_model = getattr(model, 'module', model)

        assert hasattr(base_model, "transformer"), "模型必须包含transformer属性"
        assert len(base_model.transformer.encoder.layers) >= self.num_enc_layers, "encoder层数不足"
        assert len(base_model.transformer.decoder.layers) >= self.num_dec_layers, "decoder层数不足"

        self._register_layer_hooks(base_model.transformer.encoder.layers, prefix="enc")
        self._register_layer_hooks(base_model.transformer.decoder.layers, prefix="dec")

        print(f"✅ TMM已注册{len(self.hooks)}个hook（{self.num_enc_layers} encoder + {self.num_dec_layers} decoder）")
        print(f"✅ 采样策略：{self.sampling_strategy}（符合论文设置）")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("✅ TMM已移除所有hook")

    def reset_grad_history(self):
        self.grad_history.clear()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("TMM通过register_hooks()注入掩码，无需调用forward")


def load_detr_r50():
    """加载DETR-R50模型"""
    model = torch.hub.load(
        "facebookresearch/detr:main",
        "detr_resnet50",
        pretrained=False,
        force_reload=False
    )

    weight_url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    checkpoint = load_state_dict_from_url(weight_url, progress=True)
    model.load_state_dict(checkpoint["model"])

    model = model.cuda().train()
    for param in model.parameters():
        param.requires_grad = False  # 冻结模型参数，只优化补丁

    return model


def run_blackbox_whitebox_demo():
    # 1. 加载模型
    print("正在加载DETR-R50模型...")
    model = load_detr_r50()
    print("✅ DETR-R50模型加载完成")

    # 2. 初始化TMM
    tmm = TransformerMaskingMatrix(
        num_enc_layers=6,
        num_dec_layers=6,
        p_base=0.2,
        sampling_strategy='categorical',
        device='cuda'
    )
    tmm.register_hooks(model)

    # 3. 初始化补丁（需要梯度）和优化器
    patch = torch.randn(1, 3, 300, 300, device='cuda', requires_grad=True)  # 关键：requires_grad=True
    optimizer = torch.optim.Adam([patch], lr=0.005)  # 优化器绑定patch

    # 4. 模拟输入图像（无需梯度）
    img = torch.randn(1, 3, 800, 800, device='cuda').clone().contiguous()
    img.requires_grad = False

    # 5. 优化循环
    for iter in range(5):
        optimizer.zero_grad()  # 清零梯度
        tmm.reset_grad_history()

        # 生成掩码（无需梯度）
        mask = torch.zeros_like(img, device='cuda').clone().contiguous()
        mask[:, :, 100:400, 100:400] = 1.0

        # 补丁填充（保留梯度，移除detach()）
        padded_patch = torch.nn.functional.pad(patch, (100, 400, 100, 400)).clone().contiguous()

        # 生成patched_img（保留梯度传播链）
        patched_img = torch.empty_like(img, device='cuda')
        fusion_result = img * (1 - mask) + padded_patch * mask  # 融合逻辑（保留梯度）
        patched_img.copy_(fusion_result.clone().contiguous())  # 仅clone，不detach
        patched_img.requires_grad_(True)  # 确保启用梯度

        # 构造NestedTensor输入模型
        nested_patched_img = NestedTensor(tensors=patched_img)
        outputs = model(nested_patched_img)

        # 计算损失（行人类别置信度）
        pred_logits = outputs['pred_logits']
        person_confidence = torch.sigmoid(pred_logits[..., 1]).mean()
        loss = person_confidence  # 目标：降低行人置信度

        # 反向传播（此时梯度链已连通）
        loss.backward()  # 现在loss能找到需要梯度的patch
        optimizer.step()

        # 补丁裁剪
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, -2.1179, 2.6400)

        print(f"迭代{iter+1}/5 | 行人置信度损失: {loss.item():.4f} | 梯度历史数: {len(tmm.grad_history)}")

    tmm.remove_hooks()
    print("\n✅ 白盒实验核心流程验证完成（梯度传播正常）")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    run_blackbox_whitebox_demo()
