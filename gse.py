# /opt/data/private/BlackBox/gse.py
import torch
import torch.nn as nn
from typing import List, Optional

class GradientSelfEnsemble:
    """修正版：适配DETR的维度顺序与解码器层属性名差异，并兼容 NestedTensor 输入"""
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self._last_captured_layers: List[torch.Tensor] = []
        self._hooks = []

    def _find_decoder_layers(self) -> List[nn.Module]:
        """适配：同时检查'layers'（复数）和'layer'（单数）属性"""
        if hasattr(self.model, 'transformer'):
            transformer = getattr(self.model, 'transformer')
            if hasattr(transformer, 'decoder'):
                dec = getattr(transformer, 'decoder')
                # 优先检查'layers'，再检查'layer'（适配不同DETR版本）
                for attr in ['layers', 'layer']:
                    if hasattr(dec, attr):
                        layers = getattr(dec, attr)
                        # 转换为列表（处理ModuleList或list）
                        return list(layers) if isinstance(layers, (nn.ModuleList, list, tuple)) else [layers]
                # 兜底：遍历解码器子模块，找到所有层（命名含'layer'）
                decoder_layers = []
                for name, child in dec.named_children():
                    if 'layer' in name.lower():
                        decoder_layers.append(child)
                if decoder_layers:
                    return decoder_layers
        # 若所有方法都找不到，报错
        raise RuntimeError(
            "无法在model.transformer.decoder中找到解码器层（检查'layers'或'layer'属性）。"
            "请使用官方DETR模型，或修改模型以暴露解码器层。"
        )

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self._last_captured_layers = []

    def _register_decoder_hooks(self, decoder_layers: List[nn.Module]):
        self._clear_hooks()
        captured = []
        def make_hook(i):
            def hook(module, inp, out):
                # 提取输出张量（处理tuple/list）
                o = out[0] if isinstance(out, (tuple, list)) else out
                captured.append(o)
            return hook
        for idx, layer_module in enumerate(decoder_layers):
            h = layer_module.register_forward_hook(make_hook(idx))
            self._hooks.append(h)
        self._last_captured_layers = captured

    def call_model_and_get_all_layer_logits(self, imgs_list, return_mean: bool = False):
        """
        imgs_list: can be
           - list of torch.Tensor images (each [3,H,W]) -> stack into [B,3,H,W]
           - torch.Tensor batch [B,3,H,W]
           - NestedTensor instance (with .tensors and .mask) from tmm.py
        return_mean: if True, return mean logits across layers (i.e. [B,Q,C]); otherwise return [L,B,Q,C]
        """
        decoder_layers = self._find_decoder_layers()
        self._register_decoder_hooks(decoder_layers)

        # === Normalize input for model invocation ===
        is_nested = False
        batch_size = None

        # If input is list of tensors
        if isinstance(imgs_list, list) and all(isinstance(x, torch.Tensor) for x in imgs_list):
            imgs_tensor = torch.stack(imgs_list, dim=0).to(self.device)  # [B,3,H,W]
            batch_size = imgs_tensor.shape[0]
            model_input = imgs_tensor
        # If it's already a torch.Tensor (batch)
        elif isinstance(imgs_list, torch.Tensor):
            imgs_tensor = imgs_list.to(self.device)
            batch_size = imgs_tensor.shape[0]
            model_input = imgs_tensor
        else:
            # Could be NestedTensor or other object expected by DETR
            # Detect by duck-typing: has attribute 'tensors'
            if hasattr(imgs_list, 'tensors'):
                # treat as NestedTensor-like (we pass it directly to model)
                is_nested = True
                model_input = imgs_list  # pass as-is (expected by DETR.forward)
                # determine batch size from underlying tensors if possible
                try:
                    batch_size = imgs_list.tensors.shape[0]
                except Exception:
                    # fallback: None
                    batch_size = None
            else:
                # last resort: try to convert to tensor
                try:
                    imgs_tensor = torch.stack(list(imgs_list), dim=0).to(self.device)
                    batch_size = imgs_tensor.shape[0]
                    model_input = imgs_tensor
                except Exception as e:
                    raise RuntimeError("Unsupported imgs_list type for GSE: must be list/tensor/NestedTensor") from e

        # call model (pass NestedTensor if that's what DETR expects)
        out = self.model(model_input)

        captured = self._last_captured_layers
        self._clear_hooks()

        if len(captured) == 0:
            raise RuntimeError("未捕获到解码器层输出")

        # captured is a list of tensors (each layer output). Normalize them to [B,Q,D]
        normalized = []
        for t in captured:
            # t expected shape either [Q, B, D] or [B, Q, D] or possibly [B, Q, D] with Q=100 as in DETR
            if not isinstance(t, torch.Tensor):
                raise RuntimeError("捕获到的解码器输出不是tensor类型")
            if t.dim() != 3:
                raise RuntimeError(f"解码器输出维度错误：预期3维，实际{t.dim()}维")
            # try to detect ordering: if first dim equals queries (commonly 100) and second equals batch
            # but we can't assume batch dimension from model_input in all cases; use determined batch_size when available
            # If batch_size is known and t.shape[1] == batch_size and t.shape[0] != batch_size -> assume [Q,B,D]
            if batch_size is not None and t.shape[1] == batch_size and t.shape[0] != batch_size:
                # common DETR case: [Q,B,D] -> transpose
                t_proc = t.transpose(0, 1).contiguous()  # -> [B,Q,D]
            else:
                # otherwise assume it's already [B,Q,D]
                t_proc = t.contiguous()
            normalized.append(t_proc)

        # apply class_embed to each layer output to get logits [B,Q,C]
        logits_per_layer = []
        for layer_out in normalized:
            try:
                logits = self.model.class_embed(layer_out)  # [B,Q,C]
            except Exception:
                # some models may expect different ordering; try transpose fallback
                logits = self.model.class_embed(layer_out.transpose(0,1)).transpose(0,1)
            logits_per_layer.append(logits)

        all_logits = torch.stack(logits_per_layer, dim=0)  # [L, B, Q, C]

        if return_mean:
            return all_logits.mean(dim=0)  # [B,Q,C]
        return all_logits  # [L,B,Q,C]

    def __call__(self, imgs_list, return_all_layers: bool = False):
        # return_all_layers True -> return [L,B,Q,C]; False -> mean across layers [B,Q,C]
        return self.call_model_and_get_all_layer_logits(imgs_list, return_mean=(not return_all_layers))
