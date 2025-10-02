import torch
import torch.nn as nn
from typing import List, Optional

class GradientSelfEnsemble:
    """修正版：适配DETR的维度顺序与解码器层属性名差异"""
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

    def call_model_and_get_all_layer_logits(self, imgs_list: List[torch.Tensor], return_mean: bool = False):
        decoder_layers = self._find_decoder_layers()
        self._register_decoder_hooks(decoder_layers)
        
        # 核心修正：提前检查class_embed是否存在（在模型forward之前）
        if not hasattr(self.model, 'class_embed'):
            raise RuntimeError("模型缺少class_embed属性，无法生成分类logits。请使用含共享分类头的DETR模型。")
        
        # 处理模型输入：转为batch tensor
        if isinstance(imgs_list, list) and all(isinstance(x, torch.Tensor) for x in imgs_list):
            imgs_tensor = torch.stack(imgs_list, dim=0)  # [B, 3, H, W]
        else:
            imgs_tensor = imgs_list
        out = self.model(imgs_tensor)  # 此时已确保class_embed存在，避免模型内部报错
        
        captured = self._last_captured_layers
        self._clear_hooks()
        
        if len(captured) == 0:
            raise RuntimeError("未捕获到解码器层输出")
    
        # 维度转换（[Q, B, D] → [B, Q, D]）
        normalized = []
        for t in captured:
            if t.dim() != 3:
                raise RuntimeError(f"解码器输出维度错误：预期3维，实际{t.dim()}维")
            if t.shape[0] == 100 and t.shape[1] == imgs_tensor.shape[0]:  # Q=100，B=批量
                t = t.transpose(0, 1)  # [Q, B, D] → [B, Q, D]
            normalized.append(t)
    
        # 应用class_embed生成logits
        logits_per_layer = []
        for layer_out in normalized:
            try:
                logits = self.model.class_embed(layer_out)  # [B, Q, C]
            except:
                logits = self.model.class_embed(layer_out.transpose(0,1)).transpose(0,1)
            logits_per_layer.append(logits)
    
        all_logits = torch.stack(logits_per_layer, dim=0)  # [L, B, Q, C]
        if return_mean:
            return all_logits.mean(dim=0)
        return all_logits

    def __call__(self, imgs_list: List[torch.Tensor], return_all_layers: bool = False):
        return self.call_model_and_get_all_layer_logits(imgs_list, return_mean=(not return_all_layers))
