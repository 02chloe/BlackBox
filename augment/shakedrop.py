# augment/shakedrop.py
import torch
import torch.nn as nn

class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=(0.0, 1.0)):
        if not training:
            return (1 - p_drop) * x
        gate = torch.bernoulli(torch.tensor(1 - p_drop)).to(x.device)
        alpha = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*alpha_range)
        ctx.save_for_backward(gate, alpha)
        return gate * x + alpha * (1 - gate) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate, alpha = ctx.saved_tensors
        beta = torch.empty_like(alpha).uniform_(-1, 1)
        grad_input = gate * grad_output + beta * (1 - gate) * grad_output
        return grad_input, None, None, None

class ShakeDrop(nn.Module):
    """可插入CNN层的ShakeDrop正则模块"""
    def __init__(self, p_drop=0.5, alpha_range=(0.0, 1.0)):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)
