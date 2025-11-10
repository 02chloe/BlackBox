# augment/patch_cutout.py
import torch, random

class PatchCutout:
    """
    T-SEA patch cutout module.
    Randomly masks part of the adversarial patch with probability φ_c.
    Compatible with autograd (no in-place ops).
    """
    def __init__(self, phi_c=0.3, size_ratio=0.3, fill_val=0.5):
        self.phi_c = phi_c
        self.size_ratio = size_ratio
        self.fill_val = fill_val

    def __call__(self, patch, prob=None):
        p = self.phi_c if prob is None else prob
        if random.random() > p:
            return patch

        own_batch = patch.dim() == 4
        if not own_batch:
            patch = patch.unsqueeze(0)
        B, C, H, W = patch.shape

        cutout_size = int(min(H, W) * self.size_ratio)
        cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
        x0, y0 = max(0, cx - cutout_size // 2), max(0, cy - cutout_size // 2)
        x1, y1 = min(W, cx + cutout_size // 2), min(H, cy + cutout_size // 2)

        # 使用掩码进行非原地操作
        mask = torch.ones_like(patch)
        mask[:, :, y0:y1, x0:x1] = 0.0
        fill_tensor = torch.full_like(patch, self.fill_val)

        patch = patch * mask + fill_tensor * (1 - mask)

        if not own_batch:
            patch = patch.squeeze(0)
        return patch
