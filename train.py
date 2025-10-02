# /opt/data/private/BlackBox/train.py
import os
import math
import random
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, draw_bounding_boxes
from torchvision.ops import box_convert, nms
from torch.nn.functional import interpolate
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from inria_dataloader import get_inria_dataloader
# from tmm import TransformerMaskingMatrix, load_detr_r50, NestedTensor
from gse import GradientSelfEnsemble
from loss import BlackBoxLoss


# -----------------------
# Config (可调整)
# -----------------------
ROOT = "/opt/data/private/BlackBox"
DATA_ROOT = os.path.join(ROOT, "data", "INRIAPerson")
SAVE_DIR = os.path.join(ROOT, "save", "demo")
os.makedirs(SAVE_DIR, exist_ok=True)

# training params
BATCH_SIZE = 8
NUM_EPOCHS = 10           # 可根据需要放大（paper 使用更大迭代）
NUM_WORKERS = 4

# patch params
PATCH_SIDE = 300          # 固定 global patch side (严格对齐论文)
PATCH_INIT_STD = 0.5
MIN_PATCH_PX = 16         # fallback minimum when resizing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model / detection params
MODEL_INPUT_H, MODEL_INPUT_W = 640, 640  # dataloader resize (H, W)
TARGET_CLASS_IDX = 1
SCORE_THRESH = 0.5
FALLBACK_TO_TOP = True
FALLBACK_SCORE_THRESH = 0.2
IOU_NMS_THRESH = 0.5
MIN_BOX_SIDE = 5

# loss weights (默认基于论文设置，可调整)
DETECTION_WEIGHT = 1.0
TV_WEIGHT = 1e-3
NPS_WEIGHT = 0.0

# EoT / augmentation switches (简单实现)
USE_EOT = True
EOT_SCALE = (0.9, 1.1)
EOT_ROT_DEG = (-10, 10)
EOT_BRIGHT = (0.9, 1.1)
EOT_CONTRAST = (0.9, 1.1)

# reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Helpers
# -----------------------
def detach_cpu(img: torch.Tensor):
    """return CPU float tensor in 0..1"""
    return img.detach().cpu().clamp(0,1)

def draw_boxes_on_tensor(img_tensor: torch.Tensor, boxes_xyxy_cpu: torch.Tensor):
    """draw boxes (cpu tensor img 3,H,W)"""
    if boxes_xyxy_cpu is None or boxes_xyxy_cpu.numel() == 0:
        return img_tensor
    img_uint8 = (img_tensor * 255).byte()
    boxes = boxes_xyxy_cpu.clone()
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, W-1)
    boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, H-1)
    valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
    boxes = boxes[valid]
    if boxes.shape[0] == 0:
        return img_tensor
    boxes_int = boxes.to(torch.int64)
    img_boxes = draw_bounding_boxes(img_uint8, boxes=boxes_int, colors="red", width=2)
    return img_boxes.float() / 255.0

def detr_boxes_to_xyxy_pixel(pred_boxes):
    """
    pred_boxes: [Q,4] cx,cy,w,h (normalized 0..1 or absolute)
    returns [Q,4] xyxy in pixel coords (CPU tensor)
    """
    pb = pred_boxes.clone()
    if pb.max() <= 1.01:
        pb[:,0] = pb[:,0] * MODEL_INPUT_W
        pb[:,1] = pb[:,1] * MODEL_INPUT_H
        pb[:,2] = pb[:,2] * MODEL_INPUT_W
        pb[:,3] = pb[:,3] * MODEL_INPUT_H
    xyxy = box_convert(pb, in_fmt='cxcywh', out_fmt='xyxy')
    return xyxy.cpu()

def eot_transform_patch(patch_tensor: torch.Tensor):
    """
    patch_tensor: [1,3,H,W] on DEVICE
    apply small random scale / rotate / brightness / contrast
    returns transformed patch [1,3,h2,w2]
    """
    if not USE_EOT:
        return patch_tensor
    # to CPU PIL for some transforms, but to keep gradient chain we operate in tensor domain
    _, _, H, W = patch_tensor.shape
    # random scale
    scale = float(np.random.uniform(EOT_SCALE[0], EOT_SCALE[1]))
    new_side = max(1, int(round(PATCH_SIDE * scale)))
    p = interpolate(patch_tensor, size=(new_side, new_side), mode='bilinear', align_corners=False)
    # random rotate (use TF.affine which accepts tensor)
    angle = float(np.random.uniform(EOT_ROT_DEG[0], EOT_ROT_DEG[1]))
    # torchvision's functional.affine expects shape [...,H,W], supports tensors
    # We'll apply rotate around center (no translate, no shear)
    p = TF.affine(
        p,
        angle=angle,
        translate=[0, 0],
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,  # 替换 resample
        fill=0
        )
    # brightness & contrast by simple scale/add
    b = float(np.random.uniform(EOT_BRIGHT[0], EOT_BRIGHT[1]))
    c = float(np.random.uniform(EOT_CONTRAST[0], EOT_CONTRAST[1]))
    p = torch.clamp((p * c) * b, 0.0, 1.0)
    return p

def paste_patch_via_mask(base_img: torch.Tensor, patch_tensor: torch.Tensor, center_xy: tuple):
    """
    base_img: [3,H,W] float on device
    patch_tensor: [1,3,ph,pw] or [3,ph,pw] on same device
    center_xy: (cx, cy) pixel coords (float)
    returns new image [3,H,W] (device) with patch pasted (non-inplace, gradient-preserving)
    Implemented via mask fusion: out = base*(1-mask) + patch*mask
    """
    if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
        p = patch_tensor[0]
    elif patch_tensor.dim() == 3:
        p = patch_tensor
    else:
        raise ValueError("invalid patch shape")

    ph, pw = p.shape[1], p.shape[2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = cx - pw // 2
    y0 = cy - ph // 2

    H, W = base_img.shape[1], base_img.shape[2]

    # compute crop ranges
    src_x0, src_y0 = 0, 0
    dst_x0, dst_y0 = x0, y0
    dst_x1, dst_y1 = x0 + pw, y0 + ph

    if dst_x0 < 0:
        src_x0 = -dst_x0; dst_x0 = 0
    if dst_y0 < 0:
        src_y0 = -dst_y0; dst_y0 = 0
    if dst_x1 > W:
        dst_x1 = W
    if dst_y1 > H:
        dst_y1 = H

    out_w = dst_x1 - dst_x0
    out_h = dst_y1 - dst_y0
    if out_w <= 0 or out_h <= 0:
        return base_img.clone()

    src_x1 = src_x0 + out_w
    src_y1 = src_y0 + out_h
    p_cropped = p[:, src_y0:src_y1, src_x0:src_x1]

    # create mask shaped [3,H,W], zeros then set box area to 1
    mask = torch.zeros_like(base_img)
    mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = 1.0

    # build padded_patch with same H,W by padding p_cropped to correct location
    padded_patch = torch.zeros_like(base_img)
    padded_patch[:, dst_y0:dst_y1, dst_x0:dst_x1] = p_cropped

    # fusion (non-inplace)
    fused = base_img * (1.0 - mask) + padded_patch * mask
    return fused

# -----------------------
# Data and model init
# -----------------------
dataloader = get_inria_dataloader(DATA_ROOT, split="Train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, disable_random_aug=False)
print("Train dataset size:", len(dataloader.dataset))

model = load_detr_r50().to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# TMM: register once and keep enabled for entire training (严格按论文)
tmm = TransformerMaskingMatrix(num_enc_layers=6, num_dec_layers=6, p_base=0.2, sampling_strategy='categorical', device=DEVICE)
tmm.register_hooks(model)
tmm.reset_grad_history()

# GSE and loss
gse = GradientSelfEnsemble(model=model, device=DEVICE)
loss_fn = BlackBoxLoss(gse=gse, target_class=TARGET_CLASS_IDX,
                       detection_weight=DETECTION_WEIGHT, tv_weight=TV_WEIGHT, l2_weight=0.0,
                       layer_aggregation='per_layer_loss', use_sigmoid_for_binary=False,
                       device=DEVICE)

# initialize patch (requires_grad=True)
patch = torch.randn(1, 3, PATCH_SIDE, PATCH_SIDE, device=DEVICE) * PATCH_INIT_STD + 0.7
patch = patch.clamp(0.0, 1.0)
patch.requires_grad_(True)
optimizer = torch.optim.Adam([patch], lr=0.005)

print("Start training (TMM enabled during all forwards). Saving to:", SAVE_DIR)

# -----------------------
# Training loop (epoch)
# -----------------------
global_step = 0
for epoch in range(NUM_EPOCHS):
    model.eval()  # model remains eval (weights frozen)
    tmm.reset_grad_history()  # clear grad history at epoch start for stability

    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(DEVICE).clamp(0,1)
        B = imgs.shape[0]

        # --- STEP: obtain detections with TMM active (we keep hooks active per-paper)
        with torch.no_grad():
            try:
                det_out = model(imgs)
            except Exception:
                det_out = model(NestedTensor(imgs))

        batch_boxes_all = []
        for bi in range(B):
            logits = det_out['pred_logits'][bi]  # [Q,C]
            boxes = det_out['pred_boxes'][bi]    # [Q,4]
            probs = torch.softmax(logits, dim=-1)
            cls_scores = probs[..., TARGET_CLASS_IDX]  # [Q]

            keep_idx = (cls_scores > SCORE_THRESH).nonzero(as_tuple=False).squeeze(1) if (cls_scores > SCORE_THRESH).any() else torch.tensor([], dtype=torch.long, device=cls_scores.device)
            if keep_idx.numel() == 0 and FALLBACK_TO_TOP:
                top_score, top_idx = torch.max(cls_scores, dim=0)
                if top_score.item() >= FALLBACK_SCORE_THRESH:
                    keep_idx = top_idx.unsqueeze(0)
                else:
                    keep_idx = torch.tensor([], dtype=torch.long, device=cls_scores.device)

            if keep_idx.numel() == 0:
                batch_boxes_all.append(torch.empty((0,4), dtype=torch.float32))
                continue

            sel_boxes = boxes[keep_idx]
            sel_scores = cls_scores[keep_idx].detach()

            sel_xyxy = detr_boxes_to_xyxy_pixel(sel_boxes.detach().cpu())
            widths = (sel_xyxy[:,2] - sel_xyxy[:,0])
            heights = (sel_xyxy[:,3] - sel_xyxy[:,1])
            large_mask = (widths >= MIN_BOX_SIDE) & (heights >= MIN_BOX_SIDE)
            if large_mask.sum() == 0:
                batch_boxes_all.append(torch.empty((0,4), dtype=torch.float32))
                continue
            sel_xyxy = sel_xyxy[large_mask]
            sel_scores_cpu = sel_scores.detach().cpu()[large_mask]

            try:
                keep_nms = nms(sel_xyxy, sel_scores_cpu, IOU_NMS_THRESH)
            except Exception:
                keep_nms = nms(sel_xyxy.cpu(), sel_scores_cpu.cpu(), IOU_NMS_THRESH)
            sel_xyxy_nms = sel_xyxy[keep_nms]
            batch_boxes_all.append(sel_xyxy_nms)

        # --- STEP: build patched images (gradient-preserving fusion)
        # keep TMM hooks enabled during forward of patched imgs (already registered)
        patched = imgs.clone()
        for bi in range(B):
            sel_boxes_cpu = batch_boxes_all[bi]  # CPU [K,4]
            if sel_boxes_cpu.numel() == 0:
                continue
            sel_boxes_dev = sel_boxes_cpu.to(DEVICE)
            for box in sel_boxes_dev:
                xmin, ymin, xmax, ymax = box.tolist()
                box_w = max(int(xmax - xmin), 1)
                box_h = max(int(ymax - ymin), 1)
                short = min(box_w, box_h)
                # use fixed patch size but optionally scale a bit relative to short side
                scale = float(np.clip(short / PATCH_SIDE, 0.5, 2.0))  # relative scale
                # ensure at least MIN_PATCH_PX
                side = max(MIN_PATCH_PX, int(round(PATCH_SIDE * scale)))
                # apply EoT transforms to patch (returns [1,3,side,side])
                patch_to_paste = eot_transform_patch(patch)
                # resize transformed patch to desired side
                patch_resized = interpolate(patch_to_paste, size=(side, side), mode='bilinear', align_corners=False)
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                # fusion (non-inplace, gradient-preserving)
                patched[bi] = paste_patch_via_mask(patched[bi], patch_resized, center_xy=(cx, cy))

        # --- STEP: compute loss and update patch
        # ensure tmm.grad_history is available for GSE/TMM internal use
        loss_dict = loss_fn(imgs, patched, patch_tensor=patch)
        total_loss = loss_dict['total_loss']
        optimizer.zero_grad()
        total_loss.backward()

        # debug: inspect patch grad
        if patch.grad is None:
            print(f"[epoch {epoch+1} batch {batch_idx}] WARNING: patch.grad is None")
            grad_norm = None
        else:
            grad_norm = patch.grad.detach().cpu().norm().item()

        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)

        # debug print
        det_loss_v = loss_dict.get('det_loss', torch.tensor(0.0)).item() if isinstance(loss_dict.get('det_loss', 0.0), torch.Tensor) else float(loss_dict.get('det_loss', 0.0))
        tv_loss_v = loss_dict.get('tv_loss', torch.tensor(0.0)).item() if isinstance(loss_dict.get('tv_loss', 0.0), torch.Tensor) else float(loss_dict.get('tv_loss', 0.0))
        nps_loss_v = loss_dict.get('nps_loss', torch.tensor(0.0)).item() if isinstance(loss_dict.get('nps_loss', 0.0), torch.Tensor) else float(loss_dict.get('nps_loss', 0.0))
        print(f"[epoch {epoch+1} batch {batch_idx}] total_loss={total_loss.item():.6f} | det_loss={det_loss_v:.6f} | tv={tv_loss_v:.6f} | nps={nps_loss_v:.6f} | grad_norm={grad_norm} | selected_counts={[b.shape[0] for b in batch_boxes_all]}")

        # save occasional visual snapshots
        if global_step % 200 == 0:
            # orig, patched, patch
            save_image(detach_cpu(imgs[0]), os.path.join(SAVE_DIR, f"step_{global_step}_orig.png"))
            save_image(detach_cpu(patched[0]), os.path.join(SAVE_DIR, f"step_{global_step}_patched.png"))
            save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"step_{global_step}_patch.png"))
        global_step += 1

    # end epoch: save epoch patch
    save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.png"))
    torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_patch.pt"))
    print(f"Epoch {epoch+1} saved patch snapshot.")

# cleanup & final save
tmm.remove_hooks()
save_image(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.png"))
torch.save(patch[0].detach().cpu(), os.path.join(SAVE_DIR, "final_patch.pt"))
print("Done. Final patch saved to", SAVE_DIR)
