# 文件名：inria_dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json

class INRIAPersonDataset(Dataset):
    """INRIA Person数据集加载器（严格对齐论文训练设置）"""
    def __init__(self, data_root, split="Train", augment=True, disable_random_aug=False):
        self.split = split
        self.augment = augment
        self.disable_random_aug = disable_random_aug
        # 加载解析后的标注
        self.annotations = json.load(open(os.path.join(data_root, f"inria_{split}_annotations.json"), "r"))
        # 论文数据增强流水线（图2隐含操作）
        self.transform = self._build_transform()

    def _build_transform(self):
        base_transform = [T.Resize((640, 640)),  # 论文隐含输入尺寸（适配DETR）
                         T.ToTensor()]
        # 判断是否禁用增强
        if not self.disable_random_aug and self.augment and self.split == "Train":
            augment_transform = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ]
            return T.Compose(augment_transform + base_transform)
        else:
            return T.Compose(base_transform)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        # 加载图像
        img = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = self.transform(img)
        # 加载边界框（仅训练时用于辅助观察，论文攻击损失不依赖标注）
        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        return img_tensor, boxes

def get_inria_dataloader(data_root, split="Train", batch_size=8, num_workers=1, disable_random_aug=False):
    """获取数据加载器（论文4.1节：batch_size=8）"""
    dataset = INRIAPersonDataset(data_root, split=split, augment=(split=="Train"),
                                 disable_random_aug=disable_random_aug)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=="Train"),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda x: (torch.stack([i[0] for i in x]), [i[1] for i in x])
    )
