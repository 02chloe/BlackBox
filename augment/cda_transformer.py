# augment/cda_transformer.py
import torch
import torch.nn.functional as F
import math
import kornia
import kornia.augmentation as K

class DataTransformer(torch.nn.Module):
    """T-SEA style Constrained Data Augmentation (CDA) based on Kornia."""
    def __init__(self, device: torch.device,
                 rand_rotate: int = 10,
                 rand_zoom_in: float = 0.3,
                 rand_brightness: float = 0.2,
                 rand_saturation: float = 0.3,
                 rand_shift: float = 0.3):
        super().__init__()
        self.device = device
        self.rand_rotate_angle = rand_rotate
        self.rand_rotate = rand_rotate / 180 * math.pi
        self.rand_zoom_in = rand_zoom_in
        self.rand_brightness = rand_brightness
        self.rand_saturation = rand_saturation
        self.rand_shift = rand_shift

    def rand_affine_matrix(self, img_tensor):
        batch_size = img_tensor.size(0)
        tx = torch.FloatTensor(batch_size).uniform_(-self.rand_shift, self.rand_shift).to(self.device)
        ty = torch.FloatTensor(batch_size).uniform_(-self.rand_shift, self.rand_shift).to(self.device)
        angle = torch.FloatTensor(batch_size).uniform_(-self.rand_rotate, self.rand_rotate).to(self.device)
        sin, cos = torch.sin(angle), torch.cos(angle)
        scale = torch.FloatTensor(batch_size).uniform_(1-self.rand_zoom_in, 1+self.rand_zoom_in).to(self.device)
        theta = torch.zeros(batch_size, 2, 3).to(self.device)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        grid = F.affine_grid(theta, img_tensor.shape)
        img_tensor_t = F.grid_sample(img_tensor, grid)
        return img_tensor_t

    def forward(self, img_tensor: torch.Tensor, p_aug: float = 0.5) -> torch.Tensor:
        if torch.rand(1).item() > p_aug:
            return img_tensor
        choice = int(torch.randint(0, 4, (1,)))
        img_tensor_t = img_tensor
        if choice == 0:
            img_tensor_t = K.RandomGaussianNoise(mean=0., std=.01, p=.5)(img_tensor)
            factor = torch.FloatTensor([0]).uniform_(0, self.rand_brightness).item()
            img_tensor_t = kornia.enhance.adjust_brightness(img_tensor_t, factor, clip_output=True)
        elif choice == 1:
            factor = torch.FloatTensor([0]).uniform_(1 - self.rand_saturation, 1 + self.rand_saturation).item()
            img_tensor_t = kornia.enhance.adjust_saturation(img_tensor, factor)
        elif choice == 2:
            img_tensor_t = self.rand_affine_matrix(img_tensor)
        elif choice == 3:
            img_tensor_t = K.RandomGrayscale(p=1)(img_tensor)
        return img_tensor_t
