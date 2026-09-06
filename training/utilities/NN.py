import torch
import torch.nn as nn
from .bestfitsolver import BestFitSolver

class NN(nn.Module):
    def __init__(self, mode: str, k: int = 1, dtype=torch.float32):
        super().__init__()
        self.mode, self.k, self.dtype = mode, k, dtype
        self.solver = BestFitSolver()
        self.register_buffer("points", None, persistent=False)         # (N, D)
        self.register_buffer("cst_transforms", None, persistent=False) # (N, 3, 3)

    @torch.no_grad()
    def add_point(self, image_path, white_point, white_balanced_image, target):
        if white_balanced_image.dim() == 3: white_balanced_image = white_balanced_image.unsqueeze(0)
        if target.dim() == 3:               target = target.unsqueeze(0)
        cst = self.solver(white_balanced_image, target).unsqueeze(0)   # (1,3,3)

        if white_point.ndim == 1: white_point = white_point[None, :]
        new_pt = white_point.to(white_balanced_image.device, self.dtype)

        self.points = new_pt if self.points is None else torch.cat([self.points, new_pt], 0)
        self.cst_transforms = cst if self.cst_transforms is None else torch.cat([self.cst_transforms, cst], 0)

    @torch.no_grad()
    def forward(self, white_point, wb_img):
        if self.points is None: raise ValueError("Call add_point first.")
        if self.points.device != wb_img.device:
            self.points = self.points.to(wb_img.device, non_blocking=True)
            self.cst_transforms = self.cst_transforms.to(wb_img.device, non_blocking=True)

        if white_point.ndim == 1: white_point = white_point[None, :]
        q = white_point.to(self.points.device, self.dtype)   # (B,D)

        d2 = (q*q).sum(1, keepdim=True) + (self.points*self.points).sum(1)[None,:] - 2*(q @ self.points.t())

        if self.k == 1:
            idx = d2.argmin(1)
            return self.cst_transforms[idx].float()[:,0,:].reshape(-1, 3, 3) # a hack to return the right shape for the get_model_output
        _, idx = torch.topk(d2, self.k, dim=1, largest=False)
        return self.cst_transforms[idx].float()[:,0,:].reshape(-1, 3, 3) # a hack to return the right shape for the get_model_output
