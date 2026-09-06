import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from .ISP import total_cosine_error
import numpy as np

from scipy.optimize import minimize

def cosine_error(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-6)
    dot = np.sum(v1_norm * v2_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot)
    return np.mean(angles)

def total_cosine_error(m_flat, RGB, XYZ):
   
    M = m_flat.reshape(3, 3)
    RGB_mapped = RGB @ M.T
    cos_error=cosine_error(RGB_mapped, XYZ)

    return cos_error

class BestFitSolver(nn.Module):
    def __init__(self):
        super(BestFitSolver, self).__init__()

    def forward(self, wb_img, targets, white_point=None):  # keep white_point for compatibility
        """
        wb_img:
            - torch.Tensor of shape [B, 3, H, W]  OR  [B, N, 3, H, W]
              (white-balanced RGB image patch(es))
        targets:
            - torch.Tensor of shape [B, 3, H, W]  OR  [B, N, 3, H, W]
              (ground truth XYZ)

        Returns:
            - torch.Tensor of shape [B, 3, 3]      if inputs were 4D
            - torch.Tensor of shape [B, N, 3, 3]   if inputs were 5D
        """

        # --- Validate rank (4D or 5D) and channel dim ---
        assert wb_img.dim() in (4, 5) and targets.dim() in (4, 5), "Inputs must be 4D or 5D"
        assert wb_img.dim() == targets.dim(), "wb_img and targets must have same rank"
        if wb_img.dim() == 4:
            assert wb_img.size(1) == 3 and targets.size(1) == 3, "Inputs must have 3 channels"
            B, C, H, W = wb_img.shape
            B_flat, N = B, None
            wb_img_flat = wb_img
            targets_flat = targets
        else:
            # 5D: [B, N, 3, H, W] -> collapse to [B*N, 3, H, W]
            assert wb_img.size(2) == 3 and targets.size(2) == 3, "Inputs must have 3 channels"
            B, N, C, H, W = wb_img.shape
            B_flat = B * N
            wb_img_flat = wb_img.reshape(B_flat, C, H, W)
            targets_flat = targets.reshape(B_flat, C, H, W)

        # --- Prepare numpy views for per-sample optimization ---
        wb_img_np = wb_img_flat.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B_flat, H, W, 3]
        target_np = targets_flat.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B_flat, H, W, 3]

        T_all = []
        fails = success = 0
        fail_error = success_error = 0.0

        # Helper: build full 3x3 from 8 free params (center fixed to 1.0)
        def build_matrix(m):
            full = np.insert(m, 4, 1.0)  # insert at index 4 -> position (1,1) in row-major 3x3
            return full.reshape(3, 3)

        # Expect user to provide this elsewhere; we just call it.
        def wrapped_error(m, *args):
            T = build_matrix(m)
            return total_cosine_error(T, *args)

        # Initial guess (8 elements; middle fixed separately). Keep values as in original code.
        M0 = np.array([
            9.362e-01, -2.131e-01,  1.421e-01,
            3.763e-01,              -2.810e-01,
            1.004e-02, -2.873e-01,  1.000e+00
        ]) * 10.0
        bounds = [(-1, 1)] * len(M0)

        for b in range(B_flat):
            RGB = wb_img_np[b].reshape(-1, 3)
            XYZ = target_np[b].reshape(-1, 3)

            result = minimize(wrapped_error, M0, args=(RGB, XYZ), method='L-BFGS-B', bounds=bounds)

            if result.success:
                success += 1
                success_error += float(result.fun)
            else:
                fails += 1
                fail_error += float(result.fun)

            t = build_matrix(result.x)
            T_all.append(torch.tensor(t, dtype=torch.float32))

        T = torch.stack(T_all, dim=0).to(wb_img.device)  # [B_flat, 3, 3]

        # --- Expand back to original batching ---
        if N is None:
            return T.view(B, 3, 3)               # 4D input -> [B, 3, 3]
        else:
            return T.view(B, N, 3, 3)            # 5D input -> [B, N, 3, 3]