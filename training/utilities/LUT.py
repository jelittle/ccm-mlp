import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LUT(nn.Module):
    def __init__(self, lut_width: int):
        super().__init__()
        self.lut_width = lut_width
        # Pre-register buffer so it exists even before calibration.
        self.register_buffer("lut_field", None, persistent=False)

    @torch.no_grad()
    def calibrate_lut(self, mlp: nn.Module, config=None):
        """
        Generate LUT field from MLP over a [-1,1]x[-1,1] grid.
        Stores as a non-persistent buffer `_lut_field` with shape [1, 9, W, W].
        """
        W = self.lut_width
        grid = np.linspace(-1, 1, W, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # [W*W, 2]

        dev = next(mlp.parameters()).device
        dtype = next(mlp.parameters()).dtype

        points_tensor = torch.tensor(points, device=dev, dtype=dtype)
        wb_img_dummy = torch.zeros(points_tensor.shape[0], device=dev, dtype=dtype)

        lut_values = mlp(points_tensor, wb_img_dummy).view(W, W, 3, 3)
        lut_field = (
            lut_values.permute(2, 3, 0, 1)
                      .contiguous()
                      .view(1, 9, W, W)
        )

        # Update buffer value (no re-registration needed)
        self.lut_field = lut_field

        return self

    def forward(self, white_point: torch.Tensor, wb_img: torch.Tensor, illum_idx=None):
        """
        Args:
            white_point: [B, 2], in [-1, 1]^2
            wb_img: [B, 3, H, W] (unused)
        Returns:
            CSTs: [B, 3, 3]
        """
        if self.lut_field is None:
            raise RuntimeError("LUT not calibrated. Call calibrate_lut() first.")

        B = white_point.shape[0]
        lut = self.lut_field

        # CPU fallback for float16
        if lut.device.type == "cpu" and lut.dtype == torch.float16:
            lut_for_sample = lut.float()
        else:
            lut_for_sample = lut

        grid = white_point.to(device=lut_for_sample.device,
                              dtype=lut_for_sample.dtype).view(B, 1, 1, 2)

        sampled = F.grid_sample(
            lut_for_sample.expand(B, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(B, 9)

        sampled[:, 4] = sampled.new_tensor(1.0)
        return sampled.view(B, 3, 3)
