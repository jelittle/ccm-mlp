import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFCST(nn.Module):
    """Interpolate CSTs in 2D xy chromaticity space via a radial basis function.

    Two kernels are supported via model_config["interp_method"]:
        "linear_rbf"  — φ(r)=r,         degree=1 polynomial augmentation
        "tps"         — φ(r)=r²log(r),  degree=1 polynomial augmentation (thin-plate spline)

    Both are exact at the 7 anchor xy points and extrapolate linearly outside

    """

    KERNEL_MAP = {
        "linear_rbf": "linear",
        "tps": "thin_plate_spline",
    }

    def __init__(self, model_config):
        super().__init__()
        self.interp_method = model_config.get("interp_method", "tps")
        assert self.interp_method in self.KERNEL_MAP, (
            f"Unknown interp_method '{self.interp_method}'. "
            f"Choose from {list(self.KERNEL_MAP)}"
        )
        self.single_led_csts = None
        self._interp = None  # built lazily

    def _build_interp(self):
        anchors = self.single_led_csts
        xy_anchors = np.stack([a["xy"] for a in anchors])        # (7, 2)
        cst_flat   = np.stack([a["cst"].numpy().ravel() for a in anchors])  # (7, 9)
        kernel = self.KERNEL_MAP[self.interp_method]
        self._interp = RBFInterpolator(xy_anchors, cst_flat, kernel=kernel, degree=1, smoothing=0)

    def forward(self, white_point, wb_img):
        """
        white_point: Tensor[B, 2]  — normalized xy in [-1, 1]²
        wb_img:      Tensor[B, 3, H, W]
        Returns:     Tensor[B, 3, 3]
        """
        if self._interp is None:
            assert self.single_led_csts is not None, (
                "single_led_csts not set; call runner.py wiring first."
            )
            self._build_interp()

        xy_np = white_point.detach().cpu().numpy()          # (B, 2)
        cst_flat = self._interp(xy_np)                      # (B, 9)
        csts = torch.from_numpy(cst_flat).float().view(-1, 3, 3)
        return csts.to(wb_img.device)
