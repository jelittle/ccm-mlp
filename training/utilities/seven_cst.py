import torch
import torch.nn as nn
import numpy as np

from .data_utils import denormalize_cct


class SevenCST(nn.Module):
    """Calibrate one CST per single-LED illuminant and interpolate in reciprocal-CCT space.

    Attributes set by runner.py after dataset init:
        single_led_csts: list of dicts {"cst": Tensor[3,3], "xy": ndarray[2], "cct": float}
                         sorted in ascending CCT order.
    """

    def __init__(self):
        super().__init__()
        self.single_led_csts = None

    def forward(self, white_point, wb_img):
        """
        white_point: Tensor[B, 1]  — normalized CCT in [-1, 1]
        wb_img:      Tensor[B, 3, H, W]
        Returns:     Tensor[B, 3, 3]  — interpolated CST per sample
        """
        anchors = self.single_led_csts
        assert anchors is not None, "single_led_csts not set; call runner.py wiring first."

        cst_list = []
        for cct_norm in white_point:
            cct = float(denormalize_cct(cct_norm))
            cct = max(anchors[0]["cct"], min(anchors[-1]["cct"], cct))

         
            lo, hi = anchors[0], anchors[-1]
            for i in range(len(anchors) - 1):
                if anchors[i]["cct"] <= cct <= anchors[i + 1]["cct"]:
                    lo, hi = anchors[i], anchors[i + 1]
                    break

            cct1, cct2 = lo["cct"], hi["cct"]
            if abs(cct1 - cct2) < 1e-6:
                g = 1.0
            else:
                g = (1.0 / cct - 1.0 / cct2) / (1.0 / cct1 - 1.0 / cct2)
                g = max(0.0, min(1.0, g))

            cst = g * lo["cst"] + (1.0 - g) * hi["cst"]
            cst_list.append(cst)

        return torch.stack(cst_list).to(wb_img.device)
