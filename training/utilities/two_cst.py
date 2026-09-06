import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from .ISP import total_cosine_error, get_xy_and_cct, estimate_cst
import numpy as np


class TwoCst(nn.Module):
    def __init__(self):
        super(TwoCst, self).__init__()
        self.low_temp_cst = None
        self.high_temp_cst = None

    def forward(self, white_point, wb_img):
        """
        whitepoint: torch.Tensor of shape [B, 3] (white point in rawrgb)
        wb: torch.Tensor of shape [B, 3, H, W] (white-balanced RGB)
        gt: torch.Tensor of shape [B, 3, H, W] (ground truth XYZ)
        
        Returns: torch.Tensor of shape [B, 3, 3]
        """
        output=estimate_cst(white_point, self.low_temp_cst, self.high_temp_cst).to(wb_img.device) #wihtepoint is on gpu(if using cuda)
        return output

       
        
