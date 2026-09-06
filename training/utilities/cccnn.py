import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from .ISP import total_cosine_error, get_xy_and_cct, estimate_cst
import numpy as np


class CCCNN(nn.Module):
    def __init__(self, mlp, mode, exposure_invariant=False):
        super(CCCNN, self).__init__()
        self.mlp = mlp
        self.low_temp_cst = None
        self.high_temp_cst = None
        #make illum_idx a tensor full of -1s a 1000 of them
        self.ccts_stored = torch.full((1000,), -1, dtype=torch.int32)
        self.xys_stored = torch.full((1000, 2), -1, dtype=torch.float32)
        self.mode = mode
        self.exposure_invariant = exposure_invariant
        if self.exposure_invariant:
            self.scale_predictor = nn.Linear(in_features=3, out_features=1, bias=True, dtype=torch.float32)
            #initialize weights so each of the 3 input channels has  1/3 weight and the bias is 0
            nn.init.constant_(self.scale_predictor.weight, 1/3)
            nn.init.constant_(self.scale_predictor.bias, 0)

    def forward(self,white_point, wb_img, illum_idx=None):
        """
        whitepoint: torch.Tensor of shape [B, 3] (white point in rawrgb)
        wb: torch.Tensor of shape [B, 3, H, W] (white-balanced RGB)
        gt: torch.Tensor of shape [B, 3, H, W] (ground truth XYZ)
        
        Returns: torch.Tensor of shape [B, 3, 3]
        """



        N, C, H, W = wb_img.shape

        # Flatten spatial dimensions
        wb_flat = wb_img.view(N, C, -1).permute(0, 2, 1)  # [N, H*W, 3]
        mlp_input = white_point  # [N, C]

        # Ensure mlp_input is [N, C] and expand to [N, H*W, C]
        mlp_input = mlp_input.view(N, 1, -1).expand(-1, H * W, -1)

        #normalize each of the input wb_flat by its sum
        if self.exposure_invariant:
            wb_flat_for_scale = wb_flat.reshape(-1, wb_flat.size(-1))
            scale_pred = self.scale_predictor(wb_flat_for_scale)
            scales = torch.sum(wb_flat, dim=2, keepdim=True) + 1e-8
            wb_flat = wb_flat / scales

        # Concatenate -> [N, H*W, (3 + C)]
        mlp_input = torch.cat([wb_flat, mlp_input], dim=2)


        # Flatten to pass into MLP -> [(N * H * W), 3 + C]
        mlp_input = mlp_input.view(-1, mlp_input.size(-1))

        # Pass through MLP
        mlp_output = self.mlp(mlp_input)  # [(N * H * W), 3]

    
        if self.exposure_invariant:
            mlp_output = mlp_output * scale_pred
            
        # Reshape back to [N, 3, H, W]
        output = mlp_output.view(N, H * W, 3).permute(0, 2, 1).view(N, 3, H, W)

        return output
            
        
