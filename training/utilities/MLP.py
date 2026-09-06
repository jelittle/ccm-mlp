import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from .ISP import total_cosine_error, get_xy_and_cct, estimate_cst
import numpy as np


class MLP(nn.Module):
    def __init__(self, mlp, mode, matrix_type = "linear"):
        super(MLP, self).__init__()
        self.mlp = mlp
        self.low_temp_cst = None
        self.high_temp_cst = None
        #make illum_idx a tensor full of -1s a 1000 of them

        self.mode = mode
        self.matrix_type = matrix_type
        #store all

    def forward(self,white_point, wb_img, illum_idx=None):
        """
        white_points: torch.Tensor of shape [B, 2] (white point in rawrgb)
        inputs: torch.Tensor of shape [B, 3, H, W] (white-balanced RGB)
        gt: torch.Tensor of shape [B, 3, H, W] (ground truth XYZ)
        
        Returns: torch.Tensor of shape [B, 3, 3]
        """
        # Pass through MLP
        mlp_output = self.mlp(white_point)  # [(N * H * W), 3]

        #if self doesn't have matrix type set it to linear (just to be backward compatiable)
        if not hasattr(self, "matrix_type") or self.matrix_type is None:
            self.matrix_type = "linear"

        if self.matrix_type == "linear":
            # force the 5th channel (middle entry of 3x3 matrix) to 1.0 without in-place writes
            output = torch.cat([
                mlp_output[..., :4],
                torch.ones_like(mlp_output[..., 4:5]),
                mlp_output[..., 5:]
            ], dim=-1)

            #reshape to [B, 3, 3]
            output = output.view(-1, 3, 3)
        elif self.matrix_type == "polynomial2":
            #generate the 3x9 matrix 
            output = mlp_output.view(-1, 3, 9)
            mask = torch.zeros_like(output)
            mask[:, 1, 1] = 1
            output = output * (1 - mask) + mask * 1.0
        elif self.matrix_type == "polynomial3":
            #generate the 3x19 matrix
            output = mlp_output.view(-1, 3, 19)
            mask = torch.zeros_like(output)
            mask[:, 1, 1] = 1
            output = output * (1 - mask) + mask * 1.0
        elif self.matrix_type == "rootpolynomial2":
            #3x6 matrix
            output = mlp_output.view(-1, 3, 6)
            mask = torch.zeros_like(output)
            mask[:, 1, 1] = 1
            output = output * (1 - mask) + mask * 1.0
        elif self.matrix_type == "rootpolynomial3":
            #3x13 matrix
            output = mlp_output.view(-1, 3, 13)
            mask = torch.zeros_like(output)
            mask[:, 1, 1] = 1
            output = output * (1 - mask) + mask * 1.0

        return output
            
        
