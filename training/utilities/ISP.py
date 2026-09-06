import colour
colour.utilities.filter_warnings(colour_usage_warnings=True)
import torch

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from camera_pipeline.pipeline_lib.pipeline_utils import get_metadata
from .data_utils import denormalize_cct



def get_xy_and_cct(low_cst,high_cst,torch_whitepoints):
  
    """
    Estimate CCT for a batch of neutral patches.
    pytorch tensor with N,3 whitepoints
    
    Adds "cct" and "xy" to the batch dict.
    """
    
    t1, cct1 = low_cst
    t2, cct2 = high_cst

    cct1=float(cct1)
    cct2=float(cct2)
    #if torch, convert to numpy
    t1 = t1.cpu().numpy()
    t2 = t2.cpu().numpy()
    whitepoints = torch_whitepoints.cpu().numpy() if isinstance(torch_whitepoints, torch.Tensor) else torch_whitepoints
    if len(whitepoints.shape) == 1:
        whitepoints = whitepoints[np.newaxis, :]

    maxPasses = 30

    cct_list = []
    xy_list = []
    
    for neutral in whitepoints:
        #ensure netural is 3, array(if 2, add 1 in center)
   
        if neutral.shape[0] == 2:
            neutral = np.array([neutral[0], 1 , neutral[1]])

        est = colour.temperature.CCT_to_xy(5000)
        
        for i in range(maxPasses):
           
            cct_est = colour.temperature.xy_to_CCT(est)
            g = (cct_est**-1 - cct2**-1) / (cct1**-1 - cct2**-1)
            T_est = g * t1 + (1 - g) * t2
       
            
            neutral_xyz = T_est @ neutral
            X, Y, Z = neutral_xyz.flatten()
            next_xy = np.array([X / (X + Y + Z), Y / (X + Y + Z)])
            
            if np.abs(next_xy[0] - est[0]) < 1e-5 and np.abs(next_xy[1] - est[1]) < 1e-5:
                est = next_xy
                break
            est = next_xy
        if i == maxPasses - 1:
            est = [(est[0] + next_xy[0]) / 2, (est[1] + next_xy[1]) / 2]
        cct = colour.temperature.xy_to_CCT(est)
     
        cct_list.append(cct)
        xy_list.append(est)
   
    return np.array(xy_list), np.array(cct_list)




# def get_xy_and_cct(low_cst,high_cst,torch_whitepoints):
  
#     """
#     Estimate CCT for a batch of neutral patches.
#     pytorch tensor with N,3 whitepoints
    
#     Adds "cct" and "xy" to the batch dict.
#     """
    
#     t1, cct1 = low_cst
#     t2, cct2 = high_cst

#     cct1=float(cct1)
#     cct2=float(cct2)
#     whitepoints = torch_whitepoints.cpu().numpy() if isinstance(torch_whitepoints, torch.Tensor) else torch_whitepoints
#     if len(whitepoints.shape) == 1:
#         whitepoints = whitepoints[np.newaxis, :]

#     t1 = t1.cpu().numpy()
#     t2 = t2.cpu().numpy()

#     maxPasses = 30

#     cct_list = []
#     xy_list = []
    
#     for neutral in whitepoints:
#         #ensure netural is 3, array(if 2, add 1 in center)
   
#         if neutral.shape[0] == 2:
#             neutral = np.array([neutral[0], 1 , neutral[1]])
#         est = colour.temperature.CCT_to_xy(5000)
        
#         for i in range(maxPasses):
           
#             cct_est = colour.temperature.xy_to_CCT(est)
#             g = (cct_est**-1 - cct2**-1) / (cct1**-1 - cct2**-1)
#             T_est = g * t1 + (1 - g) * t2
       
            
#             neutral_xyz = T_est @ neutral.T
#             X, Y, Z = neutral_xyz.flatten()
#             next_xy = np.array([X / (X + Y + Z), Y / (X + Y + Z)])
            
#             if np.abs(next_xy[0] - est[0]) < 1e-5 and np.abs(next_xy[1] - est[1]) < 1e-5:
#                 est = next_xy
#                 break
#             est = next_xy
#         if i == maxPasses - 1:
#             est = [(est[0] + next_xy[0]) / 2, (est[1] + next_xy[1]) / 2]
#         cct = colour.temperature.xy_to_CCT(est)
     
#         cct_list.append(cct)
#         xy_list.append(est)
   
#     return np.array(xy_list), np.array(cct_list)



   
def estimate_cst(ccts, low_cst,high_cst, mid_cst=None, ):
    

    cst_list=[]
    for cct in ccts:

        cct = denormalize_cct(cct)
        

        # (fm1,cct1),(fm2,cct2)=cam.get_closest_csts(cct)
        (fm1,cct1),(fm2,cct2) =  low_cst,high_cst

        #clamp cct between cct1 and cct2
        cct = torch.clip(cct, cct1, cct2)

        

        if mid_cst is not None: #change interpolation points
          
            fm3, cct3 = mid_cst
            if cct < cct3:
                fm2, cct2 = fm3, cct3
            elif cct >= cct3:
                fm1, cct1 = fm3, cct3


        fm1=fm1.cpu().numpy()
        fm2=fm2.cpu().numpy()
        cct = cct.cpu().numpy()

       

        cct_inv=1/cct
        cct1_inv=1/cct1
        cct2_inv=1/cct2
     
        #get g
        g=(cct_inv-cct2_inv)/(cct1_inv-cct2_inv)
        cst_est=g*fm1+(1-g)*fm2
        cst_list.append(cst_est)
    csts = torch.from_numpy(np.array(cst_list)).float()
    return csts
    


def total_cosine_error(M_flat, RGB, XYZ):
    """better than angular error"""
    M = M_flat.reshape(3, 3)

    
    mr = RGB @ M.T  # shape: (N, 3)

    # Normalize both vectors
    mr_norm = mr / np.linalg.norm(mr, axis=1, keepdims=True)
    x_norm = XYZ / np.linalg.norm(XYZ, axis=1, keepdims=True)

    # Cosine error: 1 - cos(theta)
    cos_error = 1 - np.sum(mr_norm * x_norm, axis=1)
 

    

    return np.sum(cos_error)
def estimate_forward_matrix(RGB_wb, XYZ_gt):
   
    # Flatten inputs
    RGB = RGB_wb.reshape(-1, 3)
    XYZ = XYZ_gt.reshape(-1, 3)


    # Initialize with identity
    M0 = np.eye(3).flatten()

    # Minimize angular error
    result = minimize(total_cosine_error, M0, args=(RGB, XYZ), method='BFGS')
    
    t = result.x.reshape(3, 3)

    t_c=t[1,1]
    t = t / t_c
  
    return t