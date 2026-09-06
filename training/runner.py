import copy
import logging
import os
from pathlib import Path
from time import time
import cv2
import numpy as np

import torch
from torch import optim
import yaml
from utilities.model import apply_cst, get_device, get_model,get_model_output,delta_e_loss, cos_loss, ang_loss, get_white_point_type
from utilities.paths import resolve_config
from utilities.checkpoints import save_checkpoint, load_checkpoint
from utilities.utils import wandb_enabled, setup_wandb, merge_dicts, calc_model_performance, build_run_name
from utilities.dataset import MLPDataset, get_dataloaders
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from functools import partial
try:
    import wandb
except ImportError:  # optional: training runs offline without it
    wandb = None
from tqdm import tqdm
import argparse
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
import cv2

        


def _resolve_unique_eval_output_dir(base_save_path, run_name):
    base_dir = os.path.join(base_save_path, run_name)
    if not os.path.exists(base_dir):
        return base_dir

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    candidate = f"{base_dir}"
    suffix = 1
   
    return candidate


def training(model_og, dataloaders, config, device):
    """
    Placeholder function for training the model.
    """
    model_config = config['model']
    data_config = config['data']
    epochs = model_config['epochs']
    batch= data_config['batch_size']
    lr= model_config['lr']
    logging.info(f'''Starting training:
    Epochs:          {epochs} epochs
    Batch size:      {batch}
    Learning rate:   {lr}
    Device:          {device}
    ''')
    


    model = model_og.to(device, non_blocking=True)

    if model_config['type'] in ["MLP2DWP","MLP2DANG", "MLP2DXY", "MLP1DCCT", "CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    mean_on_val=torch.inf

    best_val_model=None

    iteration = 0

    #LOAD CALIBRATION CSTS
    if model_config['type'] in ["original2cst", "2cst","3cst", "CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG","MLP2DANG", "MLP2DWP", "MLP2DXY", "MLP1DCCT"]:


        model.low_temp_cst, model.high_temp_cst, mid_temp_csts = dataloaders['calibration_points_loader'].dataset.low_temp_cst, dataloaders['calibration_points_loader'].dataset.high_temp_cst, dataloaders['calibration_points_loader'].dataset.mid_temp_cst
        if model_config['type'] == "3cst":
            if mid_temp_csts is None:
                raise ValueError("Mid temperature calibration matrix is None. Ensure that the mid temperature exists in the dataset.")

        model.mid_temp_cst = mid_temp_csts
        best_val_model = copy.deepcopy(model)

    elif model_config['type'] in ["7cst", "RBF2DXY"]:
        single_led_csts = dataloaders['calibration_points_loader'].dataset.single_led_csts
        assert single_led_csts is not None, (
            f"single_led_csts is None for model type {model_config['type']}. "
            "Only lightbox datasets have single-LED calibration patches."
        )
        model.single_led_csts = single_led_csts
        best_val_model = copy.deepcopy(model)

    elif model_config['type'] in ["NN2DXY", "NN2DWP","NN2DANG", "NN1DCCT"]:
        model.low_temp_cst, model.high_temp_cst, mid_temp_csts = dataloaders['calibration_points_loader'].dataset.low_temp_cst, dataloaders['calibration_points_loader'].dataset.high_temp_cst, dataloaders['calibration_points_loader'].dataset.mid_temp_cst
 
        best_val_model = copy.deepcopy(model)
   
       
        for batch in tqdm(dataloaders["train_loader"], leave=False):
            
            image_paths= batch['path']
            white_point = get_white_point_type(batch, config['model']).to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            wb_img = batch['wb_img'].to(device, non_blocking=True).float()

            for i in range(len(image_paths)):
                model.add_point(image_paths[i], white_point[i],wb_img[i], targets[i])

        best_val_model = copy.deepcopy(model)
    
   
    if model_config['type'] in ["MLP2DWP", "MLP2DXY","MLP2DANG", "MLP1DCCT", "CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]:


        # I have commented this out for now as it is needed just once to get the normalization values
        # max_wp = None
        # #if name has "XY" or "WP" in it, we need to find the max white point in the training data for normalization
        # for batch in tqdm(dataloaders["white_calibrate_loader"], desc="Finding max white point", leave=False):
        #     white_point = get_model_inputs(model, batch, device, config, test=True, normalize_wp=False)["white_point"]
        #     max_wp_batch, _ = torch.max(white_point, dim=0)
        #     max_wp = max_wp_batch if max_wp is None else torch.max(max_wp, max_wp_batch)
        # print("Max whitepointinfo in training data: ", max_wp)
        # print("Max xy in training data: ", dataloaders['white_calibrate_loader'].dataset.max_xy)
        # print("Max wp in training data: ", dataloaders['white_calibrate_loader'].dataset.max_wp)

        for batch in tqdm(dataloaders["train_loader"], desc=f"Training Epoch {iteration+1}", leave=False):

            if model_config['type'] in ["MLP2DWP","MLP2DANG", "MLP2DXY", "MLP1DCCT"]:

                targets = batch['target'].to(device, non_blocking=True)
                wb_img = batch['wb_img'].to(device, non_blocking=True).float()
                #best_fit_matrix = batch['best_fit_matrix'].to(device, non_blocking=True).float()
                #linearly ramp the stochastic precision from 0.1 to 0 for the first 50000 iterations
                if "precon" in model_config and not model_config["precon"]:
                    stochastic_prec = 0
                    print("Preconditioning disabled, setting stochastic precision to 0")
                elif model_config["type"] in ["MLP2DWP","MLP2DANG", "MLP2DXY"]:
                    if "precon_amount" in model_config:
                        stochastic_prec = model_config["precon_amount"]
                    else:
                        stochastic_prec = 0.05
                else:
                    stochastic_prec = 0
                output=get_model_output(model, batch, device, config, stochastic_prec=stochastic_prec)  

                # Calculate loss
                xyz = apply_cst(wb_img, output, matrix_type=model_config.get("matrix_type", "linear"))
                closs = cos_loss(xyz, targets)
                aloss = ang_loss(xyz, targets)
                #best_fit_matrix_loss = F.mse_loss(output, best_fit_matrix)
                delta_e_loss = F.mse_loss(xyz, targets)

                log_dict = {
                    'cos Loss/train': closs.item(),
                    'Angular_Loss/train': aloss.item(),
                    #'Best_Fit_Matrix_Loss/train': best_fit_matrix_loss.item(),
                    'L2_Loss/train': delta_e_loss.item(),
                    "Epoch": iteration
                }
                if wandb_enabled(config):
                    wandb.log(log_dict)
                
                optimizer.zero_grad()
                closs.backward() #Use the cosine loss for backprop
                #aloss.backward() #Use the angular loss for backprop
                #closs.backward() #Use the best fit matrix loss for backprop
                optimizer.step()
                iteration += 1
                if (iteration + 1) % 500 == 0:
                   
                    this_mean_on_val = evaluate_model(model, config, device, mode="val", dataloader=dataloaders["val_loader"], output_detail="none")
                    if this_mean_on_val < mean_on_val:
                        mean_on_val = this_mean_on_val
                        best_val_model = copy.deepcopy(model)

            elif model_config['type'] in ["CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]:

                if "precon" in model_config and not model_config["precon"]:
                    stochastic_prec = 0
                    print("Preconditioning disabled, setting stochastic precision to 0")
                elif model_config["type"] in ["CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]:
                    if "precon_amount" in model_config:
                        stochastic_prec = model_config["precon_amount"]
                    else:
                        stochastic_prec = 0 #default value
                else:
                    stochastic_prec = 0
                    
                # print("test1")
                targets = batch['target'].to(device, non_blocking=True)
          
            
                wb_img = batch['wb_img'].to(device, non_blocking=True).float()
                
                output = get_model_output(model, batch, device, config, stochastic_prec=stochastic_prec)
                
                # Calculate loss
                mse_loss = F.mse_loss(output, targets)
                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()
                iteration += 1
          
                log_dict = {
                    'mse Loss/train': mse_loss.item(),
                    "Epoch": iteration
                }
                if wandb_enabled(config):
                    wandb.log(log_dict)
                # print("test3")

                if (iteration + 1) % 2000 == 0:
                    # print("test4")
             
                    this_mean_on_val = evaluate_model(model, config, device, mode="val", dataloader=dataloaders["val_loader"], output_detail="none")

                    # print(f"Validation loss: {this_mean_on_val}")
                    # print("Previous best mean loss: ", mean_on_val)
                    if this_mean_on_val < mean_on_val:
                        mean_on_val = this_mean_on_val
                        best_val_model = copy.deepcopy(model)            
            else:
                break
    elif model_config['type'] in ["bestfitsolver"]:
            best_val_model = model

    evaluate_model(best_val_model, config, device, mode="test", dataloader=dataloaders["test_loader"], output_detail="lite")


    return best_val_model  


def downscale(image, scale_ratio, order="chw"):
    if scale_ratio == 1:
        return image
    if image.ndim == 4:
        test=None
        pass
    if order in "chw":
        image_hwc = np.transpose(image, (1, 2, 0))
    elif order == "bchw":
        # b, c, h, w = image.shape
        image_hwc = image.transpose(0, 2, 3, 1)
    else:
        image_hwc = image


    if order == "bchw":
        b, c, h, w = image.shape
        target_h = max(1, int(h / scale_ratio))
        target_w = max(1, int(w / scale_ratio))
        
        # Pre-allocate output array
        resized_hwc = np.empty((b, target_h, target_w, c), dtype=image.dtype)
        
        for i in range(b):
            resized_hwc[i] = cv2.resize(image_hwc[i], (target_w, target_h), 
                                   interpolation=cv2.INTER_AREA)
    else:
        
        target_h = max(1, int(image_hwc.shape[0] / scale_ratio))
        target_w = max(1, int(image_hwc.shape[1] / scale_ratio))
        resized_hwc = cv2.resize(image_hwc, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    
    if order == "chw":
        return np.transpose(resized_hwc, (2, 0, 1))
    elif order == "bchw":
        resized_chw = resized_hwc.transpose(0,3, 1, 2)
        return resized_chw
    else:
        return resized_hwc


def evaluate_model(model, config, device, mode: str, dataloader=None, data=None, temps=None, output_detail="none", show_progress=False):
    """
    Unified evaluator:
    - dataloader: use directly (validation-style)
    - data: build dataloader (test-style)
    - output_detail: "none" | "lite" | "full"
    """

    model_config, data_config = config['model'], config.get('data', {})

    # Build dataloader if missing
    if dataloader is None:
        if data is None: raise ValueError("Provide either dataloader or data.")
        if model_config['type'] in ["MLP2DXY", "MLP2DWP","MLP2DANG", "MLP1DCCT", "2DNNXY","2DNNWP"]:
            ds = MLPDataset(data, data_config, temps=temps)
            dataloader = DataLoader(ds, batch_size=data_config['batch_size'], shuffle=True, num_workers=16, prefetch_factor=2, pin_memory=True)


    model.eval()
    angular_e_list, delta_e_list = [], []
    will_write = output_detail in ("full")
    out_folder = config.get("_eval_output_dir", os.path.join(config['save_path'], wandb.run.name if wandb_enabled(config) else build_run_name(config)))
    os.makedirs(out_folder, exist_ok=True)
    #if will_write: path_list, rawpatch_img_list, wb_img_list, tgt_list, xyz_list, xyz_scaled_list, wp_list, xy_list, cct_list, ang_list, de_list = [], [], [], [], [], [], [], [], [], [], []

    #if mode in ["wild", "mixed"] and will_write:
    #    xyzfull_img_list, xyzfull_scaled_list, rawfull_img_list, wbfull_img_list = [], [], [], []
    #if mode in ["mixed"]:
    #    illuminationfull_map_list, mixturefull_map_list, xyzfull_each_list, illuminationpatch_map_list, mixturepatch_map_list, xyzpatch_each_list = [], [], [], [], [], []
    if will_write:
        macs, run_time, model_size = calc_model_performance(model, config, device=device, runs = 100)
    with torch.no_grad():
        if show_progress:
            dataloader_iter = tqdm(dataloader, desc=f"Testing {mode}", unit="batch")
        else:
            dataloader_iter = dataloader
        for batch in dataloader_iter:
            if mode not in ["mixed"]:
                wb_img, targets = batch['wb_img'].to(device).float(), batch['target'].to(device)
                xyz = get_model_output(model, batch, device, config, test=True)
                
                _, per_img_ang = ang_loss(xyz, targets, return_per_image=True)
                
                angular_e_list.append(per_img_ang.cpu())
                scale = torch.mean((targets[:, 1:2, :, :] + 1e-8), dim=(1, 2, 3), keepdim=True) / \
                        torch.mean((xyz[:, 1:2, :, :] + 1e-8), dim=(1, 2, 3), keepdim=True)  # match scale of all Y channels
                xyz_scaled = xyz * scale
                de_per_img = delta_e_loss(xyz_scaled, targets).cpu(); delta_e_list.extend(de_per_img.tolist())
            else:
                wb_img, targets = batch['wb_img'].to(device).float(), batch['target'].to(device)
                #merge the batch and illum dimensions
                from einops import rearrange
                wb_img_flatter = rearrange(wb_img, 'b i c h w -> (b i) c h w')
                targets_flatter = rearrange(targets, 'b i c h w -> (b i) c h w')
                xyz, xyz_each = get_model_output(model, batch, device, config, test=True, mode="mixed_patch")
                xyz_flatter = rearrange(xyz, 'b i c h w -> (b i) c h w')
                _, per_img_ang = ang_loss(xyz_flatter, targets_flatter, return_per_image=True)
                #reshape per_img_ang back to (b,i)
                per_img_ang = rearrange(per_img_ang, '(b i) -> b i', b=wb_img.shape[0])

                angular_e_list.append(per_img_ang.cpu())
                scale = torch.mean((targets_flatter[:, 1:2, :, :] + 1e-8), dim=(1, 2, 3), keepdim=True) / \
                        torch.mean((xyz_flatter[:, 1:2, :, :] + 1e-8), dim=(1, 2, 3), keepdim=True)  # match scale of all Y channels

                xyz_scaled = xyz_flatter * scale
                #rshaape xyz_scaled back to (b,i,c,h,w)
                xyz_scaled = rearrange(xyz_scaled, '(b i) c h w -> b i c h w', b=wb_img.shape[0])
                de_per_img = delta_e_loss(rearrange(xyz_scaled, 'b i c h w -> (b i) c h w'), targets_flatter).cpu(); delta_e_list.extend(de_per_img.tolist())
                

            if will_write:

                items = []
                for i in range(len(batch['path'])):
                    
                    item = {
                        'path': batch['path'][i],
                        "rawpatch_img": batch['rawpatch_img'][i].cpu().numpy(),
                        "wb_img": batch['wb_img'][i].cpu().numpy(),
                        "target": batch['target'][i].cpu().numpy(),
                        "output_xyz": xyz[i].cpu().numpy(),
                        "output_xyz_scaled": xyz_scaled[i].cpu().numpy(),
                        "whitepoint": batch['wp_unnormalized'][i].cpu().numpy(),
                        "xy": batch['xy_unnormalized'][i].cpu().numpy(),
                        "cct": batch['cct_unnormalized'][i].cpu().numpy(),
                        "angular_error": per_img_ang[i].cpu().numpy(),
                        "delta_e00": de_per_img[i].cpu().numpy(),
                        "model_size": model_size,
                        "run_time": run_time,
                        "macs": macs,
                        "scale": scale[i].cpu().numpy()
                    }

                    downscale_full = data_config.get('downscale_full', 1)

                    if mode == "wild":
                        #assert batch size is 1
                        assert batch['rawfull_img'].shape[0] == 1
                        scale = torch.mean(scale)
                        xyzfull = get_model_output(model, batch, device, config, test=True, mode="full")
                        xyzfull_scaled = xyzfull * scale
                        item.update({
                            "rawfull_img": downscale(batch['rawfull_img'][i].cpu().numpy(), downscale_full),
                            "wbfull_img": downscale(batch['wbfull_img'][i].cpu().numpy(), downscale_full),
                            "output_xyzfull": downscale(xyzfull[i].cpu().numpy(), downscale_full),
                            "output_xyz_scaled_full": downscale(xyzfull_scaled[i].cpu().numpy(), downscale_full),
                            "scale": scale.cpu().numpy()
                        })
                    if mode == "mixed":
                        #assert batch size is 1
                        assert batch['rawfull_img'].shape[0] == 1
                        scale = torch.mean(scale)
                        xyzfull, xyzfull_each = get_model_output(model, batch, device, config, test=True, mode="mixed_full")
                        xyzfull_each = xyzfull_each * scale
                        xyzfull_scaled = xyzfull * scale

                        item.update({
                            "illuminationfull_map": downscale(batch['illumination_map'][i].cpu().numpy(), downscale_full),
                            "mixturefull_map": downscale(batch['mixture_map'][i].cpu().numpy(), downscale_full, order="hwc"),
                            "output_xyzfull": downscale(xyzfull[i].cpu().numpy(), downscale_full),
                            "output_xyzfull_each": downscale(xyzfull_each[i].cpu().numpy(), downscale_full, order="bchw"),
                            "output_xyz_scaled_full": downscale(xyzfull_scaled[i].cpu().numpy(), downscale_full),
                            "wbfull_img": downscale(batch['wbfull_img'][i].cpu().numpy(), downscale_full),
                            "rawfull_img": downscale(batch['rawfull_img'][i].cpu().numpy(), downscale_full),
                            "chart_map": downscale(batch['chart_map'][i].cpu().numpy(), downscale_full, order="hwc"),
                            "illuminationpatch_map": batch['illumination_map_patches'][i].cpu().numpy(),
                            "mixturepatch_map": batch['mixture_map_patches'][i].cpu().numpy(),
                            "output_xyz_each": xyz_each[i].cpu().numpy(),
                            "scale": scale.cpu().numpy()
                        })
                    items.append(item)
                    
                def save_item(item, out_folder, mode):
                    out_path = os.path.join(out_folder, mode, Path(item["path"]).stem,)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, item, allow_pickle=True)

                fn = partial(save_item, out_folder=out_folder, mode=mode)
                for item in items:
                    fn(item)

    full_angular_e = torch.cat(angular_e_list,0) if angular_e_list else torch.tensor([])
    mean_loss = torch.mean(full_angular_e).item() if full_angular_e.numel() else float('nan')
    percentiles = torch.quantile(full_angular_e, torch.tensor([0.1,0.25,0.5,0.75,0.9],dtype=full_angular_e.dtype),dim=0).cpu().numpy().tolist() if full_angular_e.numel() else [float('nan')]*5
    full_de00, mean_de = np.array(delta_e_list), (float(np.mean(delta_e_list)) if delta_e_list else float('nan'))

    log_dict = {f'Angular_Loss/{mode}_mean':mean_loss,
                f'Angular_Loss/{mode}_10th':percentiles[0], f'Angular_Loss/{mode}_25th':percentiles[1],
                f'Angular_Loss/{mode}_50th':percentiles[2], f'Angular_Loss/{mode}_75th':percentiles[3],
                f'Angular_Loss/{mode}_90th':percentiles[4], f'DeltaE00/{mode}_mean':mean_de}
    if wandb_enabled(config):
        wandb.log(log_dict)
    logging.info(f"[{mode}] mean angular loss: {mean_loss:.6f}, mean ΔE00: {mean_de:.6f}")
    return mean_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the Colour Correction model.")
    parser.add_argument(
        '--config',
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to one or more configuration files."
    )
    args = parser.parse_args()

    config = {}
    for config_path in args.config:
        with open(config_path, 'r') as file:
            new_config = yaml.safe_load(file) or {}
            config = merge_dicts(config, new_config)
        
    
    # Expand ${data_root}/${output_root}/... placeholders against training/configs/paths.yaml
    config = resolve_config(config)
    config["config_files"] = args.config  # Store the list of config files used
    # Extract values from config
    data_config = config['data']
    model_config = config['model']

    #matrix hack:
    if "matrix" in config:
        # merge any keys from matrix_config into model_config
        matrix_cfg = config.get('matrix', {})
        if isinstance(matrix_cfg, dict):
            for k, v in matrix_cfg.items():
                model_config[k] = v
    print("Model config: ", model_config) 
   


    gpus_chosen, device = get_device()

    run_name = setup_wandb(config, device)
    model_name =run_name
  
    config["_eval_output_dir"] = _resolve_unique_eval_output_dir(config['save_path'], run_name)

    #set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



    torch.backends.cudnn.benchmark = True  # CUDA optimization

    model=get_model(model_config)
    dataloaders = get_dataloaders(config)

  
    save_path= os.path.join(config['save_path'], model_name)
    model_save_path = os.path.join(save_path, "model.pth")

    # run_name = setup_wandb_config(args,net)
    if config['train']:
        model = training(model_og=model,dataloaders=dataloaders, config=config, device=device) 
        os.makedirs(save_path, exist_ok=True)
        save_checkpoint(model, save_path, model_config)
         # Placeholder for model_og
    else:
        #load the model
        if model_config['type'] in ["LUT"]:
            #self.model.mlp #load the MLP first
            distill_config = config.copy()
            distill_config['model'] = model_config['model_distill']
            distill_run_name = build_run_name(distill_config)
            distill_dir = os.path.join(config['save_path'], distill_run_name)
            print("Loading distillation model from:", distill_dir)
            mlp = load_checkpoint(distill_dir, model_config['model_distill'], device)
            model = model.calibrate_lut(mlp, model_config)  # Re-initialize the model
            #then delete the mlp so it is not counted in size calculation later
        elif data_config.get("wp_error", False) or data_config.get("wp_source", False):
            distill_config = config.copy()
            distill_config['data']['wp_error'] = False
            distill_config['data']['wp_source'] = False
            distill_run_name = build_run_name(distill_config)
            distill_dir = os.path.join(config['save_path'], distill_run_name)
            print("Loading distillation model from:", distill_dir)
            model = load_checkpoint(distill_dir, model_config, device)
        else:
            model = load_checkpoint(save_path, model_config, device)
        
    # test the model
    if not config.get("skip_mixed_and_wild", False):
        if "mixed_loader" in dataloaders:
           evaluate_model(model, config, device, mode="mixed", dataloader=dataloaders["mixed_loader"], output_detail="full", show_progress=True)
        if "wild_loader" in dataloaders:
            evaluate_model(model, config, device, mode="wild", dataloader=dataloaders["wild_loader"], output_detail="full", show_progress=True)
        
            print("Only wild evaluation requested, skipping test set evaluation.")
    if config.get("wild_only", False):
        print("Only wild evaluation requested, skipping test set evaluation.")
    else:    
        evaluate_model(model, config, device, mode="test", dataloader=dataloaders["test_loader"], output_detail="full", show_progress=True)


    if wandb_enabled(config):
        wandb.finish()

    
    
