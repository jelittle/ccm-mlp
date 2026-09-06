

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.functional as F
from pandas import DataFrame
from camera_pipeline.pipeline_lib.pipeline_utils import get_metadata
from .image_processing import white_balance,get_calibrated_csts,get_single_led_csts
from .ISP import get_xy_and_cct
import logging
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from .data_utils import normalize_cct, denormalize_cct
from .bestfitsolver import BestFitSolver
# Iterations per epoch. Overridable per-run with `dataset_iters` in the data
# config, which is what makes a quick smoke run possible:
#     data:
#       dataset_iters: 200
DATASET_ITERS = 100000


def _dataset_iters(data_config):
    return int((data_config or {}).get("dataset_iters", DATASET_ITERS))
def get_dataloaders(config):
    model_config = config['model']
    data_config = config['data']
    epochs = model_config['epochs']


    if config['train']:
        if model_config['type'] in ["original2cst"]:
            calibration_points_dataset = MLPDataset(data_config, split="original_cst", batch_size=data_config['batch_size'])
            logging.info("Using original 2cst for training in original_cst mode.")
            calibration_points_loader = DataLoader(calibration_points_dataset, batch_size=data_config['batch_size'], shuffle=True)
        if model_config['type'] in ["2cst", "3cst", "7cst", "RBF2DXY", "CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP",  "EXPINV2DANG","NN2DXY","NN2DANG", "NN2DWP", "NN1DCCT", "MLP2DWP", "MLP2DXY","MLP2DANG", "MLP1DCCT"]:
            calibration_points_dataset = MLPDataset(data_config, split="temps")
            logging.info("Using temperatures for training in 2cst/3cst mode.")
            calibration_points_loader = DataLoader(calibration_points_dataset, batch_size=data_config['batch_size'], shuffle=True)
    
    

    if config['train']:
        if model_config['type'] in ["CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]: #special case for simulated data
            train_dataset = FullSimDataset(config['sim_data'], scale=epochs, batch_size=config['sim_data']['batch_size'])
            train_loader = DataLoader(train_dataset, batch_size=config['sim_data']['batch_size'], shuffle=True, num_workers=16, prefetch_factor=2, pin_memory=True)
        elif model_config['type'] in ["NN2DXY", "NN2DWP","NN2DANG", "NN1DCCT"]:
            train_dataset=MLPDataset(data_config, split="train", scale=1, batch_size=1,use_ITERS=False) 
            train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True,num_workers=16, prefetch_factor=2,pin_memory=True)
        elif config.get('use_sim_at_train', False):
            train_dataset = FullSimDataset(config['sim_data'], scale=epochs, batch_size=config['sim_data']['batch_size'], mode="complex")
            train_loader = DataLoader(train_dataset, batch_size=config['sim_data']['batch_size'], shuffle=True, num_workers=16, prefetch_factor=2, pin_memory=True)
        else:
            train_dataset=MLPDataset(data_config, split="train", scale=epochs, batch_size=data_config['batch_size'])
            train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True,num_workers=16, prefetch_factor=2,pin_memory=True)
        val_dataset=MLPDataset(data_config, split="val", batch_size=data_config['batch_size'])
        val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False,num_workers=16,pin_memory=True)

    test_dataset=MLPDataset(data_config, split="test", batch_size=1, wp_error= data_config.get("wp_error", False))
    test_loader = DataLoader(test_dataset, batch_size = 1 if not config['train'] else data_config['batch_size'], shuffle=False,num_workers=16,pin_memory=True)

    if 'wild_path' in data_config:
        wild_dataset=MLPDataset(data_config, split="wild", batch_size=1, wp_error= data_config.get("wp_error", False))
        wild_loader = DataLoader(wild_dataset, batch_size=1, shuffle=False,num_workers=0,pin_memory=True)

    if 'mixed_path' in data_config:
        mixed_dataset=MLPDataset(data_config, split="mixed", batch_size=1)
        mixed_loader = DataLoader(mixed_dataset, batch_size=1, shuffle=False,num_workers=0,pin_memory=True)


    dataloaders = {}
    #check if train_loader exists
    if 'train_loader' in locals():
        dataloaders['train_loader'] = train_loader
    if 'val_loader' in locals():
        dataloaders['val_loader'] = val_loader
    if 'test_loader' in locals():
        dataloaders['test_loader'] = test_loader
    if 'wild_loader' in locals():
        dataloaders['wild_loader'] = wild_loader
    if 'calibration_points_loader' in locals():
        dataloaders['calibration_points_loader'] = calibration_points_loader
    if 'mixed_loader' in locals():
        dataloaders['mixed_loader'] = mixed_loader
    return dataloaders

class MLPDataset(Dataset):
    #TODO: move mlp to xy and 2cst to wp
    def __init__(self, data_config, split, wp_error = False, scale=1, batch_size=1, use_ITERS=True):
        """
        Initialize the MLPDataset with data configuration and data.
        
        Args:
            data_config (dict): Configuration for the dataset.
        """
        self.use_ITERS = use_ITERS
        self.data_config = data_config
        self.wp_error = wp_error
   
        if data_config["chart_type"] == 24:
            self.target = np.load(data_config["target_24"])
            #move the last channel to the first position
            self.target = np.moveaxis(self.target, -1, 0)
        else:
            raise ValueError("Unsupported patch format. Only 24 is supported.")
        self.split = split
        self.scale = scale
        self.batch_size = batch_size

        self.solver = BestFitSolver()

        split_file = data_config['split']

        if self.split in ["wild"]:

            if self.data_config["type"] in ["LSMI", "NUS"]:
                patch_path = Path(data_config["wild_path"])/"patches"
                raw_path = Path(data_config["wild_path"])/"rawnormalized"
            else:
                patch_path = Path(data_config["wild_path"])/"wildpatches"
                raw_path = Path(data_config["wild_path"])/"wildrawnormalized"
        elif self.split in ["mixed"]:
            patch_path = Path(data_config["mixed_path"])/"mixed"
        else:
            patch_path = Path(data_config["processed_path"])/"patches"
        
        patch_paths = sorted(list(patch_path.glob("*.npy")))
     

        self.max_xy = data_config.get("max_xy", None)
        self.max_wp = data_config.get("max_wp", None)
        if self.max_wp is None:
            print("MAX WP not provided in config, please provide it for better training stability.")
        else:
            self.max_wp = np.array(self.max_wp)
        if self.max_xy is None:
            print("MAX XY not provided in config, please provide it for better training stability.")
        else:
            self.max_xy = np.array(self.max_xy)

        if len(patch_paths) == 0:
            temp="test"
        self.single_led_csts = None
        if split == "original_cst":
            #load first patch_path to get forward_matrix
            sample_patch = np.load(patch_paths[0], allow_pickle=True).item()
            self.low_temp_cst = sample_patch["forward_matrix_1"] #should be 3x3 numpy array
            self.high_temp_cst = sample_patch["forward_matrix_2"]
            self.mid_temp_cst = np.zeros_like(self.low_temp_cst[0]), 0

            #make each one of the csts a tensor with shape 3,3, remember the csts are actually tuples of (matrix, temp)
            self.low_temp_cst = (torch.tensor(self.low_temp_cst[0], dtype=torch.float32), self.low_temp_cst[1])
            self.high_temp_cst = (torch.tensor(self.high_temp_cst[0], dtype=torch.float32), self.high_temp_cst[1])
            self.mid_temp_cst = (torch.tensor(self.mid_temp_cst[0], dtype=torch.float32), self.mid_temp_cst[1])
        elif self.data_config["type"] in ["LSMI", "NUS"]:
            #filler using original cst method temporarily, need to change later
            # always load forward matrices from processed patches, not mixed files
            _cst_patches = sorted((Path(data_config["processed_path"]) / "patches").glob("*.npy"))
            sample_patch = np.load(_cst_patches[0], allow_pickle=True).item()
            self.low_temp_cst = sample_patch["forward_matrix_1"] #should be 3x3 numpy array
            self.high_temp_cst = sample_patch["forward_matrix_2"]
            self.mid_temp_cst = np.zeros_like(self.low_temp_cst[0]), 0

            #make each one of the csts a tensor with shape 3,3, remember the csts are actually tuples of (matrix, temp)
            self.low_temp_cst = (torch.tensor(self.low_temp_cst[0], dtype=torch.float32), self.low_temp_cst[1])
            self.high_temp_cst = (torch.tensor(self.high_temp_cst[0], dtype=torch.float32), self.high_temp_cst[1])
            self.mid_temp_cst = (torch.tensor(self.mid_temp_cst[0], dtype=torch.float32), self.mid_temp_cst[1])

        else: #this is one way of getting the csts (for other datasets we will have to do something else)
            temp_patch_path = Path(data_config["processed_path"])/"patches" #I do this to be in convention with simulated area
            temp_patch_paths  = sorted(list(temp_patch_path.glob("*.npy")))

            self.temp_patch_paths = [p for p in temp_patch_paths 
                        if any(f"{i}" in p.stem for i in range(797, 834))]
            #assert that all temps_paths are found
            assert len(self.temp_patch_paths) == 37, f"Expected 37 temperature patch files, but found {len(self.temp_patch_paths)}. Check the patch files in {patch_path}."

            self.low_temp_cst , self.high_temp_cst, self.mid_temp_cst = get_calibrated_csts(self.temp_patch_paths, data_config, self.target)

            single_led_paths = sorted([p for p in temp_patch_paths
                        if any(p.stem == f"light{i:03d}" for i in range(0, 7))])
            assert len(single_led_paths) == 7, f"Expected 7 single-LED patches (light000-light006), found {len(single_led_paths)} in {temp_patch_path}."
            self.single_led_csts = get_single_led_csts(single_led_paths, self.target, max_xy=self.max_xy)

        if self.split not in ["temps", "original_cst", "wild", "mixed"]:
            self.split_file = Path(data_config["processed_path"]) / "../.."/ "splits" / f"{split_file}.yaml"
            self.split_indices = yaml.safe_load(open(self.split_file, 'r'))[self.split]
            #filter patch paths to only those in split indices
            self.patch_paths = sorted([p for p in patch_paths if  p.stem in self.split_indices])
        elif self.split == "wild":
            self.patch_paths = patch_paths
            #load raw paths
            self.raw_paths = sorted(list(raw_path.glob("*.npy")))

            if data_config["type"] in ["LSMI"]:
                self.split_file = Path(data_config["processed_path"]) / "../.."/ "splits" / f"{split_file}.yaml"
                self.split_indices = yaml.safe_load(open(self.split_file, 'r'))["test"]
                self.patch_paths = sorted([p for p in patch_paths if  p.stem in self.split_indices])
                self.raw_paths = sorted([p for p in self.raw_paths if  p.stem in self.split_indices])
                
            assert len(self.raw_paths) == len(self.patch_paths), f"Number of raw paths {len(self.raw_paths)} does not match number of patch paths {len(self.patch_paths)}"
        elif self.split == "mixed":
            self.split_file = Path(data_config["processed_path"]) / "../.."/ "splits" / f"{split_file}.yaml"

            self.split_indices = yaml.safe_load(open(self.split_file, 'r'))["test_places"]
            # self.split_indices=[p[:-7] for p in self.split_indices] #chop of the last section(_chartX)
            self.patch_paths = sorted([p for p in patch_paths if   p.stem.split('_')[0] in self.split_indices])
            self.raw_paths = sorted([p for p in patch_paths if  p.stem.split('_')[0] in self.split_indices])

            
        self.cache = {}


        
    def __len__(self):
        if self.split in ["temps", "original_cst"]:
            return 1 #special case for temps
        elif self.split == "train" and self.use_ITERS:
            return _dataset_iters(self.data_config) * self.batch_size
        else:
            return len(self.patch_paths)
    
    def __getitem__(self, idx):
        idx = idx % len(self.patch_paths)
        # if len(self.cache) == len(self.patch_paths) and self.split == "train": #all of cache is loaded
        #     data_point = self.cache[idx]
        #     #return data_point
        #     xy = data_point["xy"]
        #     xy += np.random.normal(0, 0.001, xy.shape) #add random gaussian noise to xy during training
        #     xy = np.clip(xy, -1, 1)  #clip xy to be between -1 and 1
        #     closest_idx = min(self.cache.keys(), key=lambda k: np.linalg.norm(self.cache[k]["xy"] - xy)) #find point in cache with closest xy
        #     data_point = self.cache[closest_idx]
        #     # else: #same point is closest (but we added noise, so we need to update xy)
        #     #data_point["xy"] = xy
        #     return data_point

        if idx in self.cache:
            return self.cache[idx]

        patch_path = Path(self.patch_paths[idx])
        patch_data = np.load(patch_path, allow_pickle=True).item()
        if self.split in ["mixed"]:
            illumination_map = patch_data["illumination_map"]
            rawfull_img = patch_data["original_raw"]
            patch_image= patch_data["original_patches"] #Checkers * Height * Width * 3
            mixture_map = patch_data["mixture_map"]
            chart_map = patch_data["chart_map"]
            mixture_map_patches = patch_data["mixture_map_patches"]
            illumination_map_patches = patch_data["illumination_map_patches"]
            wp = patch_data['whitepoints'] #Illumination*Chart
                


          
            #get brightest in patch_image
            #OLD
            # brightness = patch_image.mean((1,2,3))
            # idx = brightness.argmax()                      # index of brightest
            # brightest_patch_image = patch_image[idx:idx+1]           # [1, H, W, C]
            # brightest_patch_image = np.repeat(brightest_patch_image, wp.shape[0], axis=0)
            #illumination_map_patches_for_max_illum = np.repeat(illumination_map_patches[idx:idx+1], wp.shape[0], axis=0)  # [Illums, H, W, C]
            #TRYING MAX ILLUMINATION COLOR CHART PER LIGHT SOURCE
            single_illum_means = patch_data["single_illum_patches"].mean(axis=(2,3,4))  # [Illums]
            best_idx = single_illum_means.argmax(axis=1)  # shape: (2,)
            brightest_patch_image = patch_image[best_idx]  # shape: (2, 4, 6, 3)
            illumination_map_patches_for_max_illum = illumination_map_patches[best_idx]  # shape: (2, 4, 6, 3)
            wb_single_illum_patches = np.clip(white_balance(brightest_patch_image, illumination_map_patches_for_max_illum), 0, 1)
            illuminant_colorchecker = wb_single_illum_patches
        else:
            patch_image= patch_data["patches"]
            whitepoint_patch = patch_data["whitepoint"]

          
    

          
        if self.split == "wild":

            #what we want to do for WILD and MXIED
            #New things - > we need white level, and black level, and raw unnormalized
            # load - raw unnormalized (demosaiced)
            # raw_wb = (rawunnormalized - black_level) / whitepoint (same one)
            # wb = raw_wb / (white_level - black_level)


            rawfull_img = np.load(self.raw_paths[idx], allow_pickle=True)*1.0 #raw_normalized
            #make WB green channel 1 (don't change scale too much)
            #whitepoint_patch /= whitepoint_patch[1]
            wbfull_img = np.clip(white_balance(rawfull_img, whitepoint_patch), 0, 1) #white blaance of raw_normalized
            wbfull_img = np.moveaxis(wbfull_img, -1, 0)  # Reshape to C × H × W
            rawfull_img = np.moveaxis(rawfull_img, -1, 0)  # Reshape to C × H × W

            
        if self.split == "mixed":
            wbfull_img = np.clip(white_balance(rawfull_img, illumination_map), 0, 1)
            wbfull_img = np.moveaxis(wbfull_img, -1, 0)  # Reshape to C × H × W
            rawfull_img = np.moveaxis(rawfull_img, -1, 0) 
            
            wp = whitepoint(wp)  # Illumination * 2

             # Reshape to C × H × W
            illumination_map = whitepoint(illumination_map)
            illumination_map = np.moveaxis(illumination_map, -1, 0)  # Reshape to C × H × W

            wb_img = np.clip(white_balance(patch_image, illumination_map_patches), 0, 1)
            wb_img = np.moveaxis(wb_img, -1, 1)  # Reshape to Illum * C × H × W
            patch_image = np.moveaxis(patch_image, -1, 1)  # Reshape to Illum * C × H × W
            xys = []
            ccts = []
            for x in wp:
                xy, cct = get_xy_and_cct(self.low_temp_cst, self.high_temp_cst, x)
                xys.append(xy[0])
                ccts.append(cct)


            xy= np.array(xys)
            cct = np.array(ccts)

            #stack target for number of charts len(wp)
            target = np.tile(self.target[np.newaxis, :, :, :], (wb_img.shape[0], 1, 1, 1))

            per_illum_target = np.tile(self.target[np.newaxis, :, :, :], (wp.shape[0], 1, 1, 1))
        else:
            if self.wp_error:
                #perturb the whitepoint_patch by 1degree
                def rotate_whitepoint_random(a, degrees, clip=True):
                    a = np.asarray(a, float)
                    th = np.radians(degrees)

                    # Random axis perpendicular to a  -> exact angular separation = degrees
                    ah = a / (np.linalg.norm(a) + 1e-12)
                    #set seed of numpy to sum of a for reproducibility
                    np.random.seed(int(np.sum(a)*1e6)% (2**32 -1))
                    r = np.random.randn(3)
                    axis = r - ah * np.dot(r, ah)
                    n = np.linalg.norm(axis)
                    if n < 1e-12:
                        # deterministic fallback axis ⟂ a
                        axis = np.array([ah[1]-ah[2], ah[2]-ah[0], ah[0]-ah[1]])
                        n = np.linalg.norm(axis)
                    axis /= n

                    # Rodrigues’ rotation
                    c, s = np.cos(th), np.sin(th)
                    K = np.array([[0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0]])
                    R = np.eye(3)*c + (1-c)*np.outer(axis, axis) + s*K
                    v = R @ a

                    # Single scalar stretch so no component decreases
                    tiny = 1e-12
                    scale = np.max(np.where(v < a, a/np.maximum(v, tiny), 1.0))
                    v *= max(1.0, float(scale))
                    if clip:
                        v = np.clip(v, 0, np.inf)
                    return v
                
                def ang_err(a, b):
                    a, b = np.array(a), np.array(b)
                    return np.degrees(np.arccos(np.clip(np.dot(a,b) /
                        (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)))

                    
                rotated_whitepoint_patch = rotate_whitepoint_random(whitepoint_patch, degrees=self.wp_error)
                ang_err_diff = ang_err(whitepoint_patch, rotated_whitepoint_patch)
                whitepoint_patch = rotated_whitepoint_patch

            wb_img = np.clip(white_balance(patch_image, whitepoint_patch),0,1)  # Apply white balance patch transformation
            wb_img = np.moveaxis(wb_img, -1, 0)  # Reshape to C × H × W
            patch_image = np.moveaxis(patch_image, -1, 0)  # Reshape to C × H × W
            wp = whitepoint(whitepoint_patch)
           
            xy, cct = get_xy_and_cct(self.low_temp_cst, self.high_temp_cst, wp)
            xy = xy[0]
            target = self.target


            #best_fit_matrix = self.solver(torch.tensor(wb_img).unsqueeze(0), torch.tensor(target).unsqueeze(0))[0] #dummy call to initialize solver




        wp_unnormalized = wp.copy()
        xy_unnormalized = xy.copy()
        cct_unnormalized = cct.copy()

        # normalize to [-1, 1] with numpy
        wp  = (wp  / self.max_wp)  * 2 - 1
        xy  = (xy  / self.max_xy)  * 2 - 1
        #clip between -1 and 1
        wp = np.clip(wp, -1, 1)
        xy = np.clip(xy, -1, 1)

        cct = normalize_cct(cct)

        # spectrally-measured GT chromaticity & CCT (used by 7cst / RBF2DXY models)
        xy_from_spectra = patch_data.get("xy_from_spectra")
        if xy_from_spectra is not None:
            import colour
            xy_spectra_arr = np.asarray(xy_from_spectra, dtype=np.float64).reshape(2)
            cct_gt_raw = float(colour.temperature.xy_to_CCT(xy_spectra_arr))
            xy_gt = np.clip((xy_spectra_arr / self.max_xy) * 2 - 1, -1, 1) if self.max_xy is not None else xy_spectra_arr
            cct_gt = normalize_cct(np.array([cct_gt_raw]))  # shape (1,), matches cct
        else:
            # Fallback for datasets without spectral xy (LSMI/NUS) — reuse the iterative-solver values.
            xy_gt = xy
            cct_gt = cct

        wp_raw = patch_data.get("whitepoint", patch_data.get("whitepoints"))



        angles = rgb_to_az_el(wp_raw)

        values = {'wp':wp,'ang':angles, 'xy':xy, 'cct':cct,
                  'xy_gt': xy_gt, 'cct_gt': cct_gt,
                  "wp_unnormalized":wp_unnormalized, "xy_unnormalized":xy_unnormalized, "cct_unnormalized":cct_unnormalized,
                  'rawpatch_img': patch_image, 'wb_img':wb_img, 'target':target,
                   "path":str(patch_path)}
        if self.split == "wild":
            values['rawfull_img'] = rawfull_img
            values['wbfull_img'] = wbfull_img
            values['raw_path'] = str(self.raw_paths[idx])
        if self.split in ["mixed"]:
            values['rawfull_img'] = rawfull_img
            values['wbfull_img'] = wbfull_img
            values['mixture_map'] = mixture_map
            values['illumination_map'] = illumination_map
            values['mixture_map_patches'] = mixture_map_patches
            values['illumination_map_patches'] = illumination_map_patches
            values['illuminant_colorchecker'] = illuminant_colorchecker
            values['per_illum_target'] = per_illum_target
            values['chart_map'] = chart_map

        self.cache[idx] = values
        return values

def whitepoint(wp):
    wp = np.asarray(wp)
    eps = 1e-8

    if wp.ndim == 1:
        # Single whitepoint: shape (3,)
        r, g, b = wp

        return np.array([r / (g + eps), b / (g + eps)])

    elif wp.ndim == 2:
        # List of whitepoints: shape (N, 3)
        r = wp[:, 0]
        g = wp[:, 1]
        b = wp[:, 2]
        rg = r / (g + eps)
        bg = b / (g + eps)
        return np.stack([rg, bg], axis=-1)  # (N, 2)

    elif wp.ndim == 3:
        # Illumination map: shape (H, W, 3)
        r = wp[..., 0]
        g = wp[..., 1]
        b = wp[..., 2]
        rg = r / (g + eps)
        bg = b / (g + eps)
        return np.stack([rg, bg], axis=-1)  # (H, W, 2)

    else:
        raise ValueError(f"Unexpected shape {wp.shape}, expected (3,), (N, 3), or (H, W, 3)")

class FullSimDataset(Dataset):
    def __init__(self, data_config, scale=1, batch_size=1, mode="simple"):
        """
        Initialize the FullSimDataset with simulated data and configuration.
        
        Args:
            data_config (dict): Contains 'src_path' and 'target_path' for loading npy files.
            scale (int): Optional scale factor for oversampling.
        """
        self.data_config = data_config
        if data_config["chart_type"] == 24:
            self.target = np.load(data_config["target_24"])
            #move the last channel to the first position
            self.target = np.moveaxis(self.target, -1, 0)
        else:
            raise ValueError("Unsupported patch format. Only 24 is supported.")
        self.batch_size = batch_size
        self.src_path = data_config["src_path"]
        self.target_path = data_config["target_path"]
        self.src_patches = np.load(self.src_path)  # shape: (834, 1270, 3)
        self.target_patches = np.load(self.target_path)  # shape: (834, 1270, 3)
  
        # Extract the white patch (last column in reflectances)
        self.white_point = self.src_patches[:, -1, :]  # shape: (834, 3)

        temp_patch_path = Path(data_config["processed_path_for_temps"])/"patches" #I do this to be in convention with simulated area
        temp_patch_paths  = sorted(list(temp_patch_path.glob("*.npy")))

        self.temp_patch_paths = [p for p in temp_patch_paths 
                    if any(f"{i}" in p.stem for i in range(797, 834))]
        #assert that all temps_paths are foun
        assert len(self.temp_patch_paths) == 37, f"Expected 37 temperature patch files, but found {len(self.temp_patch_paths)}."
        self.low_temp_cst , self.high_temp_cst, self.mid_temp_cst = get_calibrated_csts(self.temp_patch_paths, data_config, self.target)



        # Remove the white patch from src and target
        self.src_patches = self.src_patches[:, :-1, :]  # shape: (834, 1269, 3)
        self.target_patches = self.target_patches[:, :-1, :]  # shape: (834, 1269, 3)

        self.max_xy = data_config.get("max_xy", None)
        self.max_wp = data_config.get("max_wp", None)
        if self.max_wp is None:
            print("MAX WP not provided in config, please provide it for better training stability.")
        else:
            self.max_wp = np.array(self.max_wp)
        if self.max_xy is None:
            print("MAX XY not provided in config, please provide it for better training stability.")
        else:
            self.max_xy = np.array(self.max_xy)


        self.num_illuminations, self.num_reflectances, _ = self.src_patches.shape
        self.total_samples = self.num_illuminations * self.num_reflectances
        self.scale = scale
        self.illum_cache = {}
        self.mode = mode

    def __len__(self):
        return _dataset_iters(self.data_config) * self.batch_size

    def __getitem__(self, idx):
        # Randomly pick a (illumination, reflectance) pair
        idx = idx % self.total_samples

        illum_idx = idx // self.num_reflectances

        # if len(self.illum_cache) == len(self.num_illuminations) and self.split == "train": #all of cache is loaded
        #     data_point = self.illum_cache[illum_idx]
        #     xy = data_point["xy"]
        #     #add random gaussian noise to xy during training
        #     xy += np.random.normal(0, 0.1, xy.shape)
        #     #find point in cache with closest xy
        #     illum_idx = min(self.illum_cache.keys(), key=lambda k: np.linalg.norm(self.illum_cache[k]["xy"] - xy))
        #     using_cache = True

        if self.mode=="simple":
            refl_idx = idx % self.num_reflectances

            input_rgb = self.src_patches[illum_idx, refl_idx]
            target_rgb = self.target_patches[illum_idx, refl_idx]
            white_rgb = self.white_point[illum_idx]

            # White balance
            wb_img = input_rgb / (white_rgb + 1e-8)  # avoid division by zero

            # Reshape to C × H × W where H = 1 and W = 1
            wb_img = wb_img.reshape(3, 1, 1).astype(np.float32)
            target_rgb = target_rgb.reshape(3, 1, 1).astype(np.float32)
        else:
            #grab a random number between (0 and self.num_reflectances -24)
            refl_start_idx = np.random.randint(0, self.num_reflectances - 24)
            refl_end_idx = refl_start_idx + 24
            input_rgb = self.src_patches[illum_idx, refl_start_idx:refl_end_idx]
            target_rgb = self.target_patches[illum_idx, refl_start_idx:refl_end_idx]
            white_rgb = self.white_point[illum_idx]

            # White balance
            wb_img = input_rgb / (white_rgb + 1e-8)  #avoid division by zero
            # Reshape to C × H × W where H = 4 and W = 6
            wb_img = np.moveaxis(wb_img, -1, 0).astype(np.float32).reshape(3, 4, 6)  # C × H × W

            target_rgb = np.moveaxis(target_rgb, -1, 0).astype(np.float32).reshape(3, 4, 6)  # C × H × W

        if illum_idx in self.illum_cache:
            wp,xy, cct =  self.illum_cache[illum_idx]
        else:
            # Compute R/G and B/G
            wp = whitepoint(white_rgb)
            xy, cct = get_xy_and_cct(self.low_temp_cst, self.high_temp_cst, wp)
            xy = xy[0]

            # normalize to [-1, 1] with numpy
            wp  = (wp  / self.max_wp)  * 2 - 1
            xy  = (xy  / self.max_xy)  * 2 - 1
            cct = normalize_cct(cct)
            self.illum_cache[illum_idx] = (wp, xy, cct)


        out = {
            'wb_img': wb_img,
            'target': target_rgb,
            'ang': rgb_to_az_el(white_rgb),
            'wp': wp,
            'xy': xy,
            'cct': cct,
            'illum_idx': illum_idx
        }
        return out


# def whitepoint_double(data):
#     wp = data["whitepoint"]
#     rg_bg = wp[[0, 2]] / wp[1]  # Calculate r/g and b/g
#     return np.concatenate([rg_bg, rg_bg])  # Repeat r/g, b/g twice

# def xy_coords(data):
#     xy= data["xy"]
#     return np.array(xy)
def get_yaw_azimuth(RGB):
    """
    Calculate yaw and azimuth angles from an RGB vector.
    
    Args:
        RGB (array-like): An array or list containing R, G, B values.
    Returns:
        tuple: A tuple containing yaw and azimuth normalized 0,1.
    """
    assert len(RGB) == 3, "Input RGB must have three components."
    assert type(RGB) in [list, np.ndarray], f"Input RGB must be a list or numpy array, got {type(RGB)}"
    R, G, B = RGB
    norm = np.sqrt(R**2 + G**2 + B**2) + 1e-8  # avoid division by zero
    r, g, b = R / norm, G / norm, B / norm

    yaw = np.arctan2(b, r)  # Yaw angle in rad
    azimuth = np.arccos(g)  # Azimuth angle in rad
    #rads are bound to pi/2 since rgb values are always positive
    # Since both angles are in [0, π/2] for positive RGB
    yaw = yaw / (np.pi / 2)  # Normalize to [0, 1]
    azimuth = azimuth / (np.pi / 2)  # Normalize to [0, 1]
    assert 0 <= yaw <= 1, f"Yaw {yaw} out of bounds"
    assert 0 <= azimuth <= 1, f"Azimuth {azimuth} out of bounds"
    yaw -= 0.5
    azimuth -= 0.5



    return np.array([yaw, azimuth])



# def rgb_to_az_el(RGB):
#     R, G, B = RGB
#     norm = np.sqrt(R**2 + G**2 + B**2) + 1e-8
#     r, g, b = R / norm, G / norm, B / norm

#     a = np.arctan2(g, r)          # azimuth
#     b = np.arcsin(np.clip(b, -1.0, 1.0))  # elevation
#     return np.array([a, b], dtype=np.float32)
# def rgb_to_az_el(RGB):
#     # R, G, B = RGB

#     R, G, B = RGB
#     az= np.arctan2(G,R)
#     el=np.arctan2(B,R)
#     # Normalize to [-1, 1]
#     az = az / np.pi * 2  # Normalize to [-1, 1]
#     el = el / (np.pi / 2) * 2 - 1  # Normalize to [-1, 1]

#     return np.array([az, el], dtype=np.float32)


def rgb_to_az_el(RGB):
    # R, G, B = RGB
    if RGB.ndim == 1:
        RGB = RGB[np.newaxis, :]  # Add batch dimension if input is a single RGB vector
    R, G, B = RGB[:,0], RGB[:,1], RGB[:,2]
    norm= np.sqrt(R**2 + G**2 + B**2) + 1e-8
    r, g, b = R/norm, G/norm, B/norm
 
    az = np.arctan2(r, g)
    el = np.arcsin(b)

    # az = az / np.pi * 2  # Normalize to [-1, 1]
    az = az / (np.pi / 2) * 2 - 1
    el = el / (np.pi / 2) * 2 - 1  # Normalize to [-1, 1]

    out=np.stack([az, el], axis=-1).astype(np.float32)
  
    if out.shape[0] == 1:
        return out[0]  # Return 1D array if input was a single RGB vector
    return out



# def rgb_to_rb_chroma(RGB):
#     R, G, B = RGB
#     s = (R + G + B) + 1e-8
#     r = R / s
#     b = B / s
#     a = 2.0 * r - 1.0
#     b = 2.0 * b - 1.0
#     return np.array([a, b], dtype=np.float32)

# def rgb_to_logratios_tanh(RGB):
#     R, G, B = RGB
#     eps = 1e-8
#     x1 = np.log((R + eps) / (G + eps))
#     x2 = np.log((B + eps) / (G + eps))
#     a = np.tanh(x1)
#     b = np.tanh(x2)
#     return np.array([a, b], dtype=np.float32)