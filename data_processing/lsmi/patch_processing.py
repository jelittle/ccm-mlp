"""Extract ColorChecker patches from LSMI, including the multi-illuminant set.

Run the upstream LSMI preprocessing FIRST (0_cvt2tiff.py -> 1_make_mixture_map.py
-> 2_preprocess_data.py from https://github.com/DY112/LSMI-dataset); this consumes
its output and writes patches/, rawnormalized/, raw/ and mixed/.

`mixed/` is the multi-illuminant set -- the part this project contributes -- and is
what the `mixed` split in utilities/dataset.py reads.

    python data_processing/lsmi/patch_processing.py sony
"""

# Make the repo's packages importable when this script is run directly, i.e.
# `python data_processing/.../this.py` without `pip install -e .` first.
# No-op once the package is installed.
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "training" / "utilities").is_dir():
        for _p in (_root / "training", _root / "third_party"):
            if str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        break


import argparse
import concurrent.futures
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from camera_pipeline.pipeline_lib.pipeline_utils import get_metadata
from utilities.paths import resolve

RAW_TYPES = {"galaxy": "dng", "sony": "arw", "nikon": "nef"}

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("camera", nargs="?", default="sony", choices=sorted(RAW_TYPES))
_parser.add_argument("--src", type=Path, default=None,
                     help="LSMI dataset root holding the per-camera folders")
_parser.add_argument("--dest", type=Path, default=None,
                     help="where to write patches/rawnormalized/raw/mixed")
_parser.add_argument("--limit", type=int, default=None,
                     help="process only the first N scenes (mixed records are ~330 MB "
                          "each, so a full run needs a lot of disk)")
_args, _ = _parser.parse_known_args()

CAMERA = _args.camera
RAW_TYPE = RAW_TYPES[CAMERA]

src_path = _args.src if _args.src else Path(resolve("${raw_data_root}/LSMI"))
dest_path = _args.dest if _args.dest else Path(resolve("${data_root}/LSMI/colorchecker24"))

CONFIG = {
    "save": True,
    "save_single": True,
    "save_mixed": True,
    "patch_path": os.path.join(dest_path, CAMERA, "patches"),
    "raw_normalized_path": os.path.join(dest_path, CAMERA, "rawnormalized"),
    "raw_path": os.path.join(dest_path, CAMERA, "raw"),
    "mixed_path": os.path.join(dest_path, CAMERA, "mixed"),
    "src_dir": os.path.join(src_path, CAMERA),
}




# dest_path = Path("test")
if CAMERA == "galaxy":
    WHITE_LEVEL = 1023
    BLACK_LEVEL = 0 #from LSMI code
elif CAMERA == "sony":
    WHITE_LEVEL = 4095#from LSMI code
    BLACK_LEVEL = 128
elif CAMERA == "nikon":
    WHITE_LEVEL = 15520 #from LSMI code
    BLACK_LEVEL = 600
else:
    raise ValueError("White level value for camera not defined")
def load_lsmi_tiff(image_path, substract_black_level=False):
    """Load and normalize a LSMI TIFF image and return as a numpy array"""
    raw_tiff_bgr = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    if substract_black_level:
        raw_tiff_bgr = np.clip(raw_tiff_bgr.astype(np.float32) - BLACK_LEVEL, 0, None)
    if raw_tiff_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    raw_tiff = cv2.cvtColor(raw_tiff_bgr, cv2.COLOR_BGR2RGB)#convert BGR to RGB
    raw_tiff = raw_tiff.astype(np.float32)/WHITE_LEVEL#scale to 0,1
    return raw_tiff
def get_mixture_map(placeInfo):
    place=placeInfo["place"]
    illums=placeInfo["info"]["NumOfLights"]
    map={2:"12",3:"123"}
    src_dir=CONFIG["src_dir"]
    mixture_map_path=os.path.join(src_dir, place, f"{place}_{map.get(illums)}.npy")
    mixture_map=np.load(mixture_map_path)
    return mixture_map
def get_patch_coordinates(corners, cols=6, rows=4):
    outer_margin_frac = 0.16  # fraction of a patch on each edge (left/right/top/bottom)
    gutter_frac = 0.02        # fraction of a patch between adjacent patches

    # total width/height in "patch units"
    W = cols * 1.0 + (cols - 1) * gutter_frac + 2 * outer_margin_frac
    H = rows * 1.0 + (rows - 1) * gutter_frac + 2 * outer_margin_frac
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    dst = np.array([tl, tr, br, bl], dtype=np.float32)

    # source rectangle in "checker coordinates" with gutters and margins
    src = np.array([[0, 0],
                    [W, 0],
                    [W, H],
                    [0, H]], dtype=np.float32)

    # homography from checker coords -> image
    Hmat = cv2.getPerspectiveTransform(src, dst)

    # build centers with gutters and outer margins
    cc = []
    for r in range(rows):
        for c in range(cols):
            x = outer_margin_frac + 0.5 + c * (1.0 + gutter_frac)
            y = outer_margin_frac + 0.5 + r * (1.0 + gutter_frac)
            cc.append([x, y])
    cc = np.array(cc, dtype=np.float32).reshape(-1, 1, 2)

    # map centers to image via homography
    centers_img = cv2.perspectiveTransform(cc, Hmat).reshape(-1, 2)
    return centers_img
  
def get_patch_average(image, center_x, center_y, radius=2):
    """Extract and average a square patch around a center point"""
    # Ensure integer coordinates
  
    x, y = int(np.round(center_x/2)), int(np.round(center_y/2))
    patch=image[ y-radius:y+radius,x-radius:x+radius,:]
    
    return np.mean(patch, axis=(0, 1))

def check_patch_clipping(image, center_x, center_y, radius=2, white_level=WHITE_LEVEL):

    x, y = int(np.round(center_x/2)), int(np.round(center_y/2))
    # x,y = int(center_x), int(center_y)
    patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]

    if np.any(patch == True):
        return True
    return False
def build_patch_coordinates(placeInfo):
    all_patch_coordinates=[]
    for key in placeInfo["MCCCoord"]:
        corners=placeInfo["MCCCoord"][key]
        patch_coords=get_patch_coordinates(corners)
        all_patch_coordinates.append(patch_coords)

    all_patch_coordinates=np.array(all_patch_coordinates)
    return all_patch_coordinates
def _load_image_pairs(place,i):
    illum_image_path = os.path.join(src_path, CAMERA,place, place +f"_{i}.tiff")
    # Map i to the corresponding full illumination image filename
    full_illum_map = {
        1: f"{place}_1.tiff",
        2: f"{place}_12.tiff",
        3: f"{place}_123.tiff"
    }
    full_illum_image_path = os.path.join(src_path, CAMERA, place, full_illum_map.get(i, f"{place}_{i}.tiff"))

    full_raw_tiff_bgr = np.clip(cv2.imread(full_illum_image_path,cv2.IMREAD_UNCHANGED).astype(np.float32) - BLACK_LEVEL, 0, None)
    substract_black_level= BLACK_LEVEL if i ==1 else False
    # substract_black_level= 0
    raw_tiff_bgr = np.clip(cv2.imread(illum_image_path,cv2.IMREAD_UNCHANGED).astype(np.float32) - substract_black_level, 0, None)


    if full_raw_tiff_bgr is None:
        raise FileNotFoundError(f"Image not found: {full_illum_image_path}")

    if raw_tiff_bgr is None:
        raise FileNotFoundError(f"Image not found: {illum_image_path}")
    full_raw_tiff = cv2.cvtColor(full_raw_tiff_bgr, cv2.COLOR_BGR2RGB)#convert BGR to RGB
    raw_tiff = cv2.cvtColor(raw_tiff_bgr, cv2.COLOR_BGR2RGB)#convert BGR to RGB

  
    clip_mask = (full_raw_tiff >= WHITE_LEVEL*.99)

    
  

    return raw_tiff, full_raw_tiff, clip_mask

def _process_chart(chart, raw_tiff, full_raw_tiff, clip_mask,mixture_map, rows=4, cols=6):
    """Process a single chart and return values and clipping status"""
    chart_values = []
    full_chart_values = []
    mixture_map_values = []
    chart_has_clipping = False
    
    for x, y in chart:
        patch_avg = get_patch_average(raw_tiff, x, y, radius=2)
        full_patch_avg = get_patch_average(full_raw_tiff, x, y, radius=2)
        mixture_map_value = get_patch_average(mixture_map, x, y, radius=2)
        
        if check_patch_clipping(clip_mask, x, y, radius=2):
            chart_has_clipping = True
        
        chart_values.append(patch_avg)
        full_chart_values.append(full_patch_avg)
        mixture_map_values.append(mixture_map_value)
  
    
    # Normalize values
    chart_values = np.clip(np.array(chart_values).reshape(rows, cols, 3) / (WHITE_LEVEL - BLACK_LEVEL), 0.0, 1.0)
    if raw_tiff.max() != 1:
       
        raw_tiff = raw_tiff.astype(np.float32) / (WHITE_LEVEL - BLACK_LEVEL)
        raw_tiff = np.clip(raw_tiff, 0, 1)
    else:
        raw_tiff = raw_tiff
    full_chart_values = np.clip(np.array(full_chart_values).reshape(rows, cols, 3) /(WHITE_LEVEL),0,1)
    mixture_map_values = np.array(mixture_map_values).reshape(rows, cols, -1)

    # Create a mask for values below 0.02
    chart_threshold_mask = np.mean(chart_values, axis=-1) < 0.01
    threshold_mask = np.mean(raw_tiff, axis=-1) < 0.01

    return chart_values, full_chart_values, chart_has_clipping, mixture_map_values, chart_threshold_mask, threshold_mask
def get_illumination_map(mixture_map, lights):

    return np.einsum('ijm,mr->ijr', mixture_map, lights)
def _save_scene_data(scene_data):
    """Save processed image data to files"""

    def save_by_image(scene_data,raw_path_end_dict_src,raw_path_end_dict_dst):
        place=scene_data["place"]
        chart_map=scene_data["chart_map"]
   
        for image_data in scene_data["image_data"]:
           
            light_index=image_data["light_index"]
            image_chart_values=image_data["patches"]
            whitepoints=image_data["whitepoints"]
            t1,t2=image_data["forward_matrix_1"],image_data["forward_matrix_2"]
            src_dir=CONFIG["src_dir"]

    
            src_path_raw=os.path.join(src_dir, place,raw_path_end_dict_src.get(light_index))
            
            patch_path=CONFIG["patch_path"]
            raw_normalized_dir=CONFIG["raw_normalized_path"]
            raw_path=CONFIG["raw_path"]

            os.makedirs(patch_path, exist_ok=True)
            os.makedirs(raw_normalized_dir, exist_ok=True)
            os.makedirs(raw_path, exist_ok=True)
            # Save raw DNG file in a new location
            raw_dng_src = os.path.join(src_path_raw)
            raw_dng_dst = os.path.join(raw_path, raw_path_end_dict_dst.get(light_index))
            os.makedirs(os.path.dirname(raw_dng_dst), exist_ok=True)


            #save raw normalized image
            raw_image_path = os.path.join(src_dir, place, f"{place}_{light_index}.tiff")
            scaled_raw_tiff= load_lsmi_tiff(raw_image_path, True if light_index==1 else False)
      

            

        
            #save raw dng if it exists
            if os.path.exists(raw_dng_src):
                shutil.copy2(raw_dng_src, raw_dng_dst)
                pass
            else:
                print(f"Warning: Raw DNG file not found: {raw_dng_src}")

            #save patches with required data
            # for i in range(image_chart_values.shape[0]):
    
         

            max_mean_index = np.argmax(np.mean(image_chart_values, axis=(1, 2, 3)))

            # Create mask for the selected chart based on RGB channels
            chart_color = np.zeros((3,), dtype=np.uint8)
            chart_color[max_mean_index % 3] = 255
            
            # Find pixels that match the selected chart's color - this is now a boolean mask
            single_chart_map = np.all(chart_map == chart_color, axis=2)
            
            # Save the single chart map as a PNG image
  




            data={}
            data["patches"]=image_chart_values[max_mean_index]
            data["whitepoint"]=whitepoints[max_mean_index]
            data["xy_from_spectra"]=None
            data["forward_matrix_1"]=t1
            data["forward_matrix_2"]=t2
            data["threshold_mask"]=image_data["threshold_mask"]
            data["chart_threshold_mask"]=image_data["chart_threshold_mask"][max_mean_index]
            data["chart_map"]=single_chart_map
            file_name=f"{place}_light{light_index}_chart{max_mean_index+1}.npy"

            np.save(os.path.join(patch_path,file_name),data)
            # save raw illum once
            np.save(os.path.join(raw_normalized_dir,file_name),scaled_raw_tiff)
 
        

    def save_for_mixed(scene_data):
        num_illums=len(scene_data["image_data"])
        place=scene_data["place"]
        raw_tiff_map={
            1: f"{place}_1.tiff",
            2: f"{place}_12.tiff",
            3: f"{place}_123.tiff"
        }
        # print(scene_data.keys())
        # print(scene_data["image_data"][0].keys())
        src_dir=CONFIG["src_dir"]
        src_path_raw=os.path.join(src_dir, place, raw_tiff_map.get(num_illums))
        #load tiff(pre black level substracted)
    
        

        mixed_dir=CONFIG["mixed_path"]


        light_idxs=[]
        illum_patches=[]
        raw_images=[]
        whitepoints = []
        illumination_map_patches=[]
     
        for image_data in scene_data["image_data"]:
            whitepoints.append(image_data["whitepoints"])
            light_idx = image_data["light_index"]
            light_idxs.append(image_data["light_index"])
            illum_patches.append(image_data["patches"])
            illumination_map_patches.append(image_data["chart_illumination_map"])
            t1,t2=image_data["forward_matrix_1"],image_data["forward_matrix_2"]
            raw_image_path = os.path.join(src_dir, place, f"{place}_{light_idx}.tiff")
            raw_images.append(load_lsmi_tiff(raw_image_path))

        light_idxs = np.array(light_idxs)
        single_illum_patches = np.array(illum_patches)

        max_mean_index = np.argmax(np.mean(single_illum_patches, axis=(1, 2, 3)))
        # max_mean_index = scene_data["max_mean_index"]


        illumination_map_patches = np.array(illumination_map_patches)
        # single_illum_raw = np.array(raw_images)
        normalized_bl_original_raw=load_lsmi_tiff(src_path_raw, substract_black_level=True)
        #get all required orginal patch data from last image(will have full illum)
        full_illum_image_data=scene_data["image_data"][-1]#this a single illum but has full illum info
        original_patches=full_illum_image_data["real_patches"]
        # original_whitepoints=full_illum_image_data["real_whitepoints"]
        illumination_map_patches=full_illum_image_data["chart_illumination_map"]
        mixture_map_patches=full_illum_image_data["mixture_map"]

   
        
        lights=[scene_data["info"][f"Light{num_illums}"] for num_illums in range(1,num_illums+1)]
        lights=np.array(lights)
      
    
        
     
        saved_obj={
            "place":place,
            "info":scene_data["info"],
            "chart_map":scene_data["chart_map"],
            "single_illum_patches":single_illum_patches, #need for bestfitsolver.
            "max_mean_index":max_mean_index, #need for bestfitsolver.
            # "single_illum_raw":single_illum_raw,
            "mixture_map_patches":mixture_map_patches,
            "illumination_map_patches":illumination_map_patches,
            "original_patches":original_patches,
            "original_raw":normalized_bl_original_raw,
            "light_idxs":light_idxs,
            "whitepoints":lights,
            "mixture_map":scene_data["mixture_map"],
            "illumination_map":scene_data["illumination_map"], #whitepoint value(illumination effect) at each pixel
            "forward_matrix_1":t1,
            "forward_matrix_2":t2
        }

        os.makedirs(mixed_dir, exist_ok=True)
        illum = ''.join(str(i) for i in range(1, num_illums + 1))
        # print("saving mixed for ", place, illum)
        np.save(os.path.join(mixed_dir, f"{place}_illum{illum}.npy"), saved_obj)

        
        
    place=scene_data["place"]
    
    raw_path_end_dict_dst={
        1: f"{place}_illum1.{RAW_TYPE}",
        2: f"{place}_illum12.{RAW_TYPE}",
        3: f"{place}_illum123.{RAW_TYPE}"
    }
    raw_path_end_dict_src={
        1: f"{place}_1.{RAW_TYPE}",
        2: f"{place}_12.{RAW_TYPE}",
        3: f"{place}_123.{RAW_TYPE}"
    }

  
    

 
   
    if CONFIG["save_single"]:
        save_by_image(scene_data,raw_path_end_dict_src,raw_path_end_dict_dst)
    if CONFIG["save_mixed"]:
        save_for_mixed(scene_data)


        
def get_forward_matrices(image_path):
    # Suppress stdout and stderr
    if CAMERA in ["sony"]:
        image_path=resolve("${data_root}/LSMI/converted_dngs/sony_converted.dng")
    if CAMERA in ["nikon"]:
        image_path=resolve("${data_root}/LSMI/converted_dngs/nikon_converted.dng")
    metadata = get_metadata(image_path)


    # exit(1)
    forward_matrix_1 = np.array(metadata["forward_matrix_1"], dtype=np.float32).reshape(3, 3)
    forward_matrix_2 = np.array(metadata["forward_matrix_2"], dtype=np.float32).reshape(3, 3)

    calibration_illuminant_1 = metadata["calibration_illuminant_1"][0]
    calibration_illuminant_2 = metadata["calibration_illuminant_2"][0]
    ill_lookup = {17: 2856, 21: 6500}  # A and D65
    #assert the illuminants are in the lookup
    assert calibration_illuminant_1 in ill_lookup, f"Illuminant {calibration_illuminant_1} not in lookup"
    assert calibration_illuminant_2 in ill_lookup, f"Illuminant {calibration_illuminant_2} not in lookup"
    calibration_illuminant_1 = ill_lookup[calibration_illuminant_1]
    calibration_illuminant_2 = ill_lookup[calibration_illuminant_2]
    
    t1 = (forward_matrix_1, calibration_illuminant_1)
    t2 = (forward_matrix_2, calibration_illuminant_2)

    #swap if t1 is higher temp
    if t1[1] > t2[1]:
        t1, t2 = t2, t1

    return t1, t2

def get_whitepoints(scene_data, i):
    """gets whitepoints from scene_data"""
    # Assuming the white patch is the last patch in the chart
    num_illums=scene_data["info"]["NumOfLights"]
    lights=[scene_data["info"][f"Light{num_illums}"] for num_illums in range(1,num_illums+1)]
    lights=np.array(lights)
    #1st row last column
    whitepoint = np.tile(lights[i - 1], (3, 1))
    
    return whitepoint


def get_chart_map_rgb(scene_data):
    """black matrix with each chart assigned a unique color, [1,0,0],[0,1,0],[0,0,1] for chart 1,2,3"""
    H, W = scene_data["mixture_map"].shape[:2]
    c = 3
    chart_map = np.zeros((H, W, c), dtype=np.float32)
    patch_corners = scene_data["info"]["MCCCoord"]
    
    for patch_idx, key in enumerate(patch_corners):
        corners = patch_corners[key]
        
        # Convert corners to the right coordinate system and integer coordinates
        corners_array = np.array(corners, dtype=np.float32) // 2  # Convert to float first
        corners_array = corners_array.astype(np.int32) # Scale down by 2 and convert to int
        
        # Create a mask for this chart
        chart_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Fill the rectangle defined by the 4 corners
        cv2.fillPoly(chart_mask, [corners_array], 1)
        
        # Assign a unique color to this chart
        color = np.zeros((3,), dtype=np.float32)
        color[patch_idx % 3] = 1.0  # Cycle through R, G, B for different charts
        
        # Apply the mask to all color channels
        chart_map += chart_mask[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]
    
    # Save the chart map
    chart_map = np.clip(chart_map, 0, 1)
    chart_map = (chart_map * 255).astype(np.uint8)
    # cv2.imwrite("./test_patch.png", chart_map)
    return chart_map
    
def _handle_scene_data(place, placeInfo):
    """Process patch values for all illumination images in a scene"""
    # Initialize counters
    stats = {
        'total_images': 0,
        'total_clipped_images': 0,
        'total_charts': 0,
        'total_charts_clipped': 0,
        'used_in_dataset': 0,
    }

    n_lights = placeInfo["NumOfLights"]
    all_patch_coordinates = build_patch_coordinates(placeInfo)
    

    raw_path = os.path.join(src_path, CAMERA, place, f"{place}_1.{RAW_TYPE}")
    scene_data = {
        "place": place,
        "info": placeInfo,
        "image_data": [],
        "mixture_map": None,
        "lights":None,
        "chart_map":None,
        "illumination_map":None,
        "clipping": False,
        "too_dark": False,
    }
    mixture_map = get_mixture_map(scene_data)
    scene_data["mixture_map"] = mixture_map
    #get illumination information
    num_illums=scene_data["info"]["NumOfLights"]
    lights=[scene_data["info"][f"Light{num_illums}"] for num_illums in range(1,num_illums+1)]
    lights=np.array(lights)
    scene_data["lights"]=lights
    scene_data["illumination_map"] = get_illumination_map(mixture_map, lights)
    
    
    for light_index in range(1, n_lights + 1):
        image_data = {}
        stats['total_images'] += 1

        # Load images
        raw_tiff, full_raw_tiff, clip_mask = _load_image_pairs(place, light_index)
        # Process all charts for this image
        image_chart_values = []
        full_image_chart_values = []
        image_mixture_map_values = []
        image_illumination_maps=[]
        current_image_clipped_charts = 0
        chart_threshold_masks = []
      
        for chart in all_patch_coordinates:

            chart_values, full_chart_values, clipping_found, chart_mixture_map,chart_threshold_mask, threshold_mask= _process_chart(
                chart, raw_tiff, full_raw_tiff, clip_mask,mixture_map
            )

            if clipping_found:
                stats['total_charts_clipped'] += 1
                current_image_clipped_charts += 1
                scene_data["clipping"] = True
            chart_threshold_masks.append(chart_threshold_mask)
            image_chart_values.append(chart_values)
            full_image_chart_values.append(full_chart_values)
            image_mixture_map_values.append(chart_mixture_map)
            image_illumination_maps.append(get_illumination_map(chart_mixture_map, lights))
            

        # update total_charts by number of charts processed for this image
        stats['total_charts'] += len(all_patch_coordinates)

        # if this image had any clipped charts, count it as a clipped image
        if current_image_clipped_charts > 0:
            stats['total_clipped_images'] += 1

        image_chart_values = np.array(image_chart_values)
        full_image_chart_values = np.array(full_image_chart_values)
        image_mixture_map_values = np.array(image_mixture_map_values)
        image_illumination_maps = np.array(image_illumination_maps)

        scene_data["too_dark"] = brightness_threshold(image_chart_values)

        image_data["patches"] = image_chart_values
        image_data["real_patches"] = full_image_chart_values
        # image_data["real_whitepoints"] = get_whitepoints(full_image_chart_values)
        image_data["light_index"] = light_index
        image_data["whitepoints"] = get_whitepoints(scene_data,light_index)
        image_data["forward_matrix_1"], image_data["forward_matrix_2"] = get_forward_matrices(raw_path)
        image_data["mixture_map"]= image_mixture_map_values
        image_data["chart_illumination_map"]= image_illumination_maps
        image_data["chart_threshold_mask"]= np.array(chart_threshold_masks)
        image_data["threshold_mask"]= threshold_mask
        
        
        scene_data["image_data"].append(image_data)
    scene_data["chart_map"]=get_chart_map_rgb(scene_data)
    if not (scene_data["clipping"] or scene_data["too_dark"]):
        stats['used_in_dataset'] += 1
        _save_scene_data(scene_data)
    else:
        print(f"Skipping scene {place} due to clipping or darkness.")

    return (stats['total_charts_clipped'], stats['total_charts'],
            stats['total_images'], stats['total_clipped_images'], stats["used_in_dataset"])
def brightness_threshold(data):
    min_threshold=0.0005#threshold assumes white level of 1
    #if anything in data npy is below threshold, return True
    if np.any(data < min_threshold):
        return True
    return False
    
def process_scene(place, placeInfo):
    clipped_charts, total_charts, total_images, total_clipped_images,used = _handle_scene_data(place, placeInfo)
    scene_clipped = 1 if total_clipped_images > 0 else 0
    return {
        "scene_clipped": scene_clipped,
        "total_clipped_images": total_clipped_images,
        "total_images": total_images,
        "clipped_charts": clipped_charts,
        "total_charts": total_charts,
        "used_in_dataset":used
    }








if __name__ == "__main__":
    
    # open json annotation file
    meta_path=os.path.join(src_path, CAMERA, "meta.json")
    with open(meta_path, 'r') as json_file:
        jsonData = json.load(json_file)

    # get directory list
    places = sorted([f for f in os.listdir(os.path.join(src_path, CAMERA)) if os.path.isdir(os.path.join(src_path, CAMERA, f))])
    if _args.limit:
        places = places[:_args.limit]
        print(f"--limit {_args.limit}: processing {len(places)} of the available scenes")
    print(f"processing and saving images too:{dest_path}")
   
 

    results = []
    



    from concurrent.futures import ProcessPoolExecutor
    # num_threads = max(1, multiprocessing.cpu_count() // 2)

    num_workers = 8# max(1, multiprocessing.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_scene, place, jsonData[place]): place for place in places}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(places)):
            result = future.result()
            results.append(result)


    #copy metadata file to dest
    if CONFIG["save"]:
        #make directory if not exists
        os.makedirs(os.path.join(dest_path,CAMERA), exist_ok=True)
        shutil.copy2(meta_path, os.path.join(dest_path,CAMERA, "meta.json"))

    # Combine numerical data
    total_scenes = len(results)
    total_scene_clipped = sum(r["scene_clipped"] for r in results)
    total_images = sum(r["total_images"] for r in results)
    total_clipped_images = sum(r["total_clipped_images"] for r in results)
    total_charts = sum(r["total_charts"] for r in results)
    total_clipped_charts = sum(r["clipped_charts"] for r in results)
    total_scenes_used = sum(r["used_in_dataset"] > 0 for r in results)
    

    print("Summary:")
    print(f"Total scenes processed: {total_scenes}")
    print(f"Scenes with clipped images: {total_scene_clipped}")
    print(f"Total images: {total_images}")
    print(f"Total clipped images: {total_clipped_images}")
    print(f"Total charts: {total_charts}")
    print(f"Total clipped charts: {total_clipped_charts}")
    print(f"Total scenes used: {total_scenes_used}")


         

