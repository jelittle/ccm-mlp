
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

from pathlib import Path
import rawpy
import cv2
import numpy as np
import sys
import os
from camera_pipeline.pipeline_lib.pipeline_utils import get_metadata
from utilities import image_processing as img_proc
import multiprocessing
from utilities.paths import resolve
"""
todo list: 
- adjust aruco detection params for proper chart setup
- handle aruco detection failure (currently throws error on not 4 ids)
- quick script to rename images from camera output (once we have full datasets)
- better brightness handling for aruco detection 
"""

# Layout of the distributed dataset:
#   InTheWild/<camera>/raw/*.dng      the captures
#   InTheWild/<camera>/corners/*.npy  chart corners, one per capture,
#                                     same basename as the .dng
import argparse

CAMERAS = ["pixel", "samsung"]
_parser = argparse.ArgumentParser(
    description="Extract ColorChecker patches from the in-the-wild captures.")
_parser.add_argument("camera", nargs="?", default="pixel", choices=CAMERAS)
_parser.add_argument("--src", type=Path, default=None,
                     help="root holding <camera>/raw and <camera>/corners")
_parser.add_argument("--dest", type=Path, default=None,
                     help="where to write wildpatches/wildrawnormalized/images_png")
_parser.add_argument("--limit", type=int, default=None,
                     help="process only the first N captures")
_args, _ = _parser.parse_known_args()

SRC_PATH = _args.src if _args.src else Path(resolve("${raw_data_root}/InTheWild"))
dst_PATH = _args.dest if _args.dest else Path(resolve("${data_root}/in_the_wild"))
CAMERA = _args.camera




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
  


def get_patch_average(image, center_x, center_y, radius=5):
    """Extract and average a square patch around a center point"""
    # Ensure integer coordinates
  
    x, y = int(np.round(center_x)), int(np.round(center_y))
    patch=image[ y-radius:y+radius,x-radius:x+radius,:]
    
    return np.mean(patch, axis=(0, 1))
def get_whitepoint(patch):
    # Assuming patch is a numpy array of shape (6, 4, 3)
    # Calculate the mean color of the top row (first row)
    assert patch.shape == (4, 6, 3), "Expected patch shape to be (4, 6, 3)"
    whitepoint = patch[-1, 0, :]
    return whitepoint

def _process_patch(image,corners):

    # Save the normalized image for debugging


    #build the patch
    # Compute corners based on smallest/largest x, y coordinates
    
    # Ensure corners is in the right format (N, 2) where N >= 4
    if corners.ndim == 3:
        # If corners is (N, 1, 2), reshape to (N, 2)
        corners = corners.reshape(-1, 2)
    elif corners.ndim == 1:
        # If corners is flat, reshape assuming it's [x1, y1, x2, y2, ...]
        corners = corners.reshape(-1, 2)
    downsample_rate=4
    # Downsample the image and adjust corner coordinates to the smaller image
    h, w = image.shape[:2]
    new_w = max(1, int(round(w / downsample_rate)))
    new_h = max(1, int(round(h / downsample_rate)))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    corners = corners.astype(np.float32) / float(downsample_rate)
    # Ensure corners remain inside the downsampled image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, new_w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, new_h - 1)
    
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]

    tl_corner = corners[np.argmin(x_coords + y_coords)]  # Top-left: smallest x + y
    br_corner = corners[np.argmax(x_coords + y_coords)]  # Bottom-right: largest x + y
    tr_corner = corners[np.argmax(x_coords - y_coords)]  # Top-right: largest x - y
    bl_corner = corners[np.argmin(x_coords - y_coords)]  # Bottom-left: smallest x - y
    
    corner_points = np.array([tl_corner, tr_corner, br_corner, bl_corner], dtype=np.float32)
    
    center_coords = get_patch_coordinates(corner_points)
    patch_averages = []
    for (cx,cy ) in center_coords:
        avg = get_patch_average(image, cx, cy, radius=5)
        patch_averages.append(avg)
   
    patch_averages = np.array(patch_averages).reshape(4, 6, 3)  # Shape: (4, 6, 3)
    assert patch_averages.shape == (4, 6, 3), "Expected 6 columns, 4 rows, with 3 color channels each"
    

    #get patch data
    whitepoint=get_whitepoint(patch_averages)
    # norm_whitepoint = whitepoint / whitepoint[1]
    norm_whitepoint= whitepoint
    print(patch_averages[-1,-1])
    
    return patch_averages,norm_whitepoint


def _process_image(image_path):
    clipping=False
    raw_image = img_proc.get_raw(image_path)
    metadata=get_metadata(image_path)
    saturation_level=metadata["white_level"][0]

    #check if saturated:
    if np.max(raw_image) >= saturation_level:
        print(f"Warning: Image {image_path} has saturated pixels.")

    #simple demosaicing on the image
    demosaiced_img = img_proc.naive_demosaicing(raw_image, image_path)
    # demosaiced_img = cv2.cvtColor(demosaiced_img, cv2.COLOR_BGR2RGB)
    demosaiced_img = img_proc.fix_orientation2(demosaiced_img, image_path) #important to have this order o the CFA pattern is interpretted wrong
    normalized_image = img_proc.normalize_image(image_path, demosaiced_img)
    #get important data
    
    callibration_illuminant1=metadata['calibration_illuminant_1'][0]
    callibration_illuminant2=metadata['calibration_illuminant_2'][0]
    known_illuminants={17:2856,21:6500}
    temp_1=known_illuminants[callibration_illuminant1] #we want these to throw errors if not found
    temp_2=known_illuminants[callibration_illuminant2]
    assert temp_1 != temp_2, "Both calibration illuminants are the same"

    # Convert fraction strings to float32 arrays
    fm1_fractions = metadata["forward_matrix_1"]
    fm2_fractions = metadata["forward_matrix_2"]
    
    fm1 = (np.array([float(frac) for frac in fm1_fractions], dtype=np.float32), temp_1)
    fm2 =(np.array([float(frac) for frac in fm2_fractions], dtype=np.float32),temp_2)
    if temp_1 > temp_2:
        fm1, fm2 = fm2, fm1

    
    # Display the demosaiced image for debugging


    return normalized_image,fm1,fm2, clipping
    
def save_image(image_data):
    base_path=os.path.join(dst_PATH,CAMERA)
    base_name=os.path.basename(image_data["path"])
    patch_folder=os.path.join(base_path,"wildpatches")
    img_folder=os.path.join(base_path,"wildrawnormalized")
    png_folder=os.path.join(base_path,"images_png")
    dng_folder=os.path.join(base_path,"raw")
    dng_path=os.path.join(dng_folder,base_name)
    img_path=os.path.join(img_folder,base_name.replace(".dng",".npy"))
    patch_dest=os.path.join(patch_folder,base_name.replace(".dng",".npy"))
    os.makedirs(patch_folder,exist_ok=True)
    os.makedirs(img_folder,exist_ok=True)




    

    data = {}
    data["patches"] = image_data["chart"]
    data["whitepoint"] = image_data["whitepoint"]
    data["xy_from_spectra"] = None
    data["forward_matrix_1"] = image_data["forward_matrix_1"]
    data["forward_matrix_2"] = image_data["forward_matrix_2"]
    # data["threshold_mask"] = image_data["threshold_mask"]
    # data["chart_threshold_mask"] = image_data["chart_threshold_mask"][max_mean_index]
    # data["chart_map"] = single_chart_map
   
    np.save(patch_dest, data)
    np.save(img_path, image_data["image"])
    
    # Save full normalized image as a PNG for quick viewing
    os.makedirs(png_folder, exist_ok=True)
    png_path = os.path.join(png_folder, base_name.replace(".dng", ".png"))

    img = image_data["image"]
    print(img.shape)
    print("img max:", np.max(img), "img min:", np.min(img))
    # Normalize the image for visualization
    img_vis = img / np.max(img)
    img_vis = (img_vis * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(png_path, img_vis)

    #save the wb image
    wb_png_path = os.path.join(png_folder, base_name.replace(".dng", "_wb.png"))
    whitepoint=image_data["whitepoint"]
    wb_image=image_data["image"] / whitepoint[np.newaxis, np.newaxis, :]
    # Normalize the white-balanced image for visualization
    wb_image_normalized = wb_image / np.max(wb_image)
    wb_image_vis = (wb_image_normalized * 255).astype(np.uint8)
    wb_image_vis = cv2.cvtColor(wb_image_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(wb_png_path, wb_image_vis*20)


    # cv2.imwrite(png_path, img_vis)
    
    # Save a PNG of the patches for visualization
    patch_png_folder = os.path.join(base_path, "patches_png")
    os.makedirs(patch_png_folder, exist_ok=True)
    patch_png_path = os.path.join(patch_png_folder, base_name.replace(".dng", ".png"))
    patch_wb_png_path=os.path.join(patch_png_folder, base_name.replace(".dng", "_wb.png"))
    #white balance the image and save that too
    whitepoint=image_data["whitepoint"]
    wb_patch_image=image_data["chart"] / whitepoint[np.newaxis, np.newaxis, :]
    # Resize and convert the white-balanced patch image for saving
    
    # Normalize the white-balanced patch image for easier visualization
    wb_patch_image_normalized = wb_patch_image #/ np.max(wb_patch_image)
    wb_patch_image_resized = cv2.resize(wb_patch_image_normalized, (6 * 60, 4 * 60), interpolation=cv2.INTER_NEAREST)
    wb_patch_image_resized = cv2.cvtColor(wb_patch_image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(patch_wb_png_path, (wb_patch_image_resized * 255).astype(np.uint8))

    # Normalize the patch image for easier visualization
    patch_image_normalized = image_data["chart"] / np.max(image_data["chart"])
    patch_image = (patch_image_normalized * 255).astype(np.uint8)  # Scale to 0-255 for visualization
    patch_image = cv2.resize(patch_image, (6 * 60, 4 * 60), interpolation=cv2.INTER_NEAREST)  # Resize for better visibility
    patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(patch_png_path, patch_image)

    # Copy the original DNG file to the destination folder
    os.makedirs(dng_folder, exist_ok=True)
  
    if not os.path.exists(dng_path):
        os.system(f"cp {image_data['path']} {dng_path}")


def process_folder(folder_path):
    invalid_aruco=False 
    image_names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.dng')])
    image_data={}
    image_path= os.path.join(folder_path, [f for f in image_names if not f.endswith("_sub.dng")][0])
    image_data["path"]=image_path
    normalized_image, image_data["forward_matrix_1"], image_data["forward_matrix_2"], clipping=_process_image(image_path)


    final_image=normalized_image
    image_data["image"]=final_image

    #patch stuff
    corners= np.load(os.path.join(folder_path,"corners.npy"),allow_pickle=True)
    corners = np.round(corners).astype(int)
    chart,whitepoint=_process_patch(final_image,corners)
    #check if any thing on chart is clipped
    if np.any(chart >= 0.99):
        print("Highlight clipping detected in chart patches.")
        clipping=True
    # if np.any(chart < 0.001):
    #     print()
    #     print("Shadows clipping detected in chart patches.")
    #     clipping=True
    image_data["chart"]=chart
    image_data["whitepoint"]=whitepoint
    # Save a PNG of the chart for visualization
    if clipping:
        print("Clipping detected, not saving image data.")
    else:
        save_image(image_data)
    return invalid_aruco,clipping

def process_image(dir_path, image_path):
    corner_dir=os.path.join(dir_path,"corners")
    invalid_aruco=False 
    image_data={}
    image_data["path"]=image_path
    normalized_image, image_data["forward_matrix_1"], image_data["forward_matrix_2"], clipping=_process_image(image_path)


    final_image=normalized_image
    image_data["image"]=final_image

    #patch stuff
    corners= np.load(os.path.join(corner_dir,os.path.basename(image_path).replace(".dng",".npy")),allow_pickle=True)
    corners = np.round(corners).astype(int)
    chart,whitepoint=_process_patch(final_image,corners)
    #check if any thing on chart is clipped
    if np.any(chart >= 0.99):
        print("Highlight clipping detected in chart patches.")
        clipping=True
    # if np.any(chart < 0.001):
    #     print()
    #     print("Shadows clipping detected in chart patches.")
    #     clipping=True
    image_data["chart"]=chart
    image_data["whitepoint"]=whitepoint
    # Save a PNG of the chart for visualization
    if clipping:
        print("Clipping detected, not saving image data.")
    else:
        save_image(image_data)
    return invalid_aruco,clipping    
 


if __name__ == "__main__":
    import concurrent.futures
    from tqdm import tqdm
    main_dir = os.path.join(SRC_PATH, CAMERA)
    image_dir=os.path.join(main_dir,"raw")
    images= sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    if _args.limit:
        images = images[:_args.limit]
        print(f"--limit {_args.limit}: processing {len(images)} of the available captures")
    results = []


    for image_name in tqdm(images):
        image_path=os.path.join(image_dir,image_name)
        print(f"Processing image: {image_path}")
        invalid_aruco,clipping=process_image(main_dir, image_path) 
        if invalid_aruco:
            print(f"  - ArUco detection failed for image: {image_path}")
        if clipping:
            print(f"  - Clipping detected in image: {image_path}")
    # image_folders=image_folders[:1] 
    # for folder in image_folders:
    #     folder_path=os.path.join(SRC_PATH,CAMERA,folder)
    #     print(f"Processing folder: {folder_path}")
    #     invalid_aruco,clipping=process_image(folder_path) 
    #     if invalid_aruco:
    #         print(f"  - ArUco detection failed for folder: {folder_path}")
    #     if clipping:
    #         print(f"  - Clipping detected in folder: {folder_path}")
    # num_threads = max(1, multiprocessing.cpu_count() // 2)
    # num_threads = 1
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = {executor.submit(process_data, image_name): image_name for image_name in image_names}
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(image_names)):
    #         try:
    #             results.append(future.result())
    #         except Exception as e:
    #             print(f"Error processing {futures[future]}: {e}")


    

