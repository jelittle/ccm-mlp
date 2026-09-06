
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
import os
from pathlib import Path
import yaml
import sys
from utilities import image_processing as img_proc
import numpy as np
import colour
import pandas as pd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import cv2
from utilities.paths import resolve, resolve_config


def get_xy_from_lss():
    path=resolve("${repo_root}/assets/lss/combined.lss")
    df = pd.read_csv(path, sep='\t', header=None)
    # Set the first row as column names
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)
    # Convert all columns to float
    df = df.astype(float)
    # Create a list of wavelengths from the column names (excluding the first column if it's not a wavelength)
    wavelengths = [float(col) for col in df.columns[1:]]
    # Convert each row into a SpectralDistribution and compute xy chromaticity
    xy_list = []
    
    for idx, row in df.iterrows():
        values = row.values[1:]  # skip the first column if it's not a wavelength
        sd = colour.SpectralDistribution(dict(zip(wavelengths, values)))
        xy = colour.sd_to_XYZ(sd)
        xy = colour.XYZ_to_xy(xy)
        xy_list.append(xy)
    
    return xy_list

def process_image(image_path, xy, target_dir, chart_type, corner1, corner2):

    """Process a single image and return all needed data"""
    # try:
    # Get name from image path
    if chart_type == 24:
        patch_shape = (4, 6, 3)

    name = os.path.splitext(os.path.basename(image_path))[0]

    # Load raw image
    raw_image = img_proc.get_raw(image_path)
    #if image_path contains "Pixel", rotate the image 180 degrees

    #make sure none of the images clip
    img_proc.check_bit_clip(raw_image,image_path)
    # Save raw image
    raw_file_path = Path(target_dir) / "raw" / f"{name}.npy"
    raw_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    np.save(raw_file_path, raw_image)
    
    # Process image for patch extraction
    demosaiced_img = img_proc.naive_demosaicing(raw_image, image_path)
    demosaiced_img = img_proc.fix_orientation2(demosaiced_img, image_path) #important to have this order o the CFA pattern is interpretted wrong

    normalized_image = img_proc.normalize_image(image_path, demosaiced_img)
    normalized_file_path = Path(target_dir) / "rawnormalized" / f"{name}.npy"
    normalized_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    np.save(normalized_file_path, normalized_image)
    # Get patch
    patches, patch_viz = img_proc.get_patch(normalized_image, patch_shape, corner1, corner2)
    whitepoint = patches[-1, 0, :]
    xy = np.array(xy, dtype=np.float32)

    forward_matrix_1, forward_matrix_2 = img_proc.get_forward_matrices(image_path)
    condensed_out = {
        "patches": patches,
        "whitepoint": whitepoint,
        "xy_from_spectra": xy,
        "forward_matrix_1": forward_matrix_1,
        "forward_matrix_2": forward_matrix_2,
    }
    #Save condensed output to .npy file
    patch_file_path = Path(target_dir) / "patches" / f"{name}.npy"
    patch_file_path.parent.mkdir(parents=True, exist_ok=True)  # 
    np.save(patch_file_path, condensed_out, allow_pickle=True)

    #write a 4x6 image of the patches for visualization
    #upsample with nearest neighbors
    patches_upsampled = cv2.resize(patches, (patch_shape[1]*100, patch_shape[0]*100), interpolation=cv2.INTER_NEAREST)
    patch_image_path = Path(target_dir) / "patch_viz" / f"{name}.png"
    patch_image_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    cv2.imwrite(str(patch_image_path), patches_upsampled[:,:,::-1])
    print(f"Processed and saved data for image: {image_path}")

    patches_viz_path = Path(target_dir) / "patch_viz_on_image" / f"{name}.png"
    patches_viz_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    cv2.imwrite(str(patches_viz_path), patch_viz[:,:,::-1])

CAMERAS = ["sony", "canon", "pixel", "samsung"]

_parser = argparse.ArgumentParser(
    description="Extract ColorChecker patches from the raw lightbox captures.")
_parser.add_argument("cameras", nargs="*", default=CAMERAS, choices=CAMERAS + [[]],
                     help="cameras to process (default: all four)")
_args, _ = _parser.parse_known_args()

config_files = [resolve("${repo_root}/training/configs/cameras/%s.yaml" % c)
                for c in (_args.cameras or CAMERAS)]

for process_file in config_files:
    print("=== %s ===" % os.path.basename(process_file))
    with open(process_file, 'r') as file:
        config = resolve_config(yaml.safe_load(file))

    data_config = config['data']

    # Get all image paths
    src_dir = data_config['src_path']
    dng_folder = os.path.join(src_dir, "dng_renamed")
    image_paths = sorted(img_proc.get_image_paths_from_dir(dng_folder))

    num_processes = min(mp.cpu_count()/2, len(image_paths))
    xy = get_xy_from_lss()  #let this be global


    # Keep your static args bundled
    base_process_func = partial(
        process_image,
        target_dir=data_config['processed_path'],
        chart_type=data_config["chart_type"],
        corner1=data_config["corner1"],
        corner2=data_config["corner2"],
    )

    # Pair each image_path with its corresponding xy
    inputs = [(path, xy_val) for path, xy_val in zip(image_paths, xy)]

    print(f"Processing {len(image_paths)} images using {num_processes} processes...")
    with mp.Pool(processes=int(num_processes)) as pool:
        results = []
        for result in tqdm(pool.starmap(base_process_func, inputs), total=len(inputs)):
            results.append(result)
