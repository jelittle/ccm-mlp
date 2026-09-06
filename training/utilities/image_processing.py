from rawpy import imread
import numpy as np
import os
import torch
from tqdm import tqdm
from .bestfitsolver import BestFitSolver


from camera_pipeline.pipeline_lib.pipeline_utils import fix_orientation, get_metadata
def check_bit_clip(image, image_path):
    """Check if the image clips at the bit depth"""

    metadata=get_metadata(image_path)  # Ensure metadata is extracted
    white_level = metadata["white_level"]
    max_value=white_level[0]


    if np.any(image == max_value):
        raise ValueError(f"Image clips")
    if np.any(image > max_value):
        raise ValueError(f"Image exceeds maximum value for {max_value}-bit depth: {max_value}, white level too low?")




def get_color_matrices(image_path):
    metadata = get_metadata(image_path)
    cm1= np.array(metadata["color_matrix_1"], dtype=np.float32).reshape(3, 3)
    cm2= np.array(metadata["color_matrix_2"], dtype=np.float32).reshape(3, 3)
    return cm1,cm2
def get_calibrated_csts(temp_patch_paths, data_config, target):
    """desired temp are hardcoded cause lazy"""
    low_temp_cst = None
    high_temp_cst = None
    mid_temp_cst = None
    target = torch.tensor(target[np.newaxis, ...], dtype=torch.float32)  
    solver = BestFitSolver()
    from pathlib import Path

    '''
    
        idx = idx // self.scale

        patch_path = Path(self.patch_paths[idx])
        patch_data = np.load(patch_path, allow_pickle=True).item()
        image= patch_data["patches"]
        image= white_balance_patch(image)  # Apply white balance patch transformation
        image = np.moveaxis(image, -1, 0)  # Reshape to C × H × W
        white_point = whitepoint(np.array(patch_data["whitepoint"]))
        return {'white_point':white_point, 'input':image, 'target':self.target, "path":str(patch_path), "illum_idx":idx}
'''


    temps = [1000 + i*250 for i in range(len(temp_patch_paths))]  # Example temperatures from 1000K to 10000K

    def get_cst_for_temp(temp):
        #get in
        patch_info = np.load(temp_patch_paths[temps.index(temp)], allow_pickle=True).item()
        image = patch_info['patches']
        white_patch = image[-1, 0, :]
        wb = white_balance(image, white_patch)
        wb = torch.tensor(wb[np.newaxis, ...], dtype=torch.float32)
        wb = wb.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        output = solver(wb, target)
        output = output.reshape((1, 3, 3))
        return output
    
    low_temp_cst = get_cst_for_temp(2750), 2750
    high_temp_cst = get_cst_for_temp(6500), 6500
    mid_temp_cst = get_cst_for_temp(5000), 5000 #5000 following Hakki
    return low_temp_cst, high_temp_cst, mid_temp_cst


def get_single_led_csts(single_led_patch_paths, target, max_xy=None):
    """Build per-illuminant anchors for the 7 single-LED captures (light000-light006).

    Returns a list of length 7 with entries:
        {"cst": Tensor[3,3], "xy": np.ndarray[2], "cct": float}

    xy is taken from the patch's spectrally-measured chromaticity (xy_from_spectra),
    and cct is derived from that xy via colour.temperature.xy_to_CCT.
    """
    import colour
    target = torch.tensor(target[np.newaxis, ...], dtype=torch.float32)
    solver = BestFitSolver()
    anchors = []
    for p in single_led_patch_paths:
        info = np.load(p, allow_pickle=True).item()
        image = info['patches']
        white_patch = image[-1, 0, :]
        wb = white_balance(image, white_patch)
        wb = torch.tensor(wb[np.newaxis, ...], dtype=torch.float32).permute(0, 3, 1, 2)
        cst = solver(wb, target).reshape((3, 3))

        xy = np.asarray(info['xy_from_spectra'], dtype=np.float64).reshape(2)
        cct = float(colour.temperature.xy_to_CCT(xy))
        xy_norm = np.clip((xy / max_xy) * 2 - 1, -1, 1) if max_xy is not None else xy
        anchors.append({"cst": cst.detach().cpu(), "xy": xy_norm, "cct": cct})

    anchors.sort(key=lambda a: a["cct"])
    return anchors

# def naive_demosaicing(bayer, image_path):

#     metadata = get_metadata(image_path)
#     cfa_pattern = metadata["cfa_pattern"]
#     #list of 4 numbers 0 (R), 1 (G), 2 (B) each at the 4 positions of the 2x2 bayer pattern
#     # e.g. [0,1,1,2] means 
#     # R G
#     # G B
#     print(f"CFA pattern: {cfa_pattern}")
    
#     H, W= bayer.shape
#     # Create an empty image with 3 channels
#     packed = np.zeros((H // 2, W // 2, 4), dtype=bayer.dtype)
#     rgb= np.zeros((H // 2, W // 2, 3), dtype=bayer.dtype)
#     # bayer pattern packing
#     packed[:, :, 0] = bayer[0::2, 0::2]  # R
#     packed[:, :, 1] = bayer[0::2, 1::2]  # G1
#     packed[:, :, 2] = bayer[1::2, 0::2]  # G2
#     packed[:, :, 3] = bayer[1::2, 1::2]  # B

#     # avg green pixels
#     rgb[:, :, 1] = (packed[:, :, 1] + packed[:, :, 2]) / 2.0  # G

#     #assign blue and red
#     rgb[:, :, 0] = packed[:, :, 0]  # R
#     rgb[:, :, 2] = packed[:, :, 3]  # B

#     return rgb.astype(np.float32)

def naive_demosaicing(bayer, image_path):
    """
    Demosaic a raw Bayer frame to half-res RGB by honoring the 2x2 CFA pattern
    provided in metadata["cfa_pattern"] as a list of four ints:
        0=R, 1=G, 2=B
    The list is ordered as:
        [top-left, top-right, bottom-left, bottom-right]
    """
    metadata = get_metadata(image_path)
    cfa_pattern = list(metadata["cfa_pattern"])  # e.g., [0,1,1,2]
    # Pattern indexing:
    # indices: 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)

    if len(cfa_pattern) != 4 or any(c not in (0, 1, 2) for c in cfa_pattern):
        raise ValueError("cfa_pattern must be a list of four values drawn from {0,1,2}.")

    H, W = bayer.shape
    h2, w2 = H // 2, W // 2

    # Create an empty image with 4 channels to pack the 2x2 block in pattern order
    packed = np.zeros((h2, w2, 4), dtype=bayer.dtype)
    # Extract the four sub-sampled planes from Bayer
    subs = [
        bayer[0::2, 0::2],  # top-left
        bayer[0::2, 1::2],  # top-right
        bayer[1::2, 0::2],  # bottom-left
        bayer[1::2, 1::2],  # bottom-right
    ]
    # Pack them in the same order as the CFA pattern describes the 2x2 tile
    for i in range(4):
        packed[:, :, i] = subs[i]

    # Build RGB by mapping packed planes via the pattern
    rgb = np.zeros((h2, w2, 3), dtype=np.float32)

    # Accumulate greens in case there are two G sites (usual Bayer)
    g_sum = np.zeros((h2, w2), dtype=np.float32)
    g_count = np.zeros((h2, w2), dtype=np.float32)

    for i, color in enumerate(cfa_pattern):
        plane = packed[:, :, i].astype(np.float32)
        if color == 0:       # R
            rgb[:, :, 0] = plane
        elif color == 2:     # B
            rgb[:, :, 2] = plane
        elif color == 1:     # G
            g_sum += plane
            g_count += 1.0

    # Average green(s); handle unusual cases gracefully
    has_g = g_count > 0
    rgb[:, :, 1][has_g] = g_sum[has_g] / g_count[has_g]

    return rgb

def get_average(img, x: int, y: int,patch_radius):
    patch = img[y - patch_radius:y + patch_radius + 1,
                x - patch_radius:x + patch_radius + 1, :]

    avg_rgb = np.mean(patch, axis=(0, 1))
    return avg_rgb

def get_patch(img: np.array, patch_shape, corner1, corner2, patch_radius=5):
    """gets patch values based on patch_shape for this specific image"""
    patch_viz = img.copy()
    patch_values = []
    
    # The corners are in ratios, get the image size and the real corner coords
    h, w = img.shape[:2]
    corner1_x, corner1_y = int(corner1[0] * w), int(corner1[1] * h)
    corner2_x, corner2_y = int(corner2[0] * w), int(corner2[1] * h)
    
    # Get points that fit the patch_shape
    rows, cols = patch_shape[0], patch_shape[1]  # Use actual patch_shape dimensions
    patch_width = (corner2_x - corner1_x) // cols
    patch_height = (corner2_y - corner1_y) // rows
    
    coords = []
    for i in range(rows):  # Use rows from patch_shape
        for j in range(cols):  # Use cols from patch_shape
            center_x = corner1_x + j * patch_width + patch_width // 2
            center_y = corner1_y + i * patch_height + patch_height // 2
            coords.append((center_y, center_x))
    
    for center_y, center_x in coords:
        # Ensure the coordinates are within the image bounds
        patch_value = get_average(img, int(center_x), int(center_y), patch_radius=patch_radius)
        patch_values.append(patch_value)
        # For visualization, draw rectangles on patch_viz
        import cv2
        cv2.rectangle(patch_viz,
                      (center_x - patch_width // 2, center_y - patch_height // 2),
                      (center_x + patch_width // 2, center_y + patch_height // 2),
                      (0, 255, 0), 2)
    
    patch_values = np.array(patch_values, dtype=np.float32)
    # Reshape to the desired patch shape
    patch_values = patch_values.reshape(patch_shape)
    return patch_values, patch_viz


def get_image_paths_from_dir(dir):
    """Get all images from a directory"""
    image_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.dng')):
                image_paths.append(os.path.abspath(os.path.join(root, file)))
    return sorted(image_paths)

def get_raw(image_path):
    raw_image= imread(image_path).raw_image_visible
    return raw_image
def fix_orientation2(image, image_path):
    metadata = get_metadata(image_path)
    image = fix_orientation(image, metadata["orientation"])
    return image

def get_forward_matrices(image_path):
    metadata = get_metadata(image_path)
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

def normalize_image(image_path, img, use_black_level=True):
    """Normalizes the image using black and white levels from metadata"""
    metadata = get_metadata(image_path)
    black_level_arr = np.array(metadata["black_level"], dtype=np.int32)
    white_level_arr = np.array(metadata["white_level"], dtype=np.int32)

    print(f"Black level: {black_level_arr}, White level: {white_level_arr}")

    
    # Ensure black level consistency
    if np.all(black_level_arr == black_level_arr[0]):
        black_level = black_level_arr[0]
    else:
        raise ValueError("Black level array must have the same value for all channels.")
    
    # Ensure white level consistency
    if np.all(white_level_arr == white_level_arr[0]):
        white_level = white_level_arr[0]
    else:
        raise ValueError("White level array must have the same value for all channels.")
    
    # Subtract black level
    if use_black_level:
        img = img - black_level
    img = np.clip(img, 0, None)
    
    # Normalize to [0, 1]
    denom = white_level - black_level
    if denom > 0:
        img = img / denom
    else:
        img = np.zeros_like(img, dtype=np.float32)
    
    return img.astype(np.float32)

def get_camera_whitepoint(image_path):
    metadata = get_metadata(image_path)
    white_point=metadata["as_shot_neutral"]

    return white_point

def white_balance(image, whitepoint):
    """White balances a patch using the white point"""
    # Ensure the patch is in the correct shape
    # Apply white balance
    return image / whitepoint