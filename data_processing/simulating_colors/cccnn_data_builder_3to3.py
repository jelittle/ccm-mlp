
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

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.ndimage import gaussian_filter1d
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
from colour import SpectralDistribution
from utilities.paths import resolve
"""TODO: current data is only measured to 735, but we need 780nm to perfectly match cccnn



"""
def generate_reflectance_spectrum(wavelengths: np.ndarray,
                                  n_basis: int = 10) -> np.ndarray:
    """
    Generate one synthetic reflectance spectrum over `wavelengths` by
    summing random basis functions (gaussian, logistic, ramp, sine,
    random walk), smoothing, and clipping to [0,1].
    """
    λ = wavelengths
    spec = np.zeros_like(λ, dtype=float)

    for i in range(n_basis):
        choice = np.random.choice(['gaussian','logistic','ramp','sine','randwalk'])
        if choice == 'gaussian':
            mu = np.random.uniform(λ.min(), λ.max())
            sigma = np.random.uniform(10, 80)
            spec += np.exp(-0.5*((λ - mu)/sigma)**2)
        elif choice == 'logistic':
            k = np.random.uniform(0.01, 0.1)
            x0 = np.random.uniform(λ.min(), λ.max())
            spec += 1/(1 + np.exp(-k*(λ - x0)))
        elif choice == 'ramp':
            cut = np.random.randint(1, len(λ)-1)
            ramp_up = np.linspace(0, 1, cut)
            ramp_down = np.linspace(1, 0, len(λ) - cut)
            spec[:cut] += ramp_up
            spec[cut:] += ramp_down
        elif choice == 'sine':
            freq = np.random.uniform(0.005, 0.02)
            phase = np.random.uniform(0, 2*np.pi)
            amp = np.random.uniform(0.2, 1.0)
            spec += amp * np.sin(2*np.pi*freq*λ + phase)

    # smooth to limit max slope and ensure continuity
    spec = gaussian_filter1d(spec, sigma=5)
    # normalize and clip
    spec = spec - spec.min()
    if spec.max()>0:
        spec /= spec.max()
    return np.clip(spec, 0, 1)



def load_camspecs_specs(file_paths: List[str], suffix: str = "SR"):
    data_list = []
    wave_lengths = None

    assert suffix in ["SR", "QE", "RAW"], suffix

    for file_path in file_paths:
        data = pd.read_csv(file_path, sep='\s+', skiprows=15, encoding="SHIFT-JIS")
        data[['Lambda', f'R_{suffix}', f'G_{suffix}', f'B_{suffix}']] = \
            data[['Lambda', f'R_{suffix}', f'G_{suffix}', f'B_{suffix}']].apply(pd.to_numeric, errors='coerce')

        if wave_lengths is None:
            wave_lengths = data['Lambda'].values
        else:
            assert (data["Lambda"].values == wave_lengths).all

        data_np = data[[f'R_{suffix}', f'G_{suffix}', f'B_{suffix}']].values.T
        data_list.append(data_np)

    data_np = np.array(data_list)
    assert data_np.shape == (len(file_paths), 3, len(wave_lengths)), data_np.shape
    assert wave_lengths.shape == (len(wave_lengths),), wave_lengths.shape

    # Always interpolate to 1nm resolution
    # Determine the wavelength range from the data
    min_wavelength = int(np.floor(wave_lengths.min()))
    max_wavelength = int(np.ceil(wave_lengths.max()))
    target_shape = SpectralShape(min_wavelength, max_wavelength, 1)
    
    print(f"Interpolating camera sensitivity from {len(wave_lengths)} points to {len(target_shape.wavelengths)} points")
    print(f"Wavelength range: {min_wavelength}nm to {max_wavelength}nm")
    interp_data_list = []
    
    for cam_idx, camera_data in enumerate(data_np):
        # Create SpectralDistribution objects for each channel (R, G, B)
        channel_sds = []
        for channel_idx in range(3):  # R, G, B
            # Create a spectral distribution for this channel
            sd = SpectralDistribution(
                dict(zip(wave_lengths, camera_data[channel_idx])),
                name=f'Camera_{cam_idx}_Channel_{channel_idx}'
            )
            # Align/interpolate to target shape
            sd = sd.align(target_shape)
            channel_sds.append(sd.values)
        
        # Stack the interpolated channels
        interp_camera = np.stack(channel_sds, axis=0)
        interp_data_list.append(interp_camera)
    
    # Stack all cameras
    interp_data_np = np.array(interp_data_list)
    print(f"Original shape: {data_np.shape}, Interpolated shape: {interp_data_np.shape}")
    
    return interp_data_np, target_shape.wavelengths

def build_synthetic_dataset(n_samples: int,
                            shape: SpectralShape,
                            camera_sens: np.ndarray,
                            illuminant: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset of (RGB, XYZ) pairs for `n_samples` synthetic R(λ).
    Returns:
      - RGB: shape (n_samples, 3)
      - XYZ: shape (n_samples, 3)
    """
    # load observer CMFs
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer'] \
               .copy().align(shape).values.T  # (3, N)
    Δλ = float(shape.interval)
    rgb_list = []
    xyz_list = []

    print(f"Generating {n_samples} synthetic spectra...")
    for i in range(n_samples):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i}/{n_samples} samples...")
        R = generate_reflectance_spectrum(shape.wavelengths)
        # compute camera RGB: ∑ illuminant(λ)*R(λ)*sens(λ)*Δλ
        rgb = (illuminant * R)[None, :] * camera_sens  # (3, N)
        rgb = rgb.sum(axis=1) * Δλ

        # compute XYZ: ∑ illuminant*R*cmfs*Δλ
        xyz = (illuminant * R)[None, :] * cmfs  # (3, N)
        xyz = xyz.sum(axis=1) * Δλ

        rgb_list.append(rgb)
        xyz_list.append(xyz)

    return np.vstack(rgb_list), np.vstack(xyz_list)

if __name__ == '__main__':
    # Initial placeholder for spectral shape - will be updated after camera sensitivity is loaded
    shape = SpectralShape(380, 735, 1)
    
    # load camera sensitivity first to determine the wavelength range
    camera_file = resolve("${sim_root}/sony.txt")
    try:
        # Use the improved load_camspecs_specs which now always interpolates to 1nm
        camera_sens, cam_wavelengths = load_camspecs_specs([camera_file])
        # Extract the first camera sensitivity (shape should be 3, wavelengths)
        camera_sens = camera_sens[0]
        print(f"Successfully loaded and interpolated camera sensitivity, shape: {camera_sens.shape}")
        
        # Update the spectral shape to match the interpolated camera sensitivity
        shape = SpectralShape(cam_wavelengths[0], cam_wavelengths[-1], 1)
        print(f"Adjusted spectral shape to match camera data: {shape}")
    except Exception as e:
        print(f"Error loading camera sensitivity: {e}")
        print("Trying alternative method...")
        camera_sens = pd.read_csv(camera_file, sep="\s+", comment='#')
        sens_dict = dict(zip(camera_sens['Lambda'].values, camera_sens[['R','G','B']].values.T))
        sd = SpectralDistribution(sens_dict, name=os.path.basename(camera_file))
        sd = sd.align(shape)
        camera_sens = sd.values  # shape: (3, N)
    
    # Now load D65 illuminant to match the camera wavelength range
    d65 = SDS_ILLUMINANTS['D65'].copy().align(shape).values
    
    print(f"Camera sensitivity shape: {camera_sens.shape}")
    print(f"Illuminant shape: {d65.shape}")
    
    # Ensure shapes match
    if camera_sens.shape[1] != len(d65):
        raise ValueError(f"Camera sensitivity wavelengths ({camera_sens.shape[1]}) don't match illuminant ({len(d65)})")
    
    # Reduce the number of samples for testing
    n_samples = 1000000  # Start with a smaller number for testing
    
    # build sample training set
    RGB_train, XYZ_train = build_synthetic_dataset(
        n_samples=n_samples,
        shape=shape,
        camera_sens=camera_sens,
        illuminant=d65
    )

    # save to disk
    out_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'RGB_validation.npy'), RGB_train)
    np.save(os.path.join(out_dir, 'XYZ_validation.npy'), XYZ_train)
    print(f"Saved synthetic dataset with {RGB_train.shape[0]} samples to {out_dir}")