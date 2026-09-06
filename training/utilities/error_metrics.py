"""Pure error-metric / colour-conversion helpers shared by the metrics script and figures.

These used to live inside the metrics-compiling script, which could not be imported (it read
sys.argv[1] and builds hardcoded cluster paths at module level). Keeping them here lets
figure code reuse the exact same implementations instead of copy-pasting them.
"""
import cv2
import numpy as np
import colour
from colour import CCS_ILLUMINANTS

colour.utilities.filter_warnings(colour_usage_warnings=True)


def xyz_to_srgb(xyz):
    x = np.asarray(xyz)
    batched = x.ndim == 4
    if not batched: x = x[None]

    N, _, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, 3)
    srgb = colour.XYZ_to_sRGB(x, CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'], apply_cctf_encoding=True)
    srgb = np.clip(srgb, 0, 1).reshape(N, H, W, 3).transpose(0, 3, 1, 2)

    return srgb if batched else srgb[0]


def angular_error(a, b, eps=1e-8):
    dot = (a * b).sum(0)
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    cos = np.clip(dot / (na * nb + eps), -1, 1)
    ang = np.degrees(np.arccos(cos))
    return ang[None, ...]


def error_to_cmap(error, vmin, vmax):
    err = np.squeeze(error)
    norm = np.clip((err - vmin) / (vmax - vmin), 0, 1)
    cmap = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)


def delta_e_map(a, b, space='XYZ', method='CIE 2000', illuminant='D50'):
    C,H,W = a.shape
    A, B = a.reshape(C,-1).T, b.reshape(C,-1).T
    if space.lower() == 'xyz':
        wp = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        A, B = colour.XYZ_to_Lab(A, wp), colour.XYZ_to_Lab(B, wp)
    elif space.lower() != 'lab':
        raise ValueError("space must be 'XYZ' or 'Lab'")
    dE = colour.delta_E(A, B, method=method)
    return dE.reshape(1, H, W)


def delta_c_map(a, b, space='XYZ', illuminant='D50'):
    C, H, W = a.shape
    A, B = a.reshape(C, -1).T, b.reshape(C, -1).T

    # Convert to Lab if in XYZ space
    if space.lower() == 'xyz':
        wp = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
        A = colour.XYZ_to_Lab(A, wp)
        B = colour.XYZ_to_Lab(B, wp)
    elif space.lower() != 'lab':
        raise ValueError("space must be 'XYZ' or 'Lab'")

    # Compute chroma values
    C_A = np.sqrt(A[:, 1] ** 2 + A[:, 2] ** 2)
    C_B = np.sqrt(B[:, 1] ** 2 + B[:, 2] ** 2)

    # ΔC
    delta_C = np.abs(C_A - C_B)
    return delta_C.reshape(1, H, W)
