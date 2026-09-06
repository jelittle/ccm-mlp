
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

import argparse, cv2, numpy as np
from pathlib import Path
from utilities.paths import resolve

LSMI_ROOT = Path(resolve("${data_root}/LSMI"))
DEBUG = False  # set False to process all files


def crop_resize_256(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0+s, x0:x0+s], (256, 256), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", choices=["galaxy", "nikon", "sony"])
    parser.add_argument("--lsmi_root", type=Path, default=LSMI_ROOT)
    args = parser.parse_args()

    src = args.lsmi_root / "colorchecker24" / args.camera / "rawnormalized"
    dst = args.lsmi_root / "colorchecker24" / args.camera / "cropped"
    dst.mkdir(exist_ok=True)

    paths = sorted(src.glob("*.npy"))
    if DEBUG:
        paths = paths[:10]
    for p in paths:
        img = np.load(p, allow_pickle=True).astype(np.float32)
        out = crop_resize_256(img)
        np.save(dst / p.name, out)
        print(f"{p.name} → {out.shape}")


if __name__ == "__main__":
    main()
