"""CST calibration utilities.

Importing this package also makes the repository's other top-level packages
importable when scripts are run directly (``python training/runner.py``) rather
than after ``pip install -e .``. Without this, `camera_pipeline` -- which lives
under third_party/ -- is not on sys.path and every entry point fails at import.

This is a no-op once the package is properly installed.
"""

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]

for _extra in (_REPO_ROOT / "third_party", _REPO_ROOT / "training", _REPO_ROOT):
    _p = str(_extra)
    if _extra.is_dir() and _p not in _sys.path:
        _sys.path.append(_p)
