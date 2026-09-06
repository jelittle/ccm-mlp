"""Path indirection for configs.

Configs never contain absolute paths. They refer to a small set of named roots
using ``${name}`` placeholders, e.g.::

    src_path: "${raw_data_root}/LightboxCaptures/colorchecker24/Sony"

The roots themselves are declared once in ``training/configs/paths.yaml`` and are
interpreted relative to the repository root unless given as absolute paths.

Each root can also be overridden by an environment variable without editing any
file, which is the intended way to point the code at a large data volume:

    CC_RAW_DATA_ROOT=/mnt/captures python training/runner.py --config ...

Resolution order for a root, highest priority first:
    1. ``CC_<ROOT_NAME>`` environment variable
    2. the value in ``paths.yaml``
    3. the built-in default below
"""

import os
import re
from pathlib import Path

import yaml

# training/utilities/paths.py -> training/utilities -> training -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

PATHS_FILE = REPO_ROOT / "training" / "configs" / "paths.yaml"

# Fallbacks used when paths.yaml is absent or a key is missing.
DEFAULT_ROOTS = {
    "raw_data_root": "data",          # unprocessed captures straight off the camera
    "data_root": "project_data",      # processed patches, splits, targets
    "sim_root": "assets/simulated",   # spectrally simulated training data
    "output_root": "outputs",         # checkpoints, .npy dumps, metrics
    "repo_root": str(REPO_ROOT),
}

_PLACEHOLDER = re.compile(r"\$\{([a-z_][a-z0-9_]*)\}")


def load_roots(paths_file=None):
    """Return the resolved root table, applying env-var overrides."""
    paths_file = Path(paths_file) if paths_file else PATHS_FILE

    roots = dict(DEFAULT_ROOTS)
    if paths_file.is_file():
        with open(paths_file, "r") as fh:
            declared = (yaml.safe_load(fh) or {}).get("paths", {}) or {}
        roots.update({k: v for k, v in declared.items() if v is not None})

    resolved = {}
    for name, value in roots.items():
        env = os.environ.get("CC_" + name.upper())
        if env:
            value = env
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved[name] = str(path)
    return resolved


def substitute(value, roots):
    """Recursively replace ``${root}`` placeholders in strings/lists/dicts."""
    if isinstance(value, str):
        def swap(match):
            name = match.group(1)
            if name not in roots:
                raise KeyError(
                    "Unknown path root '%s'. Known roots: %s. "
                    "Declare it in training/configs/paths.yaml."
                    % (name, ", ".join(sorted(roots)))
                )
            return roots[name]
        return _PLACEHOLDER.sub(swap, value)
    if isinstance(value, dict):
        return {k: substitute(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, roots) for v in value]
    return value


def resolve_config(config, paths_file=None):
    """Expand every ``${root}`` placeholder in a merged config, in place-ish.

    Call this once, immediately after the config files have been merged.
    """
    roots = load_roots(paths_file)
    resolved = substitute(config, roots)
    resolved["paths"] = roots
    return resolved


def resolve(value, paths_file=None):
    """Expand ``${root}`` placeholders in a single string (or nested structure).

    Convenience wrapper for standalone scripts that hold a path constant rather
    than reading a config, e.g.::

        SPLITS_DIR = resolve("${data_root}/LightboxCaptures/splits")
    """
    return substitute(value, load_roots(paths_file))
