"""Checkpoint I/O.

Models are saved as **state dicts** (``model.pt``), not whole-object pickles.
The historical format was ``torch.save(model, ...)`` producing ``model.pth``,
which has three problems:

1. ``torch.load`` defaults to ``weights_only=True`` from PyTorch 2.6 onward and
   refuses whole-object pickles outright, so those files stop loading entirely
   on any current install.
2. Pickle stores a *reference* to the model's class by import path, so the file
   only loads where that exact module layout still exists.
3. Unpickling executes arbitrary code, which makes distributing such a file
   unsafe.

Legacy ``model.pth`` files are still readable via the fallback below, so
existing runs keep working; new runs always write ``model.pt``.
"""

import logging
import os

import torch
import yaml


def _torch_load(path, device, weights_only):
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=device)


def save_checkpoint(model, save_dir, model_config=None):
    """Write ``model.pt`` (state dict) plus a ``config.yaml`` sidecar.

    The sidecar is what makes the weights usable: the model cannot be rebuilt
    without ``type``/``neurons``/``layers``/``input_dim``/``output_dim``, and
    ``matrix_type`` in particular silently defaults to "linear" if missing,
    which would produce wrong results rather than an error.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), path)

    if model_config is not None:
        with open(os.path.join(save_dir, "config.yaml"), "w") as fh:
            yaml.safe_dump({"model": dict(model_config)}, fh, sort_keys=False)
    return path


def load_checkpoint(save_dir, model_config, device="cpu", use_tinycudann=None):
    """Load a checkpoint, preferring the state-dict format.

    Falls back to a legacy whole-object ``model.pth`` with a warning so that
    runs predating the format change still load.
    """
    from .model import get_model  # local import: avoids a circular import

    pt = os.path.join(save_dir, "model.pt")
    pth = os.path.join(save_dir, "model.pth")

    if os.path.isfile(pt):
        model = get_model(model_config, use_tinycudann=use_tinycudann)
        state = _torch_load(pt, device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)

    if os.path.isfile(pth):
        logging.warning(
            "Loading legacy whole-object checkpoint %s. Re-save it as a state "
            "dict (model.pt); this format will not load on PyTorch >= 2.6.", pth
        )
        return _torch_load(pth, device, weights_only=False)

    raise FileNotFoundError("No model.pt or model.pth in %s" % save_dir)
