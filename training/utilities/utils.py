import logging

try:
    import wandb
except ImportError:  # optional: training runs offline without it
    wandb = None
import os
from datetime import datetime
import torch
import time
import numpy as np
from .model import get_model_output, get_model, copy_attributes, get_wrapped_model, get_model_inputs, test_times_get_model_output



def merge_dicts(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base

from torchinfo import summary



def calc_model_performance(model, config, device, runs=100, input_size=(1, 3, 720, 1080)):
    """
    Measures average performance of the model's forward pass using get_model_output(...)
    with a synthetic batch constructed inside this function.

    Args:
        model: torch.nn.Module
        model_config: dict with key 'type'
        device: "cuda" or "cpu"
        runs: number of iterations for averaging
        input_size: tuple for the dummy 'input' tensor shape (N,C,H,W) or (N, D, ...)

    Returns:
        avg_runtime (float): Average runtime in seconds.
    """
    model.eval()

    # --- Build a dummy batch that matches get_model_output(...) expectations ---
    # white_point: (N, 2); input: arbitrary image-like or feature tensor; 
    # target: only used by "bestfitsolver" — make it shapye-compatible with 'input'
    N = input_size[0] if len(input_size) > 0 else 1
    batch = {
        "wp":      torch.randn((N, 2), device=device).float(),
        "xy":      torch.randn((N, 2), device=device).float(),
        "xy_gt":   torch.randn((N, 2), device=device).float(),   # ← add this
        "ang":     torch.randn((N, 2), device=device).float(),
        "cct":     torch.randn((N, 1), device=device).float(),
        "cct_gt":  torch.randn((N, 1), device=device).float(),   # ← add this
        "wb_img":  torch.randn(input_size, device=device).float(),
        "target":  None,
        "illum_idx": torch.zeros(N, dtype=torch.long, device=device),
    }

    # keep a sane placeholder to avoid key errors even if unused
    batch["target"] = torch.zeros_like(batch["wb_img"])

    model_config = config["model"]
    summary_model = get_model(model_config, use_tinycudann=False).to(device)
    summary_model = copy_attributes(model, summary_model)
    summary_model = get_wrapped_model(summary_model, model_config)
    m_inputs = get_model_inputs(model, batch, device, config, test=True)
    m_inputs = {k: v for k, v in m_inputs.items() if v is not None}
    summary_model = summary_model.to(device)
    summary_model.eval()

    def input_constructor(input_res):
        # return a dict or tuple exactly as your forward() expects
        return m_inputs
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        summary_model,
        (128,),  # e.g. vector of dim 128
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    total_buffer_elems = 0
    for name, buf in model.named_buffers():
        n_elem = buf.numel()
        total_buffer_elems += n_elem

    macs = macs / (1024**2)  # convert to K
    model_size = (params + total_buffer_elems) * 4 / 1024

    if model_config["type"] in ["2cst", "original2cst"]:
        model_size = 2 * 3 * 3 * 4 / 1024  # 2 3x3 matrices
    elif model_config["type"] in ["3cst"]:
        model_size = 3 * 3 * 3 * 4 / 1024  # 3 3x3 matrices

    # Convenience CUDA sync helper
    def _maybe_sync():
        return torch.cuda.synchronize() if (device.startswith("cuda") and torch.cuda.is_available()) else None

    # Warm-up (no timing)
    with torch.no_grad():
        for _ in range(10):
            #_ = get_model_output(model, batch, device, model_config, test=True) #use the original model here as its faster
            _ = test_times_get_model_output(model, model_config, batch, device)

    _maybe_sync()
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = test_times_get_model_output(model, model_config, batch, device)#use the original model here as its faster

    _maybe_sync()
    end = time.time()

    avg_runtime = (end - start) / runs
    #convert to ms
    avg_runtime *= 1000
    print(f"Model size: {model_size:.2f} KB, MACs: {macs}, Avg runtime: {avg_runtime:.2f} ms")

    return macs, avg_runtime, model_size,

def build_run_name(config):
    # Extract values from config
    data_config = config['data']
    model_config = config['model']
    

    now = datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M_%S")

    model_type = model_config['type']
    if model_config.get("neurons"):
        model_descriptor = f"{model_config['neurons']}X{model_config['layers']}"
        if model_config.get("encoding") == "freq":
            model_descriptor += f"SIN"
        if model_config.get("activation") is not None:
            model_descriptor += f"ACT{model_config['activation']}"
        model_type += model_descriptor

    if model_config.get("lut_width"):
        model_type += f"{model_config['lut_width']}x{model_config['lut_width']}"
    

    run_name = f"{data_config['camera_name']}_{model_type}_{data_config['split']}"
    
    
    if "precon_amount" in model_config:
        run_name += f"_precon{model_config['precon_amount']}"
    
    if "matrix_type" in model_config:
        run_name += f"_mt{model_config['matrix_type']}"

    if data_config.get("wp_error", False):
        run_name += f"_wpe{data_config['wp_error']}"
    
    if data_config.get("wp_source"):
        run_name += f"_wps{data_config['wp_source']}"

    if config.get('append_date', True): #timestamp at the end of the run name
        run_name += f"_{timestamp}"
    



    return run_name

def wandb_enabled(config):
    """True only when wandb is installed AND not disabled in the config."""
    if wandb is None:
        return False
    return not (config.get("wandb") or {}).get("disabled", False)


def setup_wandb(config, device):
    run_name = build_run_name(config)
    if not wandb_enabled(config):
        logging.info("W&B disabled or not installed; running offline.")
        return run_name
    wandb.init(project=(config.get("wandb") or {}).get("project", "CST-Calibration"),
               entity=(config.get("wandb") or {}).get("entity") or None,
               name=run_name)

    #log all config values to wandb
    wandb.config.update(config)
    wandb.config.update({"device": device})
    return run_name