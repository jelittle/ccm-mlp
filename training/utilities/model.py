import torch

# tinycudann is optional. It is a CUDA source build (see setup/install_tinycudann.sh)
# and cannot be pip-installed, so it must never be imported at module load time --
# that would make the whole package unimportable on a plain `pip install`.
# The pure-PyTorch path in build_torch_network() is the default; tinycudann is a
# speed optimisation used for the paper's timing numbers.
tcnn = None


def _load_tcnn():
    """Import tinycudann on first use, with an actionable error if it is missing."""
    global tcnn
    if tcnn is None:
        try:
            import tinycudann as _tcnn
        except ImportError as exc:
            raise ImportError(
                "tinycudann is required for use_tinycudann=True but is not "
                "installed. Either build it (see setup/install_tinycudann.sh), "
                "or set 'use_tinycudann: false' under 'model:' in your config "
                "to use the equivalent pure-PyTorch network."
            ) from exc
        tcnn = _tcnn
    return tcnn

from .LUT import LUT

from .bestfitsolver import BestFitSolver
from .two_cst import TwoCst
from .three_cst import ThreeCST
from .seven_cst import SevenCST
from .rbf_cst import RBFCST
from .cccnn import CCCNN

import logging
import os
import json
import numpy as np
import torch.nn.functional as F
from .NN import NN
from .MLP import MLP
from colour.difference import delta_E_CIE2000  
import colour
colour.utilities.filter_warnings(colour_usage_warnings=True)

def build_tinycudann_network(model_config):
    tcnn = _load_tcnn()
    

    if model_config["encoding"] == "freq":
        encoding_config = {
                "otype": "Frequency",
                "n_frequencies": model_config["n_frequencies"]
        }
    elif model_config["encoding"] == "id":
        encoding_config = {
                "otype": "Identity"
        }
    elif model_config["encoding"] == "idplusfreq":
        encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": model_config["input_dim"],
                        "otype": "Identity"
                    },
                    {
                        "n_dims_to_encode": model_config["input_dim"]+1,
                        "otype": "Frequency",
                        "n_frequencies": model_config["n_frequencies"]
                    }
                ]
        }
    else:
        #throw error
        raise ValueError(f"Unknown encoding type: {model_config['encoding']}")
    network_config = {
        "otype": "CutlassMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": model_config["neurons"],
        "n_hidden_layers": model_config["layers"]
    }
    if model_config.get("activation") is not None:
        network_config["activation"] = model_config["activation"]
        print("Using activation:", model_config["activation"])

    if model_config["encoding"] == "ipplusfreq":
        net = tcnn.NetworkWithInputEncoding(n_input_dims=model_config["input_dim"]*2, n_output_dims=model_config["output_dim"], encoding_config=encoding_config, network_config=network_config)
    else:
        net = tcnn.NetworkWithInputEncoding(n_input_dims=model_config["input_dim"], n_output_dims=model_config["output_dim"], encoding_config=encoding_config, network_config=network_config)
    return net

import torch
import torch.nn as nn

class _CastToParamDtype(nn.Module):
    """Cast the input to float32 before the Linear stack.

    tinycudann casts its input internally, so the fused path tolerates the
    float64 white points the dataloader yields. nn.Linear does not, and fails
    with "mat1 and mat2 must have the same dtype". This makes the pure-PyTorch
    fallback accept the same inputs as the tinycudann path.
    """

    def forward(self, x):
        return x.float()


def build_torch_network(model_config):
    # Encoding layer
    if model_config["encoding"] == "freq":
        # Positional/Frequency encoding
        class FrequencyEncoding(nn.Module):
            def __init__(self, input_dim, n_frequencies):
                super().__init__()
                self.input_dim = input_dim
                self.n_frequencies = n_frequencies
                self.freq_bands = 2.0 ** torch.linspace(0, n_frequencies-1, n_frequencies)

            def forward(self, x):
                out = [x]
                for freq in self.freq_bands:
                    out.append(torch.sin(freq * x))
                    out.append(torch.cos(freq * x))
                return torch.cat(out, dim=-1)

        encoding = FrequencyEncoding(model_config["input_dim"], model_config["n_frequencies"])
        encoded_dim = model_config["input_dim"] * (2 * model_config["n_frequencies"] + 1)

    elif model_config["encoding"] == "id":
        encoding = nn.Identity()
        encoded_dim = model_config["input_dim"]

    else:
        raise ValueError(f"Unknown encoding type: {model_config['encoding']}")

    # Build MLP layers
    layers = []
    in_dim = encoded_dim
    for _ in range(model_config["layers"]):
        layers.append(nn.Linear(in_dim, model_config["neurons"]))
        layers.append(nn.ReLU())
        in_dim = model_config["neurons"]

    # Final layer
    layers.append(nn.Linear(in_dim, model_config["output_dim"]))

    # Combine everything
    net = nn.Sequential(
        _CastToParamDtype(),
        encoding,
        nn.Sequential(*layers)
    )
    return net


def get_model(model_config, use_tinycudann=None):
    if use_tinycudann is None:
        # torch-first by default so the repo runs on a plain pip install
        use_tinycudann = model_config.get("use_tinycudann", False)

    if model_config["type"] in ["MLP2DWP", "MLP2DXY", "MLP1DCCT","MLP2DANG"]:
         
     
        if use_tinycudann:
            net = build_tinycudann_network(model_config)
        else:
            net = build_torch_network(model_config)
        if model_config["type"] in ["MLP2DXY"]:
            net = MLP(net, mode="xy", matrix_type= model_config.get("matrix_type", "linear"))
        elif model_config["type"] == "MLP1DCCT":
            net = MLP(net, mode="cct", matrix_type= model_config.get("matrix_type", "linear"))
        elif model_config["type"] == "MLP2DWP":
            net = MLP(net, mode="wp", matrix_type= model_config.get("matrix_type", "linear"))
        elif model_config["type"] == "MLP2DANG":
            net = MLP(net, mode="ang", matrix_type= model_config.get("matrix_type", "linear"))

    elif model_config["type"] in ["CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DWP","EXPINV2DANG"]:
        if use_tinycudann:
            mlp = build_tinycudann_network(model_config)
        else:
            mlp = build_torch_network(model_config)
        if model_config["type"] == "CCCNN1DCCT":
            net = CCCNN(mlp, mode="cct")
        elif model_config["type"] == "CCCNN2DXY":
            net = CCCNN(mlp, mode="xy")
        elif model_config["type"] == "CCCNN2DWP":
            net = CCCNN(mlp, mode="wp")
        elif model_config["type"] == "CCCNN2DANG":
            net = CCCNN(mlp, mode="ang")
        elif model_config["type"] == "EXPINV1DCCT":
            net = CCCNN(mlp, mode="cct", exposure_invariant=True)
        elif model_config["type"] == "EXPINV2DXY":
            net = CCCNN(mlp, mode="xy", exposure_invariant=True)
        elif model_config["type"] == "EXPINV2DWP":
            net = CCCNN(mlp, mode="wp", exposure_invariant=True)
        elif model_config["type"] == "EXPINV2DANG":
            net = CCCNN(mlp, mode="wp", exposure_invariant=True)
    elif model_config["type"] == "LUT":
        net = LUT(model_config["lut_width"])
    elif model_config["type"] == "bestfitsolver":
        net = BestFitSolver()
    elif model_config["type"] in ["original2cst", "2cst"]:
        net = TwoCst()
    elif model_config["type"] == "3cst":
        net = ThreeCST()
    elif model_config["type"] == "7cst":
        net = SevenCST()
    elif model_config["type"] == "RBF2DXY":
        net = RBFCST(model_config)
    elif model_config["type"] == "NN2DXY":
        net = NN(mode="xy")
    elif model_config["type"] == "NN2DWP":
        net = NN(mode="wp")
    elif model_config["type"] == "NN2DANG":
        net = NN(mode="ang")
    elif model_config["type"] == "NN1DCCT":
        net = NN(mode="cct")
       
    return net

def get_white_point_type(batch, model_config):
    mtype = model_config['type']
    if mtype in ["LUT"]:
        mtype = model_config['model_distill']['type']
    if mtype in ["MLP2DXY", "CCCNN2DXY", "EXPINV2DXY", "NN2DXY"]:
        white_point = batch["xy"]
    elif mtype in ["MLP1DCCT", "CCCNN1DCCT", "EXPINV1DCCT", "NN1DCCT", "2cst", "original2cst", "3cst", "bestfitsolver"]:
        white_point = batch["cct"]
    elif mtype == "7cst":
        white_point = batch["cct_gt"]
    elif mtype == "RBF2DXY":
        white_point = batch["xy_gt"]
    elif mtype in ["MLP2DWP", "CCCNN2DWP", "EXPINV2DWP", "NN2DWP"]:
        white_point = batch["wp"]
    elif mtype in ["MLP2DANG","CCCNN2DANG", "EXPINV2DANG", "NN2DANG"]:
        white_point = batch["ang"]
    return white_point

def get_model_inputs(model, batch, device, config, test=False, normalize_wp=True, mode="patch", stochastic_prec = 0):

    targets = batch['target'].to(device)
    mtype = config["model"]['type']
    if mtype in ["LUT"]:
        mtype = config["model"]['model_distill']['type']
    if mode in ["mixed_patch", "mixed_full"] and mtype == "bestfitsolver":
        wb_img = batch['illuminant_colorchecker'].to(device).float()
        wb_img = rearrange(wb_img, 'b checker h w channel -> b checker channel h w')
    elif mode in ["full"] and mtype == "bestfitsolver":
        wb_img = batch['wb_img'].to(device).float()
    elif mode in ["patch", "mixed_patch"]:  # special case for best fit solver
        wb_img = batch['wb_img'].to(device).float()
    elif mode in ["full", "mixed_full"]:
        wb_img = batch['wbfull_img'].to(device).float()

    white_point = get_white_point_type(batch, config["model"]).to(device)
    
    if stochastic_prec != 0:
        #add random noise to white point defined by stochastic_prec
        noise = torch.randn_like(white_point) * stochastic_prec
        white_point += noise

    if mtype == "bestfitsolver":
        if mode in ["mixed_patch", "mixed_full"]:
            per_illum_target = batch['per_illum_target'].to(device).float()
            return {"wb_img": wb_img, "targets": per_illum_target, "white_point": white_point}
        else:
            return {"wb_img": wb_img, "targets": targets, "white_point": white_point}

    elif mtype in ["original2cst", "2cst", "3cst", "7cst", "RBF2DXY", "NN2DXY", "NN2DWP","NN2DANG","NN1DCCT",
                   "MLP2DWP", "MLP2DXY","MLP2DANG", "MLP1DCCT",
                   "CCCNN1DCCT", "CCCNN2DXY", "CCCNN2DWP","CCCNN2DANG",
                   "EXPINV1DCCT", "EXPINV2DXY", "EXPINV2DANG","EXPINV2DWP"]:
        return {"white_point": white_point,
                "wb_img": wb_img}



DIRECT_PREDICTION_NETS = {
    "CCCNN1DCCT","CCCNN2DXY","CCCNN2DWP","CCCNN2DANG",
    "EXPINV1DCCT","EXPINV2DXY","EXPINV2DWP","EXPINV2DANG",
}

class ModelWithCST(nn.Module):
    def __init__(self, base, mtype, cst_fn, matrix_type):
        super().__init__(); self.base, self.mtype, self.cst, self.matrix_type = base, mtype, cst_fn, matrix_type
    def forward(self, **kw):
        out = self.base(**kw).float()
        return self.cst(kw["wb_img"], out, matrix_type = self.matrix_type) if self.mtype not in DIRECT_PREDICTION_NETS and "wb_img" in kw else out

def get_wrapped_model(model, model_config):
    return ModelWithCST(model, model_config['type'], apply_cst, model_config.get("matrix_type", "linear"))


def get_model_output(model, batch, device, config, test=False, mode="patch", stochastic_prec = 0):
    inp = get_model_inputs(model, batch, device, config, test, mode=mode, stochastic_prec=stochastic_prec)
    if mode in ["mixed_patch", "mixed_full"]:
        #convert "white_point" using einops to merge first two dimensions
        from einops import rearrange
        b, num_csts, wp_info = inp["white_point"].shape
        inp["white_point"] = rearrange(inp["white_point"], 'b num_csts wp_info -> (b num_csts) wp_info')

    #For best fit solver, I need the color checker values for eahc position and illuminant
    
    out = model(**inp).float()

    #hack to make bestfitsolver to work on mixed mode... This sucks, but it will do for now.
    if mode == "mixed_patch" and config['model']['type'] == "bestfitsolver":
        inp["wb_img"] = batch['wb_img'].to(device).float()
    elif (mode == "mixed_full" and config['model']['type'] == "bestfitsolver"): 
        inp["wb_img"] = batch['wbfull_img'].to(device).float()
    elif mode == "full" and config['model']['type'] == "bestfitsolver":
        inp["wb_img"] = batch['wbfull_img'].to(device).float()

    if mode in ["mixed_patch", "mixed_full"]:
        #unmerge first two dimensions
        if out.ndim==3:
            #assert that last two dimensions are 3x3
            assert out.shape[0] == b * num_csts and out.shape[1] == 3
            out = rearrange(out, '(b num_csts) cst1 cst2 -> b num_csts cst1 cst2', b=b, num_csts=num_csts)
        elif out.ndim == 4 and config["model"]['type'] == "bestfitsolver":
            pass #all is good (already in correct order)
        else:
            print("NOT HANDLING out.ndim != 3 in mixed mode")
            exit(0)
        
    if test and config["model"]['type'] not in DIRECT_PREDICTION_NETS:
        try:
            if mode == "mixed_full":
                out = apply_cst(inp.get("wb_img"), out, batch.get("mixture_map"), mode=mode, matrix_type=config['model'].get("matrix_type", "linear"))
            elif mode == "mixed_patch":

                out_collect = [apply_cst(inp["wb_img"][:, i], out, batch["mixture_map_patches"][:, i], mode=mode, matrix_type=config['model'].get("matrix_type", "linear"))
                            for i in range(inp["wb_img"].shape[1])]
                # Transpose list of tuples -> tuple of lists
                out = tuple(torch.stack(items, dim=1) for items in zip(*out_collect))
            else:
                out = apply_cst(inp.get("wb_img"), out, matrix_type = config['model'].get("matrix_type", "linear"))
        except:
            print(config["model"]['type'])
     
            raise ValueError("Error applying CST during testing")
    return out

def test_times_get_model_output(model, model_config, inp, device):
    if model_config['type'] in ["MLP2DWP", "MLP2DXY","MLP2DANG"]:
        wp = torch.rand(1,2).to(device)
        out = model.mlp(wp)
    elif model_config['type'] in ["MLP1DCCT"]:
        cct = torch.rand(1,1).to(device)
        out = model.mlp(cct)
    elif model_config['type'] in ["CCCNN1DCCT", "EXPINV1DCCT"]:
        wb_img = inp.get("wb_img")
        wp = inp.get("wp")[:,0:1].view(1, 1, 1, 1).expand(-1, -1, wb_img.size(2), wb_img.size(3))  # [N, 1, H, W]
        wb_img = torch.cat([wb_img, wp], dim=1)
        #flatten spatial dimensions (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = wb_img.shape
        wb_img = wb_img.permute(0, 2, 3, 1)   # [N, H, W, C]
        wb_img = wb_img.reshape(-1, C)        # [N*H*W, C]
        out = model.mlp(wb_img)
        #unflatten spatial dimensions
        out = out.view(N, H, W, -1).permute(0, 3, 1, 2)  # [N, C, H, W]
    elif model_config['type'] in ["CCCNN2DXY", "EXPINV2DXY", "CCCNN2DWP","CCCNN2DANG", "EXPINV2DWP","EXPINV2DANG"]:
        wb_img = inp.get("wb_img")
        wp = inp.get("wp").view(1, 2, 1, 1).expand(-1, -1, wb_img.size(2), wb_img.size(3)) 
        wb_img = torch.cat([wb_img, wp], dim=1)
        #flatten spatial dimensions (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = wb_img.shape
        wb_img = wb_img.permute(0, 2, 3, 1)   # [N, H, W, C]
        wb_img = wb_img.reshape(-1, C)        # [N*H*W, C]
        out = model.mlp(wb_img)
        #unflatten spatial dimensions
        out = out.view(N, H, W, -1).permute(0, 3, 1, 2)  # [N, C, H, W]
    elif model_config['type'] in ["original2cst", "2cst", "3cst", "7cst", "RBF2DXY", "bestfitsolver", "NN2DXY", "NN2DWP","NN2DANG", "NN1DCCT", "LUT"]:
        #generate a random 3x3 matrix - all these methods are gonna be considered direct lookups
        out = torch.rand(1,9).to(device)

    if model_config['type'] not in DIRECT_PREDICTION_NETS:
        out = apply_cst(inp.get("wb_img"), out, matrix_type = model_config.get("matrix_type", "linear"))

def get_device():
    """Pick a compute device, preferring the GPU with the most free memory.

    Falls back to CPU when no CUDA device is visible, so the repo is runnable on
    a laptop. Returns (gpu_indices, device_string); gpu_indices is empty on CPU.
    """
    if not torch.cuda.is_available():
        logging.info("No CUDA device available; running on CPU.")
        return [], "cpu"

    gpus_chosen = get_freer_gpu(1)
    logging.info(f'Using devices {gpus_chosen}')

    all_gpus = sorted(range(get_total_num_gpus()))
    logging.info(f'ALL GPUS {all_gpus}')

    os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "50"
    device = f"cuda:{gpus_chosen[0]}"
    return gpus_chosen, device


def get_freer_gpu(num_gpus):
    """Indices of the `num_gpus` devices with the most free memory.

    Uses torch's own CUDA API rather than shelling out to `nvidia-smi | grep`,
    which is not portable off Linux and used to leave a file named 'tmp' in the
    working directory.
    """
    count = torch.cuda.device_count()
    if count == 0:
        return []

    free = []
    for idx in range(count):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except (RuntimeError, AssertionError):
            free_bytes = 0
        free.append(free_bytes)

    return sorted(range(count), key=lambda i: free[i], reverse=True)[:num_gpus]


#from deep_architect with MIT License
def get_total_num_gpus():
    """Number of visible CUDA devices (0 on a CPU-only machine)."""
    return torch.cuda.device_count()


def cosine_similarity_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    cosine_sim = torch.sum(x * y, dim=1)
    return 1 - cosine_sim.mean()


from einops import rearrange
def apply_cst(images, predictions, mixture_maps = None, mode=None, matrix_type="linear"):

    # images: (B, C, H, W)
    # predictions: (B, 9) or similar, reshaped into (B, 3, 3)

    #assert mixture_maps exists if mixed is True
    assert (mixture_maps is not None) if mode in ["mixed_patch", "mixed_full"] else True

    B, C, H, W = images.shape

    if mode in ["mixed_patch", "mixed_full"]:
        num_csts = predictions.shape[1]

        if matrix_type == "linear":
            csts = rearrange(predictions, 'b num_csts cst1 cst2 -> b cst1 cst2 num_csts')
            csts_T = csts.transpose(1, 2)
            images_flat = images.view(B, C, -1).permute(0, 2, 1).to(torch.float32)
            csts_T = csts_T.to(torch.float32)

            xyz_flat_all = torch.einsum('bhc,bcmk->bhmk', images_flat, csts_T)
            xyz_each = xyz_flat_all.permute(0, 3, 2, 1).reshape(B, num_csts, C, H, W)

            mixture_maps_flat = mixture_maps.to(torch.float32).contiguous().reshape(B, H * W, num_csts).to(images.device)
            xyz_flat = torch.einsum('bhmk,bhk->bhm', xyz_flat_all, mixture_maps_flat)
            xyz = xyz_flat.permute(0, 2, 1).view(B, C, H, W)
            return xyz, xyz_each

        xyz_each = torch.stack(
            [apply_cst(images, predictions[:, k], matrix_type=matrix_type) for k in range(num_csts)],
            dim=1
        )  # (B, K, C, H, W)

        mixture_maps_flat = mixture_maps.to(torch.float32).contiguous().reshape(B, H * W, num_csts).to(images.device)
        mix = mixture_maps_flat.permute(0, 2, 1).reshape(B, num_csts, H, W)  # (B, K, H, W)
        xyz = (xyz_each * mix[:, :, None, :, :]).sum(dim=1)  # (B, C, H, W)
        return xyz, xyz_each

    elif matrix_type == "linear":
        # Get CSTs (B, 3, 3)
        csts = predictions.view(-1, 3, 3)      # (B, 3, 3)
        csts_T = csts.transpose(1, 2)          # (B, 3, 3)
        # Flatten spatial dimensions, keep channels
        images_flat = images.view(B, C, -1).permute(0, 2, 1).to(torch.float32)  # (B, HW, C)
        csts_T = csts_T.to(torch.float32)
        # Apply CST to get xyz
        xyz_flat = torch.bmm(images_flat, csts_T)  # (B, HW, C)
        # Reshape back to (B, C, H, W)
        xyz = xyz_flat.permute(0, 2, 1).view(B, C, H, W)
        return xyz
    elif matrix_type == "polynomial2":
        # Predicted 3x9 CST per batch
        csts = predictions.view(-1, 3, 9)                         # (B, 3, 9)
        csts_T = csts.transpose(1, 2).to(torch.float32)           # (B, 9, 3)

        # Flatten spatial dims -> (B, HW, 3)
        images_flat = images.contiguous().view(B, C, -1).permute(0, 2, 1).to(torch.float32)

        # Build polynomial basis: [R, G, B, R^2, G^2, B^2, RG, RB, GB] -> (B, HW, 9)
        R, G, Bc = images_flat.unbind(dim=-1)                     # each (B, HW)
        images_poly = torch.stack([
            R, G, Bc,
            R * R, G * G, Bc * Bc,
            R * G, R * Bc, G * Bc
        ], dim=-1)                                                # (B, HW, 9)

        # Apply CST -> (B, HW, 3)
        xyz_flat = torch.bmm(images_poly, csts_T)

        # Reshape back -> (B, 3, H, W)
        xyz = xyz_flat.permute(0, 2, 1).contiguous().view(B, 3, H, W)
        return xyz
    elif matrix_type == "polynomial3":
        # Predicted 3x19 CST per batch
        csts = predictions.view(-1, 3, 19)                      # (B, 3, 19)
        csts_T = csts.transpose(1, 2).to(torch.float32)         # (B, 19, 3)

        # Flatten spatial dims -> (B, HW, 3)
        images_flat = images.contiguous().view(B, C, -1).permute(0, 2, 1).to(torch.float32)

        # Unbind channels
        R, G, Bc = images_flat.unbind(dim=-1)                   # each (B, HW)

        # Precompute powers and pairwise products
        R2, G2, B2 = R * R, G * G, Bc * Bc
        R3, G3, B3 = R2 * R, G2 * G, B2 * Bc
        RG, RB, GB = R * G, R * Bc, G * Bc

        # Mixed cubic terms
        R2G, R2B = R2 * G, R2 * Bc
        G2R, G2B = G2 * R, G2 * Bc
        B2R, B2G = B2 * R, B2 * G
        RGB = R * G * Bc

        # Build polynomial basis -> (B, HW, 19)
        images_poly = torch.stack([
            R, G, Bc,
            R2, G2, B2,
            RG, RB, GB,
            R3, G3, B3,
            R2G, R2B, G2R, G2B, B2R, B2G,
            RGB
        ], dim=-1)                                              # (B, HW, 19)

        # Apply CST -> (B, HW, 3)
        xyz_flat = torch.bmm(images_poly, csts_T)

        # Reshape back -> (B, 3, H, W)
        xyz = xyz_flat.permute(0, 2, 1).contiguous().view(B, 3, H, W)
        return xyz
    elif matrix_type == "rootpolynomial2":
        # Predicted 3x6 CST per batch
        csts = predictions.view(-1, 3, 6)                       # (B, 3, 6)
        csts_T = csts.transpose(1, 2).to(torch.float32)         # (B, 6, 3)

        # Flatten spatial dims -> (B, HW, 3)
        images_flat = images.contiguous().view(B, C, -1).permute(0, 2, 1).to(torch.float32)
        R, G, Bc = images_flat.unbind(dim=-1)                   # each (B, HW)

        # Root-polynomial terms
        eps = 1e-12
        sqrt_rg = torch.sqrt(torch.clamp_min(R * G, 0.0) + eps)
        sqrt_gb = torch.sqrt(torch.clamp_min(G * Bc, 0.0) + eps)
        sqrt_rb = torch.sqrt(torch.clamp_min(R * Bc, 0.0) + eps)

        # ρ2,3 = [r, g, b, √(rg), √(gb), √(rb)]
        feats = torch.stack([R, G, Bc, sqrt_rg, sqrt_gb, sqrt_rb], dim=-1)  # (B, HW, 6)

        # Apply CST -> (B, HW, 3)
        xyz_flat = torch.bmm(feats, csts_T)

        # Reshape back -> (B, 3, H, W)
        xyz = xyz_flat.permute(0, 2, 1).contiguous().view(B, 3, H, W)
        return xyz


    elif matrix_type == "rootpolynomial3":
        # Predicted 3x13 CST per batch
        csts = predictions.view(-1, 3, 13)                      # (B, 3, 13)
        csts_T = csts.transpose(1, 2).to(torch.float32)         # (B, 13, 3)

        # Flatten spatial dims -> (B, HW, 3)
        images_flat = images.contiguous().view(B, C, -1).permute(0, 2, 1).to(torch.float32)
        R, G, Bc = images_flat.unbind(dim=-1)                   # each (B, HW)

        # Helpers
        eps = 1e-12
        def sqrtp(x):  # stable sqrt for nonnegative domain
            return torch.sqrt(torch.clamp_min(x, 0.0) + eps)
        def cbrtp(x):  # stable real cube-root for nonnegative domain
            return torch.pow(torch.clamp_min(x, 0.0) + eps, 1.0 / 3.0)

        # Pairwise roots (degree 2)
        sqrt_rg = sqrtp(R * G)
        sqrt_gb = sqrtp(G * Bc)
        sqrt_rb = sqrtp(R * Bc)

        # Degree-3 root terms
        r_g2   = cbrtp(R * (G * G))       # ∛(r g²)
        g_b2   = cbrtp(G * (Bc * Bc))     # ∛(g b²)
        r_b2   = cbrtp(R * (Bc * Bc))     # ∛(r b²)
        g2_r   = cbrtp((G * G) * R)       # ∛(g² r)
        b2_g   = cbrtp((Bc * Bc) * G)     # ∛(b² g)
        b2_r   = cbrtp((Bc * Bc) * R)     # ∛(b² r)
        rgb    = cbrtp(R * G * Bc)        # ∛(r g b)

        # ρ3,3 = [r, g, b, √(rg), √(gb), √(rb), ∛(r g²), ∛(g b²), ∛(r b²),
        #         ∛(g² r), ∛(b² g), ∛(b² r), ∛(r g b)]
        feats = torch.stack([
            R, G, Bc,
            sqrt_rg, sqrt_gb, sqrt_rb,
            r_g2, g_b2, r_b2,
            g2_r, b2_g, b2_r,
            rgb
        ], dim=-1)                                              # (B, HW, 13)

        # Apply CST -> (B, HW, 3)
        xyz_flat = torch.bmm(feats, csts_T)

        # Reshape back -> (B, 3, H, W)
        xyz = xyz_flat.permute(0, 2, 1).contiguous().view(B, 3, H, W)
        return xyz


def cos_loss(xyz, target):
    return torch.mean(cosine_similarity_loss(xyz, target))

def delta_e_loss(xyz, targets):
    # --- compute ΔE per image ---
    # bring to CPU & numpy

    pred_xyz = xyz.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    tgt_xyz = targets.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()

    # convert to Lab
    lab_pred = colour.XYZ_to_Lab(pred_xyz)
    lab_tgt = colour.XYZ_to_Lab(tgt_xyz)
    # ΔE₀₀ for every pixel
    de00 = delta_E_CIE2000(lab_pred, lab_tgt)

     # reshape to (B, 1, H, W)
    B, C, H, W = xyz.shape
    de_per_img = torch.tensor(
        de00.reshape(B, H, W, 1).transpose(0, 3, 1, 2),
        device=xyz.device,
        dtype=xyz.dtype
    )

    return de_per_img


def ang_loss(xyz, target, return_per_image=False):
    ang_error = average_angular_error_torch(xyz, target)

    if return_per_image:
        return torch.mean(ang_error), ang_error
    return torch.mean(ang_error)


    

def average_angular_error_torch(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the average angular error between two batches of RGB images using PyTorch tensors.

    Args:
        src (torch.Tensor): First batch of RGB images of shape (B, C, H, W).
        target (torch.Tensor): Reference RGB image or batch of shape (B, C, H, W) or (C, H, W).

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the average angular error in degrees 
                      for each image pair in the batch.
    """
    assert src.shape == target.shape, "Source and target must have the same shape"
    B, C, H, W = src.shape
    t_B, t_C, t_H, t_W = target.shape

    # Flatten spatial dimensions -> (B, HW, C)
    src_flat = src.view(B, C, -1).permute(0, 2, 1)       # (B, HW, C)
    target_flat = target.view(t_B, t_C, -1).permute(0, 2, 1)  # (B, HW, C)
 

    # Normalize per-pixel vectors
    src_norm = src_flat / (src_flat.norm(dim=2, keepdim=True) + 1e-6)
    target_norm = target_flat / (target_flat.norm(dim=2, keepdim=True) + 1e-6)

    # Compute dot products per pixel
    if target_norm.shape[1] != 24:
        print(target.shape)
        print("norm shape",target_norm.shape)
        print("Target flat shape:", target_flat.shape)
    try:
        dot_products = torch.sum(src_norm * target_norm, dim=2)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
    except Exception as e:
        print("Error in computing dot products:", e)
        print("src_norm shape:", src_norm.shape)
        print("target_norm shape:", target_norm.shape)
        raise e

    # Angular errors (in radians)
    angular_errors = torch.acos(dot_products)  # (B, HW)

    # Mean angular error per image, convert to degrees
    average_error = torch.mean(angular_errors, dim=1)  # (B,)
    average_error = torch.rad2deg(average_error)

    return average_error



def copy_attributes(src_model, dst_model, skip=("mlp", "network", "tcnn", "_modules")):
    def _should_skip(k, v):
        if any(s in k.lower() for s in skip):
            return True
        # if isinstance(v, (nn.Module, nn.Parameter, torch.Tensor)):
        #     return True
        return False

    for k, v in vars(src_model).items():
        if not _should_skip(k, v):
            try:
                setattr(dst_model, k, v)
            except Exception as e:
                print(f"Could not copy attribute {k}: {e}")
    return dst_model


