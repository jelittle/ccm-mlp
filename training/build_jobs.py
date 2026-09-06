import os
import glob
import subprocess
from pathlib import Path as _P

# every relative path below is relative to training/, not the caller\'s cwd
_HERE = _P(__file__).resolve().parent
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from utilities.paths import resolve

if len(sys.argv) > 1:
    experiments = sys.argv[1]
    assert experiments in [
        "ablation_lightbox",
        "lightbox",
        "lightbox_ablation_canon",
        "lightbox_ablation_not_pixel",
        "lightbox_cccnn_and_expinv",
        "lightbox_mlp",
        "lightbox_pixel",
        "lightbox_sim",
        "lightbox_wild",
        "lsmi",
        "luts",
        "polynomials",
        "precon_ablation",
        "root_lightbox_all",
        "root_precon_all",
        "root_whitepoint_error",
        "single_led",
        "whitepoint_error"
    ], f"Invalid experiment name: {experiments}"


if experiments == "lightbox":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
  

    models_dir = resolve("${repo_root}/training/configs/models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),
    ]

if experiments == "lightbox_wild":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))

    models_dir = resolve("${repo_root}/training/configs/models")
    model_configs = [mc for mc in sorted(glob.glob(os.path.join(models_dir, "*.yaml"))) if not mc.endswith("ANG.yaml")]
    # print(model_configs)

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/wild_lightbox_experiments.yaml"),
    ]
elif experiments == "polynomials":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    #only include samsung and pixel
    # base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    base_configs = [resolve("${repo_root}/training/configs/cameras/samsung.yaml"),
                    resolve("${repo_root}/training/configs/cameras/pixel.yaml")]
  

    models_dir = resolve("${repo_root}/training/configs/ablations/polynomial_models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml"))) #just these 2 need to be added

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/polynomial_lightbox_experiments.yaml"),
    ]
elif experiments == "whitepoint_error":
    max_workers = 1
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    #only include pixel
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))


    models_dir = resolve("${repo_root}/training/configs/models")
    # model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml"))) #just these 2 need to be added
    model_configs = sorted(glob.glob(os.path.join(models_dir, "ANG.yaml"))) #just these 2 need to be added

    additional_configs = sorted(glob.glob(os.path.join(resolve("${repo_root}/training/configs/ablations/whitepoint_error_configs"), "*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/whitepoint_error_experiments.yaml"),
    ]

elif experiments == "precon_ablation":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    #only include samsung and pixel
    # base_configs = ["${repo_root}/training/configs/cameras/samsung.yaml",
    base_configs = [resolve("${repo_root}/training/configs/cameras/pixel.yaml")]


    models_dir = resolve("${repo_root}/training/configs/ablations/precon_ablation_models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml"))) #just these 2 need to be added
    # model_configs = ["${repo_root}/training/configs/ablations/precon_ablation_models/EXPINV2DXY0.05Precon.yaml"]
    # model_configs += sorted(glob.glob(os.path.join(models_dir, "EXPINV*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/precon_ablation_lightbox_experiments.yaml"),
    ]
elif experiments == "lsmi":
    # LSMI, including the multi-illuminant (mixed) split -- see README, work in progress.
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/lsmi_cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))

    models_dir = resolve("${repo_root}/training/configs/models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = [mc for mc in model_configs if os.path.basename(mc) not in
                    ["3cst.yaml","CCCNN2DXY.yaml","CCCNN2DWP.yaml","CCCNN1DCCT.yaml","EXPINV1DCCT.yaml","EXPINV2DWP.yaml","EXPINV2DXY.yaml"]]
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/lsmi_experiments.yaml"),
    ]
elif experiments == "lightbox_sim":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    #Only train on pixel camera
    base_configs = [resolve("${repo_root}/training/configs/cameras/pixel.yaml")]
  

    models_dir = resolve("${repo_root}/training/configs/models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "MLP*.yaml"))) #only mlp models
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/sim_lightbox_experiments.yaml"),
        #"${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml",

    ]
elif experiments == "lightbox_cccnn_and_expinv":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    #base_configs = ["${repo_root}/training/configs/cameras/samsung.yaml",
    #                "${repo_root}/training/configs/cameras/pixel.yaml"]

    models_dir = resolve("${repo_root}/training/configs/models")
    #model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = sorted(glob.glob(os.path.join(models_dir, "CCCNN2D*.yaml"))) #only CCCNN2D models
    model_configs += sorted(glob.glob(os.path.join(models_dir, "EXPINV2D*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),
    ]
elif experiments == "lightbox_pixel":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    base_configs = [resolve("${repo_root}/training/configs/cameras/pixel.yaml")]

    models_dir = resolve("${repo_root}/training/configs/models")
    #model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = sorted(glob.glob(os.path.join(models_dir, "MLP*.yaml"))) #only mlp models
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),      

    ]
elif experiments == "lightbox_mlp":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))

    models_dir = resolve("${repo_root}/training/configs/models")
    #model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = sorted(glob.glob(os.path.join(models_dir, "MLP*.yaml"))) #only mlp models
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),
        # "${repo_root}/training/configs/experiments/blend_lightbox_experiments.yaml",
    ]
elif experiments == "lightbox_ablation_not_pixel":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    base_configs = [bc for bc in sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml"))) if "pixel" not in bc]

    models_dir = resolve("${repo_root}/training/configs/ablations/ablation_models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/ablation_lightbox_experiments.yaml"),
    ]
elif experiments == "lightbox_ablation_canon":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    base_configs = [bc for bc in sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml"))) if "pixel" not in bc]

    models_dir = resolve("${repo_root}/training/configs/ablations/ablation_models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = [resolve("${repo_root}/training/configs/ablations/ablation_models/MLP2DXY32X1.yaml")
                     ,resolve("${repo_root}/training/configs/ablations/ablation_models/MLP2DXY32X2.yaml"),
                    resolve("${repo_root}/training/configs/ablations/ablation_models/MLP2DXY32X3.yaml") ]
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/ablation_lightbox_experiments.yaml"),
    ]

elif experiments == "luts":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))

    models_dir = resolve("${repo_root}/training/configs/ablations/lut_models")
    # model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))
    model_configs = [resolve("${repo_root}/training/configs/ablations/lut_models/lut30x30.yaml")]
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/lut_lightbox_experiments.yaml"),
    ]
elif experiments == "ablation_lightbox":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    models_dir = resolve("${repo_root}/training/configs/ablations/ablation_models")
    model_configs = sorted(glob.glob(os.path.join(models_dir, "*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/ablation_lightbox_experiments.yaml"),
    ]

elif experiments == "root_lightbox_all":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
   
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    additional_configs=[resolve("${repo_root}/training/configs/ablations/poly_override/rootpoly2.yaml")]

    models_dir = resolve("${repo_root}/training/configs/models")
    # ang=glob.glob(os.path.join(models_dir, "*MLP2DANG*.yaml"))
    # xy=glob.glob(os.path.join(models_dir, "*MLP2DXY*.yaml"))
    # model_configs = ang+xy


    # models_dir = "${repo_root}/training/configs/models"
    # model_configs = [mc for mc in sorted(glob.glob(os.path.join(models_dir, "*.yaml"))) if any(x in os.path.basename(mc) for x in ["MLP2DANG", "NN2DANG"])]
    # model_configs = [mc for mc in model_configs if os.path.basename(mc) in
    #                 ["MLP2DANG.yaml", "NN2DANG.yaml"]]
    model_configs = glob.glob(os.path.join(models_dir, "*MLP1DCCT*.yaml"))
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),
    ]
elif experiments == "root_precon_all":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    #only include samsung and pixel
    # base_configs = ["${repo_root}/training/configs/cameras/samsung.yaml",
    base_configs = [resolve("${repo_root}/training/configs/cameras/pixel.yaml")]
    additional_configs=[resolve("${repo_root}/training/configs/ablations/poly_override/rootpoly2.yaml")]

    models_dir = resolve("${repo_root}/training/configs/ablations/precon_ablation_models")
    # ang = glob.glob(os.path.join(models_dir, "*MLP2DANG*.yaml"))
    model_configs = glob.glob(os.path.join(models_dir, "*MLP1DCCT*.yaml"))
    # xy = glob.glob(os.path.join(models_dir, "*MLP2DXY*.yaml"))
    # model_configs = ang + xy
    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/precon_ablation_lightbox_experiments.yaml"),
    ]
elif experiments == "root_whitepoint_error":
    max_workers = 4
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    #only include pixel
    base_configs = sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml")))
    base_configs = [bc for bc in sorted(glob.glob(os.path.join(base_configs_dir, "*.yaml"))) if "pixel" in bc]
    print(base_configs)
    models_dir = resolve("${repo_root}/training/configs/models")
    # model_configs =["${repo_root}/training/configs/ablations/polynomial_models/MLP2DXYRootPoly2.yaml","${repo_root}/training/configs/ablations/polynomial_models/MLP2DANGRootPoly2.yaml",]
    model_configs=[resolve("${repo_root}/training/configs/ablations/polynomial_models/MLP1DCCTRootPoly2.yaml")]
    additional_configs = sorted(glob.glob(os.path.join(resolve("${repo_root}/training/configs/ablations/whitepoint_error_configs"), "*.yaml")))

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/whitepoint_error_experiments.yaml"),
        # "${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml",
    ]
elif experiments == "single_led":
    # 7-CST (CCT-based) and RBF (xy-based, linear_rbf + tps) anchored on the
    # 7 single-LED illuminants (light000–light006), tested on dirchletplushoang.
    max_workers = 3
    base_configs_dir = str(_HERE / "configs") + "/cameras"
    base_configs = [resolve("${repo_root}/training/configs/cameras/pixel.yaml")]


    models_dir = resolve("${repo_root}/training/configs/models")
    model_configs = [
        os.path.join(models_dir, "7cst.yaml"),
        os.path.join(models_dir, "RBF2DXY_linear_rbf.yaml"),
        os.path.join(models_dir, "RBF2DXY_tps.yaml"),
    ]

    save_path_configs = [
        resolve("${repo_root}/training/configs/experiments/all_lightbox_experiments.yaml"),
    ]





def run_experiment(configs):
    # sys.executable, not "python": the sweep must use the same interpreter that
    # launched it, or a venv/conda env silently falls back to a system python
    # without the dependencies. cwd is pinned to training/ so the relative config
    # paths below resolve wherever the user invoked this from.
    cmd = [sys.executable, str(_HERE / "runner.py"), "--config"] + configs
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_HERE))

#Run up to 20 experiments at once, but stagger submission by 10s

futures = {}

additional_configs = locals().get("additional_configs") or [None]

axes = {
    "base": base_configs,
    "model": model_configs,
    "save": save_path_configs,
    "addl": additional_configs,   # becomes a no-op when it's [None]
}

keys = list(axes.keys())
futures = {}

def make_cfg_list(p):
    # Keep a single canonical order; drop Nones
    order = ["base", "model", "save", "addl"]
    return [p[k] for k in order if p.get(k) is not None]

with ThreadPoolExecutor(max_workers=max_workers) as executor:

    for combo in product(*(axes[k] for k in keys)):
        params = {k: v for k, v in zip(keys, combo) if v is not None}
        cfg_list = make_cfg_list(params)

        fut = executor.submit(run_experiment, cfg_list)
        futures[fut] = params
        time.sleep(10)  #stagger by 10s

    for fut in as_completed(futures):
        p = futures[fut]
        try:
            result = fut.result()
            print(f"Finished: {p} (return code {getattr(result, 'returncode', 'n/a')})")
        except Exception as e:
            print(f"Error: {p}: {e}")