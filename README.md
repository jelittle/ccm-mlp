# Off the Planckian Locus: Using 2D Chromaticity to Improve In-Camera Color

**[SaiKiran Tedla](https://tedlasai.github.io/), Joshua E. Little,
[Hakki Can Karaimer](https://karaimer.github.io/),
[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)**

**ECCV 2026**

### [Paper (arXiv:2511.17133)](https://arxiv.org/abs/2511.17133)
### [Project Page](https://ccmmlp.github.io/)
### [Data](https://ln5.sync.com/dl/e8a199110#rmjujbfw-pn5n9qma-xf8p6stj-agizgb7u)
### [Checkpoints](checkpoints/)

---

## Getting Started

This repo learns a **colour space transform (CST)** conditioned on the illuminant
whitepoint, and benchmarks it against the analytic 2-CST and 3-CST interpolations,
nearest-neighbour, CCCNN and EXPINV baselines, and a per-image oracle.

### Environment Setup

Installing the base environment just needs:

```bash
pip install -e .
```

To recreate the exact environment used for this paper, use:

```bash
conda env create -f setup/environment.yml
conda activate cst
pip install -e .
```

### On tinycudann

The paper's models were trained with
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) fused kernels. It is a CUDA
source build, not a pip package.

This code supports training new models using just PyTorch, but the provided
checkpoints require tinycudann to run.

* To **reproduce the paper's numbers from the released weights**, build tinycudann
  (`setup/install_tinycudann.sh`) and set `use_tinycudann: true`.

---

## Dataset

### [Download the data](https://ln5.sync.com/dl/e8a199110#rmjujbfw-pn5n9qma-xf8p6stj-agizgb7u)

Unpacks to a folder containing four things:

```text
OffthePlanckianLocus/
├── lightbox/                       # raw lightbox captures
│   └── {Canon,Sony,Pixel,Samsung}/
│       ├── dng_renamed/            # <- what the processor reads
│       └── jpg_renamed/
├── InTheWild/
│   └── {pixel,samsung}/
│       ├── raw/                    # the captures (.dng)
│       ├── corners/                # chart corners, one .npy per capture
│       ├── wildrawnormalized/      # pre-extracted, ready to use
│       ├── wildpatches/
│       ├── images_png/
│       └── patches_png/
├── processed/                      # optional shortcut, see below
│   ├── LightboxCaptures/
│   │   ├── colorchecker24/{Canon,Sony,Pixel,Samsung}/patches/
│   │   └── splits/
│   └── LSMI/
│       ├── splits/
│       └── converted_dngs/
└── camera_spectral_sensitivites/   # <camera>.txt spectral response curves
```

### Telling the code where your data lives

All paths route through **`training/configs/paths.yaml`**. Edit that file once, or
override per-run with environment variables:

```bash
export CC_RAW_DATA_ROOT=/path/to/OffthePlanckianLocus
export CC_SIM_ROOT=/path/to/OffthePlanckianLocus/camera_spectral_sensitivites
export CC_DATA_ROOT=/path/to/where/processed/output/should/go
```

| Root | Holds |
|---|---|
| `raw_data_root` | the unpacked download: raw captures, corners, spectral curves |
| `data_root` | processed patches, splits, colour targets |
| `sim_root` | spectrally simulated training data |
| `output_root` | checkpoints, per-image `.npy` dumps, metrics CSVs |

### Skip processing, or do it yourself

The download includes a `processed/` folder with the patches and splits already
extracted. To use them and go straight to training:

```bash
export CC_DATA_ROOT=/path/to/OffthePlanckianLocus/processed
```

To regenerate them from the raw DNGs instead, point `CC_DATA_ROOT` somewhere
writable and run:

```bash
# per camera: dng_renamed/ -> patches under $CC_DATA_ROOT
python data_processing/lightbox/data_processor.py sony
```

The **splits ship with the download** — `dirchlet.yaml`, `hoang.yaml` and
`dirchletplushoang.yaml`, covering the 834 illuminants (7–407 Dirichlet, 407–797
near-daylight, 797–834 temperature steps). `dirchletplushoang` is the split behind
the headline results; using the shipped files is what makes your numbers comparable
to ours.

After processing, `data_root` holds:

```text
${data_root}/LightboxCaptures/
├── colorchecker24/{Canon,Sony,Pixel,Samsung}/patches/
└── splits/{dirchlet,hoang,dirchletplushoang}.yaml
```

**In-the-wild** ships pre-extracted as `wildpatches/` + `wildrawnormalized/`, which
is what the wild split reads, so no processing is required. To re-extract from the
raw captures instead:

```bash
python data_processing/in_the_wild/patch_processing.py samsung
```

That reads `InTheWild/<camera>/raw/*.dng` together with `InTheWild/<camera>/corners/`,
which holds the four chart corner coordinates for each capture. These were measured
per image — the captures are handheld, so the chart moves — and are what make the
wild patches reproducible rather than take-it-or-leave-it.

The lightbox chart corners are not per-image: the rig is fixed, so a single
normalised pair per camera lives in the camera configs as `corner1`/`corner2`.

### Simulated training data

`CCCNN*` and `EXPINV*` models train on spectrally simulated colours rather than
captures. Generate them from the downloaded sensitivity curves:

```bash
python data_processing/simulating_colors/cccnn_data_builder_3to3.py
```

This writes `<camera>_simulated.npy` and `xyz_simulated.npy` next to the `.txt`
files in `sim_root`.

---

## Training and Evaluation

`training/runner.py` does both. It takes any number of config files and merges them
left to right, in the order **camera to model to experiment**:

```bash
python training/runner.py --config \
  training/configs/cameras/sony.yaml \
  training/configs/models/MLP2DXY.yaml \
  training/configs/experiments/all_lightbox_experiments.yaml
```

Set `train: false` in the experiment config to load an existing checkpoint and
evaluate only. Evaluation writes per-image `.npy` dumps under `output_root`.

To run a whole sweep (the cartesian product of cameras and models for a named
experiment):

```bash
python training/build_jobs.py lightbox
```

### Results

`runner.py` reports mean angular error, ΔE00 and percentiles for each split as it
evaluates, and writes a per-image `.npy` record under `output_root` holding the
corrected output, the target, the white point and the per-pixel error maps. Aggregate
those however suits you — we deliberately ship no table or plot tooling.

**`extra/sample_metrics/` holds the compiled metrics from the paper's runs** — one
row per method with angular error, ΔE and ΔC (mean plus percentiles), model size,
MACs and runtime. That is the reference to compare against, and it needs no download.

---

## Multi-illuminant LSMI

Our multi-illuminant handling of the
[LSMI dataset](https://github.com/DY112/LSMI-dataset) is part of this paper's
contribution. It needs more setup than the lightbox path: the upstream LSMI
preprocessing has to be run first, and a reference DNG is required for camera
metadata.

Run the **upstream** LSMI preprocessing first (`0_cvt2tiff.py`, then
`1_make_mixture_map.py`, then `2_preprocess_data.py`); we do not redistribute it.
Our processor consumes its output rather than the raw captures directly, expecting
each scene as `<camera>/<Place>/` holding `<Place>_{1,2,12}.tiff`, the `<Place>_12.npy`
mixture map and the original raw, plus a `meta.json` at the camera root.

**It also needs a reference DNG** at
`${data_root}/LSMI/converted_dngs/<camera>_converted.dng`. TIFFs carry no camera
metadata, so the pipeline borrows colour matrices and black/white levels from a DNG
of the same camera. These ship with the dataset download; without one, processing
fails immediately with a `FileNotFoundError`.

Then:

```bash
python data_processing/lsmi/patch_processing.py sony    # -> patches/ rawnormalized/ raw/ mixed/
python data_processing/lsmi/crop_rawnormalized.py sony  # -> cropped/ (256x256)

python training/build_jobs.py lsmi                      # configs/lsmi_cameras/ x models
```

**`mixed/` is the multi-illuminant set** — the contribution — read by the `mixed`
split in `utilities/dataset.py` via each camera config's `mixed_path`.

`mixed` records are ~330 MB each, so a full camera needs a lot of disk. Pass
`--limit N` to `patch_processing.py` to process only the first N scenes while you
are checking the setup.

Note we resize to 256x256 and renormalise by camera bit depth (galaxy by 1023,
nikon/sony by 16383), which matters if you compare our LSMI numbers with theirs.

---

## Paper naming vs code naming

The paper says **CCM** (colour correction matrix); the code says **CST** (colour space
transform). They are the same thing. When matching a table to a config:

| Paper | Code |
|---|---|
| 2-CCM | `models/2cst.yaml` |
| 3-CCM | `models/3cst.yaml` |
| 7-CCM | `models/7cst.yaml` |
| 7-RBF | `models/RBF2DXY_linear_rbf.yaml`, `RBF2DXY_tps.yaml` |
| NN | `models/NN{1DCCT,2DXY,2DWP}.yaml` |
| CCCNN | `models/CCCNN{1DCCT,2DXY,2DWP}.yaml` |
| EXPINV | `models/EXPINV{1DCCT,2DXY,2DWP}.yaml` |
| **CCM-MLP** (ours) | `models/MLP{1DCCT,2DXY,2DWP}.yaml` |
| CCM-MLP (RP), root-polynomial 3x6 | `ablations/polynomial_models/*RootPoly2.yaml`, or `ablations/poly_override/rootpoly2.yaml` |
| Oracle | `models/bestfitsolver.yaml` |
| LUT | `ablations/lut_models/*.yaml` |

The coordinate suffixes map to the paper's input-coordinate ablation (Tables 2, S3):
`1DCCT` = 1D CCT, `2DWP` = 2D raw, `2DXY` = 2D xy.



---

## Contact

Questions and issues: [jelittle@yorku.ca](mailto:jelittle@yorku.ca).

---

## Citation

If you use our dataset, code, or model, please cite:

```bibtex
@inproceedings{tedla2026planckian,
  title     = {Off the Planckian Locus: Using 2D Chromaticity to Improve In-Camera Color},
  author    = {Tedla, SaiKiran and Little, Joshua E. and Karaimer, Hakki Can and Brown, Michael S.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
