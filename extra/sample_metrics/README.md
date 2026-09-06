# Sample metrics

`lightbox_compiled_metrics.csv` holds the test-split results for the eight
checkpoints in `checkpoints/`, on the lightbox `dirchletplushoang` split.

These were produced by evaluating the released weights themselves, with
tinycudann, over all 237 test illuminants — not copied from an earlier run. Every
number here is reproducible from this repository:

    python training/runner.py --config \
      training/configs/cameras/sony.yaml \
      training/configs/models/MLP2DXY.yaml \
      training/configs/experiments/all_lightbox_experiments.yaml

with `train: false` and `use_tinycudann: true`, then aggregating the per-image
`.npy` records that evaluation writes under `output_root`.

`angular_error` is stored per image; `delta_e` per chart patch; `delta_c` is
derived from the stored `output_xyz_scaled` and `target`. Percentile columns are
computed over the pooled values.

For the full method comparison against the 2-CCM/3-CCM/NN/CCCNN/EXPINV baselines,
see the tables in the paper.
