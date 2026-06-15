# README_SETUP

This is the operational runbook for the SegFormer thesis workspace.
It explains how to rebuild the environment, where the custom code lives, how to train and validate the thesis heads, and how to regenerate the figures and tables already stored under `results/`.

## 1. What This Repo Contains

This repository is a thesis workspace built around a vendored copy of `mmsegmentation/`.
The local fork adds:

- custom SegFormer heads,
- custom configs for Cityscapes,
- RunPod helper notes,
- analysis scripts for logs, parameter counts, boundary quality, and plots,
- pre-generated outputs under `results/`.

The project history, in rough order, is:

- initial clone of OpenMMLab MMSegmentation,
- baseline setup scripts for Cityscapes evaluation,
- Head A, the convolutional fusion head,
- Head C, the gated fusion head,
- Head B, the GRU/LSTM-style head,
- boundary analysis and visual inspection scripts,
- final summary / torchinfo tooling for head inspection,
- result plotting and trade-off figures.

If you want the short planning checklist behind the project, see `todo.md`.

## 2. Authoritative References

These are the official sources that match the setup used here:

- Cityscapes home: https://www.cityscapes-dataset.com/
- Cityscapes login/download page: https://www.cityscapes-dataset.com/downloads/
- MMSegmentation dataset preparation guide: https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html

For the SSH workflow to the RunPod instance, read `how_to_acces_ssh.txt` as a reference only. Do not edit it.

## 3. Repo Layout

- `mmsegmentation/` - vendored MMSegmentation clone with the custom code.
- `scripts/` - helper scripts for testing, metrics, plots, and visual inspection.
- `results/` - generated CSV, PNG, PDF, LaTeX, and visualization outputs.
- `runpod_cmd/` - shared notes and files exchanged with RunPod.
- `docs/` - thesis material and chat logs.
- `todo.md` - historical work plan and experiment outline.

## 4. Environment Setup

The current workspace was used on RunPod, with the repo mounted under:

```text
/workspace/mgr_segformer_playground
```

If your clone lives elsewhere, adjust the hardcoded `ROOT=` values in:

- `scripts/test_cityscapes.sh`
- `scripts/vis_cityscapes_configs.sh`

To prepare the expected directory tree on Linux/RunPod or on Windows, use:

```bash
./scripts/bootstrap_workspace.sh
```

or on Windows:

```bat
scripts\bootstrap_workspace.bat
```

Both scripts accept an optional target root directory, so you can also run them against a fresh clone or a separate local checkout.
They create the same baseline tree for:

- `data/cityscapes/`
- `data/pretrained_models/`
- `data/trained_models/`
- `results/{boundary,clean_vis,dataset_browse,learning_curves,plots,cityscapes/1024x1024/vis}`
- `work_dirs/`
- `runpod_cmd/`

### 4.1 Clone the workspace

If you are rebuilding from scratch, clone this thesis repo first:

```bash
git clone <this-repo-url> mgr_segformer_playground
cd mgr_segformer_playground
```

The vendored MMSegmentation code is already present under `mmsegmentation/`.
If you prefer a clean upstream checkout, clone OpenMMLab MMSegmentation into that directory and then copy the thesis-specific files into place.

### 4.2 Create the Python environment

Use Python 3.10 on a CUDA-enabled machine.
Then create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4.3 Install OpenMMLab dependencies

Install the OpenMMLab stack first, then the repo extras:

```bash
pip install -U openmim
mim install "mmengine>=0.5.0,<1.0.0" "mmcv>=2.0.0rc4,<2.2.0"
cd mmsegmentation
pip install -e .
pip install -r requirements/runtime.txt
pip install cityscapesscripts pandas opencv-python torchinfo
```

Notes:

- `requirements/runtime.txt` covers the core plotting/runtime stack used by the thesis scripts.
- `cityscapesscripts` is needed for Cityscapes support.
- `pandas` is needed by the learning-curve scripts.
- `opencv-python` is needed by the boundary and overlay scripts.
- `torchinfo` is needed by `scripts/torchinfo_summary_mmseg_head.py`.
- If your local package set is already prepared, this repo also has `mmsegmentation/requirements.txt` with the upstream runtime/optional/test extras.

### 4.4 Sanity checks

From inside `mmsegmentation/`, verify that the local package imports work:

```bash
python -c "from mmseg.utils import register_all_modules; register_all_modules(); print('mmseg OK')"
python tools/train.py --help
python tools/test.py --help
```

If those commands run, the environment is ready for training and evaluation.

## 5. Cityscapes Dataset

The thesis uses the Cityscapes fine annotations for train and val.
The official dataset download requires an account login on the Cityscapes website.

### 5.1 Download and unpack

After logging in on the official download page, download the Cityscapes archives used by MMSegmentation, then unpack them into:

```text
data/cityscapes/
```

The expected structure is:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/          # optional, only if you also download test images
└── gtFine/
    ├── train/
    ├── val/
    └── test/          # optional, labels for test are not used in this thesis
```

The MMSeg base config used in this project expects:

- images in `leftImg8bit/train` and `leftImg8bit/val`,
- masks in `gtFine/train` and `gtFine/val`,
- `data_root = 'data/cityscapes/'`.

### 5.2 If the dataset lives elsewhere

The simplest option is to symlink your dataset root to `data/cityscapes`.
If you prefer a different layout, update the base Cityscapes config:

- `mmsegmentation/configs/_base_/datasets/cityscapes.py`
- `mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py`

### 5.3 Verify the dataset

Good quick checks:

```bash
cd mmsegmentation
python tools/misc/browse_dataset.py \
  configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py \
  --not-show \
  --output-dir ../results/dataset_browse/cityscapes_b0
```

You can also run a small test or visual overlay after training a checkpoint to confirm that masks are aligned correctly.

## 6. Models, Heads, and Config Naming

The custom decode heads are registered in:

- `mmsegmentation/mmseg/models/decode_heads/segformer_conv_head.py`
- `mmsegmentation/mmseg/models/decode_heads/segformer_rnn_head.py`
- `mmsegmentation/mmseg/models/decode_heads/segformer_gated_head.py`

They are imported via:

- `mmsegmentation/mmseg/models/decode_heads/__init__.py`

The config `type=` field selects the head class directly, so no extra `custom_imports` are needed for the thesis heads.

### 6.1 Naming convention

Most thesis configs follow this pattern:

```text
segformer_<variant>_<head>_<iters>_cityscapes-1024x1024.py
```

The matching checkpoint is usually:

```text
work_dirs/<work_dir_name>/iter_<iters>.pth
```

### 6.2 Baseline and custom families

| Family | Config example | Head class | Typical work dir |
| --- | --- | --- | --- |
| B0 baseline | `configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py` | `SegformerHead` | `work_dirs/b0_baseline_160k` |
| B0 Conv | `configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py` | `SegFormerConvHead` | `work_dirs/b0_convhead_160k_v2` |
| B0 RNN | `configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py` | `SegformerRNNHead` | `work_dirs/b0_rnnB1_160k_v2` |
| B0 Gated | `configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py` | `SegformerGatedHead` | `work_dirs/b0_gated_160k` |
| B2 baseline | `configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py` | `SegformerHead` | `work_dirs/b2_baseline_160k` |
| B2 Gated | `configs/segformer_gate/segformer_mit-b2_gated_160k_cityscapes-1024x1024.py` | `SegformerGatedHead` | `work_dirs/b2_gated_160k` |
| B3 baseline | `configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py` | `SegformerHead` | `work_dirs/b3_baseline_160k` |
| B3 Gated | `configs/segformer_gate/segformer_mit-b3_gated_160k_cityscapes-1024x1024.py` | `SegformerGatedHead` | `work_dirs/b3_gated_160k` |

### 6.3 What each custom head does

- `SegFormerConvHead` replaces the final MLP fusion with `1x1 conv -> depthwise 3x3 conv -> 1x1 conv`.
- `SegformerRNNHead` projects features, fuses them, downsamples spatially, flattens to a sequence, runs `GRU` or `LSTM`, then reshapes back before classification.
- `SegformerGatedHead` applies a learned per-scale gate from global pooled features before concatenation.

### 6.4 Schedule variants

The thesis configs include 20k, 40k, 80k, and 160k variants for the custom heads.
Use the shorter schedules for quick ablations and the 160k versions for the final comparison.

## 7. Training

Run training from inside `mmsegmentation/`.
The configs already set the `work_dir` and scheduler, so the default `tools/train.py` invocation is usually enough.

### 7.1 Baseline training

```bash
cd /workspace/mgr_segformer_playground/mmsegmentation
python tools/train.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py
```

### 7.2 Custom head training

```bash
cd /workspace/mgr_segformer_playground/mmsegmentation
python tools/train.py configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer_gate/segformer_mit-b2_gated_160k_cityscapes-1024x1024.py
python tools/train.py configs/segformer_gate/segformer_mit-b3_gated_160k_cityscapes-1024x1024.py
```

For quick ablations, use the 20k, 40k, or 80k variants instead of the 160k config.

### 7.3 Checkpoints and resumes

After training, the checkpoint usually appears as:

```text
work_dirs/<run_name>/iter_<max_iters>.pth
```

If you resume a stopped run, use the standard MMSegmentation resume flow from the same `work_dir`.

## 8. Validation, Testing, and Visual Checks

### 8.1 Direct MMSeg testing

From `mmsegmentation/`, test a checkpoint with:

```bash
python tools/test.py configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py work_dirs/b0_gated_160k/iter_160000.pth
```

To save visual overlays during testing, add `--show-dir`:

```bash
python tools/test.py configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py work_dirs/b0_gated_160k/iter_160000.pth --show-dir ../results/cityscapes/1024x1024/vis/B0_Gated_160k
```

### 8.2 Wrapper for baseline cityscapes tests

From the repo root:

```bash
./scripts/test_cityscapes.sh b0 1024x1024
./scripts/test_cityscapes.sh b2 1024x1024 --vis 8
```

Important:

- the wrapper looks for official pretrained checkpoints under `data/trained_models/`,
- the script currently only changes the output path name with `RES`; the config itself stays the 1024x1024 Cityscapes config,
- if your repo lives somewhere else, update the `ROOT=` variable in the script.

### 8.3 Batch visualizations

From the repo root:

```bash
./scripts/vis_cityscapes_configs.sh --n 12
```

This produces visual overlays for the baseline and custom heads listed inside the script and writes them into `results/cityscapes/1024x1024/vis/`.

### 8.4 GT vs prediction overlays

From `mmsegmentation/`:

```bash
python ../scripts/compare_gt_pred_clean.py \
  --cfg configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py \
  --ckpt work_dirs/b0_gated_160k/iter_160000.pth \
  --outdir ../results/clean_vis/b0_gated_160k \
  --n 12
```

This saves side-by-side GT/prediction overlays for quick qualitative inspection.

## 9. Metrics, Logs, and Analysis

The repo stores the most important analysis outputs under `results/`.
The scripts below regenerate them.

### 9.1 Parameter counts

From `mmsegmentation/`:

```bash
python ../scripts/param_count.py configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py ../results/param_counts.csv
python ../scripts/param_count.py configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py ../results/param_counts.csv
python ../scripts/param_count.py configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py ../results/param_counts.csv
python ../scripts/param_count.py configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py ../results/param_counts.csv
```

The CSV is appended row by row.
For the Conv head, extra layer columns such as `fuse_conv1`, `fuse_dwconv`, and `fuse_conv2` are also added when present.

### 9.2 Log parsing

`scripts/parse_mmseg_log.py` converts a raw MMSeg/MMEngine log into JSON and CSV summaries:

```bash
python ../scripts/parse_mmseg_log.py <log_file> <out.json> <out.csv>
```

The parser extracts:

- final `aAcc`, `mIoU`, and `mAcc`,
- per-iteration timing,
- estimated FPS,
- peak logged memory,
- per-class IoU values when present.

### 9.3 Boundary analysis

The canonical input list is `results/boundary/models_list.txt`.
From `mmsegmentation/`:

```bash
python ../scripts/boundary_eval.py \
  --models ../results/boundary/models_list.txt \
  --out_csv ../results/boundary/boundary_metrics.csv \
  --out_latex ../results/boundary/boundary_tables.tex
```

You can also create boundary overlays for a few images:

```bash
python ../scripts/boundary_viz.py \
  --cfg configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py \
  --ckpt work_dirs/b0_gated_160k/iter_160000.pth \
  --name b0_gated_160k \
  --k 12
```

This writes images into `results/boundary/viz/<name>/`.

### 9.4 Learning curves

Single run:

```bash
python ../scripts/plot_learning_curves.py \
  --work-dir work_dirs/b0_gated_160k \
  --out-dir ../results/learning_curves/b0_gated_160k \
  --prefer-json
```

Multi-run comparison:

```bash
python ../scripts/plot_learning_curves_v2.py \
  --work-dir work_dirs/b0_baseline_160k \
  --work-dir work_dirs/b0_convhead_160k_v2 \
  --work-dir work_dirs/b0_rnnB1_160k_v2 \
  --work-dir work_dirs/b0_gated_160k \
  --mode lines \
  --name b0_compare_val_miou \
  --results-root ../results/learning_curves
```

For a mean ± std summary, use `--mode meanstd`.

### 9.5 Trade-off plots

Preferred current generator:

```bash
python ../scripts/plot_params_and_tradeoff.py
```

This regenerates:

- `results/plots/miou_vs_model_params.*`
- `results/plots/miou_vs_head_params.*`
- `results/plots/miou_vs_fps_tradeoff.*`

There are also legacy/static plotting helpers:

- `scripts/plot_miou_vs_params.py`
- `scripts/plot_miou_vs_params_clean.py`
- `scripts/plot_from_hardcoded_results.py`

Those are useful for reproducing exact thesis figures, but they are more brittle because they use hardcoded values.

### 9.6 Head inspection

The final inspection utility is:

```bash
python ../scripts/torchinfo_summary_mmseg_head.py \
  --cfg configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py \
  --part head \
  --hw 1024 1024 \
  --mit b0 \
  --details
```

Use `--part model` if you want a full model summary instead of only the decode head.

## 10. Reproducing `results/`

The main generated artifacts in this repo are:

- `results/param_counts.csv` - parameter summary table,
- `results/plots/` - trade-off and parameter plots,
- `results/learning_curves/` - single-run and multi-run learning curves,
- `results/boundary/` - boundary metrics, LaTeX tables, and boundary visualizations,
- `results/clean_vis/` - GT vs prediction overlays.

A practical regeneration order is:

1. Train the desired configs.
2. Run `param_count.py` for every config you want in the summary table.
3. Run `boundary_eval.py` with `results/boundary/models_list.txt`.
4. Generate visual overlays with `compare_gt_pred_clean.py` and `boundary_viz.py`.
5. Build the curve plots with `plot_learning_curves.py` or `plot_learning_curves_v2.py`.
6. Rebuild the trade-off plots with `plot_params_and_tradeoff.py`.

## 11. Script Guide and Improvement Notes

### Training and testing wrappers

- `scripts/test_cityscapes.sh`
  - Wrapper around `tools/test.py` for the baseline SegFormer models.
  - Improvement: add a real `--help`, allow a configurable repo root, and make the resolution override actually change the test config instead of only the output folder name.
- `scripts/vis_cityscapes_configs.sh`
  - Batch visualizer for the main baseline/custom-head runs.
  - Improvement: replace the hardcoded `RUNS` array with CLI arguments or a manifest file.

### Metrics and log parsers

- `scripts/param_count.py`
  - Counts model and decode-head parameters and appends them to a CSV.
  - Improvement: add `argparse`, validate the CSV schema before appending, and optionally emit a separate file per experiment family.
- `scripts/parse_mmseg_log.py`
  - Parses a raw MMSeg log into JSON and CSV metrics.
  - Improvement: add `argparse` and support more than one log format explicitly instead of relying on positional arguments only.
- `scripts/boundary_eval.py`
  - Computes Boundary F1 and Boundary IoU.
  - Improvement: the current CLI is good, but the README should keep `results/boundary/models_list.txt` as the canonical example input.
- `scripts/boundary_viz.py`
  - Saves boundary overlays for a few samples.
  - Improvement: add an explicit output-dir option and maybe a small manifest loader for model lists.
- `scripts/compare_gt_pred_clean.py`
  - Saves side-by-side GT and prediction overlays.
  - Improvement: add optional class filtering or a built-in montage layout for easier batch review.
- `scripts/torchinfo_summary_mmseg_head.py`
  - Prints a torchinfo summary plus per-layer shape details for the head or the full model.
  - Improvement: auto-detect `in_channels` more aggressively for custom heads and add a clearer fallback when the shape assumptions do not match.

### Visualizers

- `scripts/keep_first_n_images.py`
  - Utility that deletes everything after the first `N` images in a folder.
  - Improvement: add `argparse` and a dry-run flag.

### Plotting helpers

- `scripts/plot_learning_curves.py`
  - Single-run curve plotting; already has `argparse`.
  - Improvement: keep the current CLI, but add a little more normalization for log-file discovery if you move runs around.
- `scripts/plot_learning_curves_v2.py`
  - Multi-run comparison plotting; already has `argparse`.
  - Improvement: expose more explicit defaults for the output directory and add a tiny schema check for the work-dir names.
- `scripts/plot_params_and_tradeoff.py`
  - Preferred current trade-off plot generator.
  - Improvement: read the score table from a CSV instead of hardcoding the `mIoU` and `FPS` dictionaries.
- `scripts/plot_miou_vs_params.py`
  - Legacy helper for a single mIoU-vs-params figure.
  - Improvement: update it to the current `results/param_counts.csv` schema; it currently expects an older `params_m` column name.
- `scripts/plot_miou_vs_params_clean.py`
  - Static, paper-style figure helper.
  - Improvement: either add CLI inputs or keep it only as a locked figure generator for the thesis draft.
- `scripts/plot_from_hardcoded_results.py`
  - Another hardcoded figure script for quick one-off charts.
  - Improvement: replace the hardcoded arrays with file-based inputs if you plan to reuse it.

## 12. Practical Notes

- All thesis head configs inherit the standard Cityscapes pipeline and the official pretrained SegFormer backbone initialization from the upstream MMSegmentation setup.
- Baseline SegFormer configs use sliding-window inference on Cityscapes 1024x1024 validation.
- The visual inspection scripts save outputs under `results/`, so it is easy to compare them against the stored figures in this repo.
- When you change config names or work-dir names, update the helper scripts and the `results/boundary/models_list.txt` manifest together so the analysis stays consistent.
