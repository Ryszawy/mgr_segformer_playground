# Project Backbone

This file lists the minimum files and directories needed to reproduce the thesis workspace without packaging the entire repository.

## License

- `mmsegmentation/LICENSE` - Apache 2.0 license for the vendored MMSegmentation code.

If you redistribute the vendored `mmsegmentation/` tree, keep this license file with it.

## Minimum files for training and testing

### Root

- `README.md`
- `README_SETUP.md`
- `todo.md`
- `how_to_acces_ssh.txt` - reference only, do not edit

### Bootstrap and helper scripts

- `scripts/bootstrap_workspace.sh`
- `scripts/bootstrap_workspace.bat`
- `scripts/test_cityscapes.sh`
- `scripts/vis_cityscapes_configs.sh`

### Core analysis scripts

- `scripts/param_count.py`
- `scripts/parse_mmseg_log.py`
- `scripts/plot_learning_curves.py`
- `scripts/plot_learning_curves_v2.py`
- `scripts/plot_params_and_tradeoff.py`
- `scripts/boundary_eval.py`
- `scripts/boundary_viz.py`
- `scripts/compare_gt_pred_clean.py`
- `scripts/keep_first_n_images.py`
- `scripts/torchinfo_summary_mmseg_head.py`

### Optional legacy plotting helpers

- `scripts/plot_from_hardcoded_results.py`
- `scripts/plot_miou_vs_params.py`
- `scripts/plot_miou_vs_params_clean.py`

## MMSegmentation files required by the thesis

### Upstream entrypoints that must remain available

- `mmsegmentation/tools/train.py`
- `mmsegmentation/tools/test.py`
- `mmsegmentation/tools/misc/browse_dataset.py`

### Custom decode heads

- `mmsegmentation/mmseg/models/decode_heads/segformer_conv_head.py`
- `mmsegmentation/mmseg/models/decode_heads/segformer_rnn_head.py`
- `mmsegmentation/mmseg/models/decode_heads/segformer_gated_head.py`
- `mmsegmentation/mmseg/models/decode_heads/__init__.py`

### Baseline SegFormer configs

- `mmsegmentation/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py`

### Cityscapes base configs

- `mmsegmentation/configs/_base_/datasets/cityscapes.py`
- `mmsegmentation/configs/_base_/datasets/cityscapes_1024x1024.py`
- `mmsegmentation/configs/_base_/models/segformer_mit-b0.py`

### Custom thesis configs

- `mmsegmentation/configs/segformer_exp/segformer_mit-b0_convhead_cityscapes.py`
- `mmsegmentation/configs/segformer_exp/segformer_mit-b0_convhead_20k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_exp/segformer_mit-b0_convhead_40k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_exp/segformer_mit-b0_convhead_80k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_rnn/segformer_mit-b0_rnnB1_20k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_rnn/segformer_mit-b0_rnnB1_40k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_rnn/segformer_mit-b0_rnnB1_80k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b0_gated_20k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b0_gated_40k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b0_gated_80k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b2_gated_160k_cityscapes-1024x1024.py`
- `mmsegmentation/configs/segformer_gate/segformer_mit-b3_gated_160k_cityscapes-1024x1024.py`

## Results and manifests that help reproduce the thesis figures

- `results/boundary/models_list.txt`
- `results/param_counts.csv`
- `results/boundary/boundary_metrics.csv`
- `results/boundary/boundary_tables.tex`
- `results/learning_curves/`
- `results/plots/`
- `results/clean_vis/`

## Shared RunPod layout notes

- `runpod_cmd/README.md`
- `runpod_cmd/segformer_data.inside.txt`
- `runpod_cmd/mgr_segformer_data_store.inside.txt`

## Recommended copy set if you want a lightweight portable archive

If you only need the runnable thesis backbone, keep:

- `README.md`
- `README_SETUP.md`
- `todo.md`
- `scripts/`
- `mmsegmentation/LICENSE`
- `mmsegmentation/tools/train.py`
- `mmsegmentation/tools/test.py`
- `mmsegmentation/tools/misc/browse_dataset.py`
- `mmsegmentation/configs/_base_/datasets/`
- `mmsegmentation/configs/_base_/models/`
- `mmsegmentation/configs/segformer/`
- `mmsegmentation/configs/segformer_exp/`
- `mmsegmentation/configs/segformer_rnn/`
- `mmsegmentation/configs/segformer_gate/`
- `mmsegmentation/mmseg/models/decode_heads/`
- `results/boundary/models_list.txt`
- `runpod_cmd/`

For a fully reproducible thesis archive, also keep `results/` outputs and the raw `work_dirs/` checkpoints you care about.
