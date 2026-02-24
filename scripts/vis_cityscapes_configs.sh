#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/vis_cityscapes_configs.sh --n 12
#   ./scripts/vis_cityscapes_configs.sh --n 20 --device cuda:0
#
# Output:
#   results/cityscapes/1024x1024/vis/<NAME>/

N=12
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROOT=/workspace/mgr_segformer_playground
REPO=$ROOT/mmsegmentation
OUTROOT=$ROOT/results/cityscapes/1024x1024/vis

mkdir -p "$OUTROOT"

cd "$REPO"

# ---------------------------------------
# Define runs: name|config|checkpoint
# ---------------------------------------
RUNS=(
  "B0_Baseline_160k|configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py|work_dirs/b0_baseline_160k/iter_160000.pth"
  "B0_Conv_160k|configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py|work_dirs/b0_convhead_160k_v2/iter_160000.pth"
  "B0_RNN_160k|configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py|work_dirs/b0_rnnB1_160k_v2/iter_160000.pth"
  "B0_Gated_160k|configs/segformer_gate/segformer_mit-b0_gated_160k_cityscapes-1024x1024.py|work_dirs/b0_gated_160k/iter_160000.pth"

  "B2_Baseline_160k|configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py|work_dirs/b2_baseline_160k/iter_160000.pth"
  "B2_Gated_160k|configs/segformer_gate/segformer_mit-b2_gated_160k_cityscapes-1024x1024.py|work_dirs/b2_gated_160k/iter_160000.pth"

  "B3_Baseline_160k|configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py|work_dirs/b3_baseline_160k/iter_160000.pth"
  "B3_Gated_160k|configs/segformer_gate/segformer_mit-b3_gated_160k_cityscapes-1024x1024.py|work_dirs/b3_gated_160k/iter_160000.pth"
)

echo "Saving visualizations to: $OUTROOT"
echo "Keeping first N images: $N"
echo "Device: $DEVICE"
echo ""

for entry in "${RUNS[@]}"; do
  IFS="|" read -r NAME CFG CKPT <<< "$entry"

  VISDIR="$OUTROOT/$NAME"
  rm -rf "$VISDIR"
  mkdir -p "$VISDIR"

  echo "== $NAME =="
  echo "CFG : $CFG"
  echo "CKPT: $CKPT"
  echo "OUT : $VISDIR"

  # --show-dir writes overlay visualizations for the val set
  python tools/test.py "$CFG" "$CKPT" \
    --show-dir "$VISDIR" \
    --cfg-options model.test_cfg.mode=whole \
    2>/dev/null

  # keep first N images (optional but handy)
  python "$ROOT/scripts/keep_first_n_images.py" "$VISDIR" "$N" || true

  echo ""
done

echo "Done. Visualizations in: $OUTROOT"
