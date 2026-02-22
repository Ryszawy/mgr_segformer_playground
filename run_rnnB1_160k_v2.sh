#!/bin/bash
set -e

PROJECT_ROOT="/workspace/mgr_segformer_playground"
MMSEG_ROOT="$PROJECT_ROOT/mmsegmentation"
RESULTS_DIR="$PROJECT_ROOT/results/cityscapes/1024x1024/b0/metrics"
WORKDIR="$MMSEG_ROOT/work_dirs/b0_rnnB1_160k_v2"
CFG="$MMSEG_ROOT/configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py"

mkdir -p "$RESULTS_DIR"
mkdir -p "$WORKDIR"

cd "$MMSEG_ROOT"

# TRAIN
python tools/train.py \
  configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py \
  --work-dir work_dirs/b0_rnnB1_160k_v2 \
  --cfg-options default_hooks.checkpoint.interval=40000

# TEST (użyj zawsze last_checkpoint)
CKPT=$(cat work_dirs/b0_rnnB1_160k_v2/last_checkpoint)

python tools/test.py \
  configs/segformer_rnn/segformer_mit-b0_rnnB1_160k_cityscapes-1024x1024.py \
  "$CKPT" \
  | tee work_dirs/b0_rnnB1_160k_v2/test_last.log

# PARSE
cd "$PROJECT_ROOT"
python scripts/parse_mmseg_log.py \
  mmsegmentation/work_dirs/b0_rnnB1_160k_v2/test_last.log \
  $RESULTS_DIR/b0_rnnB1_160k_v2.json \
  $RESULTS_DIR/b0_rnnB1_160k_v2.csv

echo "DONE"
