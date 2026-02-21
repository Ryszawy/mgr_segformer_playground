#!/bin/bash
set -e  # przerwij jeśli coś się wysypie

PROJECT_ROOT="/workspace/mgr_segformer_playground"
MMSEG_ROOT="$PROJECT_ROOT/mmsegmentation"
RESULTS_DIR="$PROJECT_ROOT/results/cityscapes/1024x1024/b0/metrics"

mkdir -p "$RESULTS_DIR"

echo "=============================="
echo "Starting ConvHead_v2 runs..."
echo "=============================="

########################################
# 40k
########################################
echo "---- TRAIN 40k ----"
cd $MMSEG_ROOT
python tools/train.py \
  configs/segformer_exp/segformer_mit-b0_convhead_40k_cityscapes-1024x1024.py \
  --work-dir work_dirs/b0_convhead_40k_v2 \
  --cfg-options default_hooks.checkpoint.interval=10000

echo "---- TEST 40k ----"
python tools/test.py \
  configs/segformer_exp/segformer_mit-b0_convhead_40k_cityscapes-1024x1024.py \
  work_dirs/b0_convhead_40k_v2/iter_40000.pth \
  | tee work_dirs/b0_convhead_40k_v2/test_40k.log

echo "---- PARSE 40k ----"
cd $PROJECT_ROOT
python scripts/parse_mmseg_log.py \
  mmsegmentation/work_dirs/b0_convhead_40k_v2/test_40k.log \
  $RESULTS_DIR/b0_convhead_40k_v2.json \
  $RESULTS_DIR/b0_convhead_40k_v2.csv

########################################
# 80k
########################################
echo "---- TRAIN 80k ----"
cd $MMSEG_ROOT
python tools/train.py \
  configs/segformer_exp/segformer_mit-b0_convhead_80k_cityscapes-1024x1024.py \
  --work-dir work_dirs/b0_convhead_80k_v2 \
  --cfg-options default_hooks.checkpoint.interval=20000

echo "---- TEST 80k ----"
python tools/test.py \
  configs/segformer_exp/segformer_mit-b0_convhead_80k_cityscapes-1024x1024.py \
  work_dirs/b0_convhead_80k_v2/iter_80000.pth \
  | tee work_dirs/b0_convhead_80k_v2/test_80k.log

echo "---- PARSE 80k ----"
cd $PROJECT_ROOT
python scripts/parse_mmseg_log.py \
  mmsegmentation/work_dirs/b0_convhead_80k_v2/test_80k.log \
  $RESULTS_DIR/b0_convhead_80k_v2.json \
  $RESULTS_DIR/b0_convhead_80k_v2.csv

########################################
# 160k
########################################
echo "---- TRAIN 160k ----"
cd $MMSEG_ROOT
python tools/train.py \
  configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py \
  --work-dir work_dirs/b0_convhead_160k_v2 \
  --cfg-options default_hooks.checkpoint.interval=40000

echo "---- TEST 160k ----"
python tools/test.py \
  configs/segformer_exp/segformer_mit-b0_convhead_160k_cityscapes-1024x1024.py \
  work_dirs/b0_convhead_160k_v2/iter_160000.pth \
  | tee work_dirs/b0_convhead_160k_v2/test_160k.log

echo "---- PARSE 160k ----"
cd $PROJECT_ROOT
python scripts/parse_mmseg_log.py \
  mmsegmentation/work_dirs/b0_convhead_160k_v2/test_160k.log \
  $RESULTS_DIR/b0_convhead_160k_v2.json \
  $RESULTS_DIR/b0_convhead_160k_v2.csv

echo "=============================="
echo "All ConvHead_v2 runs finished!"
echo "=============================="
