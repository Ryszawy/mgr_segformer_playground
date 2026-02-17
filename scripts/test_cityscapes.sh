#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/test_cityscapes.sh b0 1024x1024
#   ./scripts/test_cityscapes.sh b2 1024x1024 --vis 8
#
# Defaults: vis disabled
MODEL=${1:-}
RES=${2:-1024x1024}

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 {b0|b1|b2|b3} {1024x1024|768x768|640x1280|512x1024} [--vis N]"
  exit 1
fi

VIS_N=0
if [[ "${3:-}" == "--vis" ]]; then
  VIS_N=${4:-8}
fi

ROOT=/workspace/mgr_segformer_playground
REPO=$ROOT/mmsegmentation
DATA=$ROOT/data
MODELS=$DATA/trained_models

CFG="$REPO/configs/segformer/segformer_mit-${MODEL}_8xb1-160k_cityscapes-1024x1024.py"
CKPT=$(ls -1 "$MODELS"/segformer_mit-${MODEL}_8x1_1024x1024_160k_cityscapes_*.pth | head -n 1)

OUTDIR="$ROOT/results/cityscapes/${RES}/${MODEL}"
LOGDIR="$OUTDIR/logs"
METDIR="$OUTDIR/metrics"
VISDIR="$OUTDIR/vis"
mkdir -p "$LOGDIR" "$METDIR" "$VISDIR"

RUN_ID="segformer_${MODEL}_cityscapes_${RES}_$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/${RUN_ID}.log"

cd "$REPO"

# Opcje do modyfikacji rozdzielczości testu (na start zostawiamy tylko 1024x1024 = config default).
# Dla innych rozdzielczości dołożymy override poniżej (sekcja 4).
CFG_OPTS=()

# Peak VRAM i timing: włączamy cuda stats w pytorch + log końcowy wyciągniemy parserem
export CUDA_LAUNCH_BLOCKING=0

echo "CFG : $CFG" | tee "$LOG"
echo "CKPT: $CKPT" | tee -a "$LOG"
echo "OUT : $OUTDIR" | tee -a "$LOG"

# Wizualizacje (MMSeg zapisuje predykcje przez --show-dir)
SHOW_ARGS=()
if [[ "$VIS_N" -gt 0 ]]; then
  SHOW_ARGS=(--show-dir "$VISDIR")
fi

python tools/test.py "$CFG" "$CKPT" \
  "${SHOW_ARGS[@]}" \
  --cfg-options "${CFG_OPTS[@]}" \
  2>&1 | tee -a "$LOG"

# Parse metryki do JSON/CSV
python "$ROOT/scripts/parse_mmseg_log.py" "$LOG" "$METDIR/${RUN_ID}.json" "$METDIR/${RUN_ID}.csv"

# (Opcjonalnie) przytnij do N wizualizacji, żeby nie zalać dysku
if [[ "$VIS_N" -gt 0 ]]; then
  python "$ROOT/scripts/keep_first_n_images.py" "$VISDIR" "$VIS_N"
fi

echo "Done: $OUTDIR"
