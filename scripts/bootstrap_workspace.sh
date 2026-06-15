#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bootstrap_workspace.sh [root_dir]

If root_dir is omitted, the script uses the repository root
based on the location of this script.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
root="${1:-$repo_root}"
mkdir -p "$root"
root="$(cd -- "$root" && pwd)"

dirs=(
  "$root/data"
  "$root/data/cityscapes"
  "$root/data/cityscapes/leftImg8bit/train"
  "$root/data/cityscapes/leftImg8bit/val"
  "$root/data/cityscapes/leftImg8bit/test"
  "$root/data/cityscapes/gtFine/train"
  "$root/data/cityscapes/gtFine/val"
  "$root/data/cityscapes/gtFine/test"
  "$root/data/pretrained_models"
  "$root/data/trained_models"
  "$root/results"
  "$root/results/boundary"
  "$root/results/boundary/viz"
  "$root/results/clean_vis"
  "$root/results/dataset_browse"
  "$root/results/learning_curves"
  "$root/results/plots"
  "$root/results/cityscapes"
  "$root/results/cityscapes/1024x1024"
  "$root/results/cityscapes/1024x1024/vis"
  "$root/work_dirs"
  "$root/runpod_cmd"
)

for dir in "${dirs[@]}"; do
  mkdir -p "$dir"
done

cat <<EOF
Workspace tree ready at: $root
Created/verified directories:
  - data/cityscapes/{leftImg8bit,gtFine}/{train,val,test}
  - data/pretrained_models
  - data/trained_models
  - results/{boundary,clean_vis,dataset_browse,learning_curves,plots,cityscapes/1024x1024/vis}
  - work_dirs
  - runpod_cmd
EOF
