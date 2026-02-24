import argparse
import csv
import os
from pathlib import Path
import numpy as np
import cv2
import torch

from mmengine.config import Config
from mmseg.utils import register_all_modules
from mmseg.registry import DATASETS
from mmseg.apis import init_model, inference_model

# -----------------------------
# Boundary utilities
# -----------------------------
def mask_to_boundary(mask: np.ndarray, width: int = 3, ignore_index: int = 255) -> np.ndarray:
    """
    Produces a binary boundary map (H,W) for a segmentation mask (H,W).
    Boundaries are computed on non-ignore pixels.
    """
    m = mask.copy()
    # treat ignore as background for morphology but we will also mask later
    m[m == ignore_index] = 0

    kernel = np.ones((3, 3), np.uint8)
    # gradient = dilate - erode gives edges
    dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=width)
    ero = cv2.erode(m.astype(np.uint8), kernel, iterations=width)
    b = (dil != ero).astype(np.uint8)
    return b

def boundary_f1(pred: np.ndarray, gt: np.ndarray, width: int = 3, ignore_index: int = 255) -> float:
    """
    Boundary F1 with tolerance: we dilate boundaries by 1px (built into width usage).
    This is a standard, practical boundary-quality proxy.
    """
    pb = mask_to_boundary(pred, width=1, ignore_index=ignore_index)
    gb = mask_to_boundary(gt,   width=1, ignore_index=ignore_index)

    # tolerance zone: dilate each boundary to allow small shifts
    kernel = np.ones((3, 3), np.uint8)
    pb_d = cv2.dilate(pb, kernel, iterations=width)
    gb_d = cv2.dilate(gb, kernel, iterations=width)

    # Only evaluate where GT is not ignore
    valid = (gt != ignore_index)

    # Precision: predicted boundary pixels that match GT boundary zone
    tp_p = np.logical_and(pb.astype(bool), gb_d.astype(bool) & valid).sum()
    fp   = np.logical_and(pb.astype(bool), ~gb_d.astype(bool) & valid).sum()

    # Recall: GT boundary pixels that match predicted boundary zone
    tp_r = np.logical_and(gb.astype(bool), pb_d.astype(bool) & valid).sum()
    fn   = np.logical_and(gb.astype(bool), ~pb_d.astype(bool) & valid).sum()

    precision = tp_p / (tp_p + fp + 1e-6)
    recall    = tp_r / (tp_r + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(f1)

def boundary_iou(pred: np.ndarray, gt: np.ndarray, width: int = 3, ignore_index: int = 255) -> float:
    """
    Boundary IoU computed inside a narrow band around GT boundary.
    """
    pb = mask_to_boundary(pred, width=1, ignore_index=ignore_index).astype(bool)
    gb = mask_to_boundary(gt,   width=1, ignore_index=ignore_index).astype(bool)

    # band around GT boundary
    kernel = np.ones((3, 3), np.uint8)
    band = cv2.dilate(gb.astype(np.uint8), kernel, iterations=width).astype(bool)

    valid = (gt != ignore_index)
    region = band & valid

    inter = np.logical_and(pb, gb) & region
    union = np.logical_or(pb, gb) & region
    iou = inter.sum() / (union.sum() + 1e-6)
    return float(iou)

def mean_ci95(x: np.ndarray):
    """
    95% CI using normal approximation (good for N~500 val images).
    """
    x = x.astype(np.float64)
    m = x.mean()
    s = x.std(ddof=1)
    n = len(x)
    se = s / np.sqrt(n)
    ci = 1.96 * se
    return m, s, (m - ci), (m + ci)

def paired_t_stat(x: np.ndarray, y: np.ndarray):
    """
    Simple paired t-test statistic + approximate p-value via normal approx (no scipy).
    For reporting: t value and p approx.
    """
    d = (y - x).astype(np.float64)
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = md / (se + 1e-12)

    # normal approx for p (two-sided)
    # p ~ 2*(1-Phi(|t|))  using erf
    from math import erf, sqrt
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return float(md), float(sd), float(t), float(p)

# -----------------------------
# Dataset iteration helper
# -----------------------------
def build_val_dataset(cfg: Config):
    # use test_dataloader if present, otherwise val_dataloader
    dl = getattr(cfg, "test_dataloader", None) or getattr(cfg, "val_dataloader", None)
    if dl is None:
        raise RuntimeError("Config has no test_dataloader/val_dataloader.")

    ds_cfg = dl["dataset"] if isinstance(dl, dict) else dl.dataset
    # Make sure we don't wrap it in RepeatDataset etc incorrectly
    dataset = DATASETS.build(ds_cfg)
    return dataset

def get_image_path(data_info):
    # Cityscapes dataset returns dict with 'img_path' in data_info
    if isinstance(data_info, dict) and "img_path" in data_info:
        return data_info["img_path"]
    # fallback
    return None

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="Path to a text file with lines: name,cfg_path,ckpt_path")
    ap.add_argument("--out_csv", default="results/boundary/boundary_metrics.csv")
    ap.add_argument("--out_latex", default="results/boundary/boundary_metrics.tex")
    ap.add_argument("--max_images", type=int, default=500, help="How many val images to evaluate (<=500 for Cityscapes).")
    ap.add_argument("--width", type=int, default=3, help="Boundary tolerance/band width (pixels).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    register_all_modules()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_latex = Path(args.out_latex)
    out_latex.parent.mkdir(parents=True, exist_ok=True)

    # Read model list
    models = []
    with open(args.models, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, cfg_path, ckpt_path = [x.strip() for x in line.split(",", 2)]
            models.append((name, cfg_path, ckpt_path))

    if not models:
        raise RuntimeError("No models in --models file.")

    # Build dataset once using first cfg (assumes same val split for all)
    cfg0 = Config.fromfile(models[0][1])
    dataset = build_val_dataset(cfg0)
    N = min(len(dataset), args.max_images)

    # We'll store per-image metrics for each model for paired comparisons
    per_model = {}

    for name, cfg_path, ckpt_path in models:
        print(f"\n=== Evaluating {name} ===")
        model = init_model(cfg_path, ckpt_path, device=args.device)
        model.eval()

        f1s = []
        bious = []

        for idx in range(N):
            # dataset[idx] returns a data sample prepared by pipeline.
            # We want original image path to run inference (simplest) AND GT mask from data_sample.
            data = dataset[idx]
            img_path = get_image_path(dataset.data_list[idx]) if hasattr(dataset, "data_list") else None
            if img_path is None:
                # fallback: try from data dict itself
                img_path = data.get("img_path", None) if isinstance(data, dict) else None
            if img_path is None:
                raise RuntimeError("Cannot resolve img_path from dataset item; adjust for your dataset format.")

            # Inference
            with torch.no_grad():
                result = inference_model(model, img_path)

            pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

            # GT from data_sample (mmseg stores it there)
            # Depending on mmseg version, field names vary slightly.
            if hasattr(result, "gt_sem_seg") and result.gt_sem_seg is not None:
                gt = result.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            else:
                # safer: get GT from dataset item (data_sample)
                # Most mmseg pipelines yield data_sample with gt_sem_seg
                dsamp = data.get("data_samples", None) if isinstance(data, dict) else None
                if dsamp is None or not hasattr(dsamp, "gt_sem_seg"):
                    raise RuntimeError("GT not found. Ensure test/val pipeline includes annotations (gt_sem_seg).")
                gt = dsamp.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8)

            f1 = boundary_f1(pred, gt, width=args.width)
            bi = boundary_iou(pred, gt, width=args.width)
            f1s.append(f1)
            bious.append(bi)

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{N}")

        f1s = np.array(f1s, dtype=np.float64)
        bious = np.array(bious, dtype=np.float64)

        per_model[name] = {"bf1": f1s, "biou": bious}

        m_f1, s_f1, lo_f1, hi_f1 = mean_ci95(f1s)
        m_bi, s_bi, lo_bi, hi_bi = mean_ci95(bious)

        print(f"{name} BF1 mean={m_f1:.4f} std={s_f1:.4f} CI95=[{lo_f1:.4f},{hi_f1:.4f}]")
        print(f"{name} BIoU mean={m_bi:.4f} std={s_bi:.4f} CI95=[{lo_bi:.4f},{hi_bi:.4f}]")

    # Write CSV summary
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "width_px",
                    "bf1_mean", "bf1_std", "bf1_ci95_lo", "bf1_ci95_hi",
                    "biou_mean", "biou_std", "biou_ci95_lo", "biou_ci95_hi"])
        for name, _cfg, _ckpt in models:
            bf1 = per_model[name]["bf1"]
            biou = per_model[name]["biou"]
            m_f1, s_f1, lo_f1, hi_f1 = mean_ci95(bf1)
            m_bi, s_bi, lo_bi, hi_bi = mean_ci95(biou)
            w.writerow([name, len(bf1), args.width,
                        f"{m_f1:.6f}", f"{s_f1:.6f}", f"{lo_f1:.6f}", f"{hi_f1:.6f}",
                        f"{m_bi:.6f}", f"{s_bi:.6f}", f"{lo_bi:.6f}", f"{hi_bi:.6f}"])

    print(f"\nWrote summary CSV: {out_csv}")

    # Paired comparisons (example: baseline vs gated per backbone if names follow convention)
    # You can extend this mapping however you like.
    pairs = []
    def try_pair(a, b):
        if a in per_model and b in per_model:
            pairs.append((a, b))

    try_pair("B0 Baseline", "B0 Gated")
    try_pair("B2 Baseline", "B2 Gated")
    try_pair("B3 Baseline", "B3 Gated")

    # Write LaTeX table
    with out_latex.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Metryki jakości granic obiektów na zbiorze walidacyjnym Cityscapes (Boundary F1 i Boundary IoU).}" + "\n")
        f.write(r"\label{tab:boundary_metrics}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Model & BF1 $\uparrow$ & 95\% CI & BIoU $\uparrow$ & 95\% CI \\" + "\n")
        f.write(r"\midrule" + "\n")
        for name, _cfg, _ckpt in models:
            bf1 = per_model[name]["bf1"]
            biou = per_model[name]["biou"]
            m_f1, _s_f1, lo_f1, hi_f1 = mean_ci95(bf1)
            m_bi, _s_bi, lo_bi, hi_bi = mean_ci95(biou)
            f.write(f"{name} & {m_f1:.4f} & [{lo_f1:.4f},{hi_f1:.4f}] & {m_bi:.4f} & [{lo_bi:.4f},{hi_bi:.4f}] \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        if pairs:
            f.write(r"\begin{table}[H]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Porównanie parowane (te same obrazy) dla metryk granic: różnica $\Delta$ (model B - model A) oraz statystyka testu parowanego.}" + "\n")
            f.write(r"\label{tab:boundary_paired}" + "\n")
            f.write(r"\begin{tabular}{lccccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Para (A vs B) & $\Delta$BF1 & $t$ & $p$ & $\Delta$BIoU & $t$ \\" + "\n")
            f.write(r"\midrule" + "\n")
            for a, b in pairs:
                da, sda, tbf, pbf = paired_t_stat(per_model[a]["bf1"], per_model[b]["bf1"])
                db, sdb, tbi, pbi = paired_t_stat(per_model[a]["biou"], per_model[b]["biou"])
                f.write(f"{a} vs {b} & {da:+.4f} & {tbf:.2f} & {pbf:.3g} & {db:+.4f} & {tbi:.2f} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

    print(f"Wrote LaTeX tables: {out_latex}")
    print("Done.")

if __name__ == "__main__":
    main()
