import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model

# -----------------------------
# Boundary utilities
# -----------------------------
def mask_to_boundary(mask: np.ndarray, width: int = 1, ignore_index: int = 255) -> np.ndarray:
    m = mask.copy()
    m[m == ignore_index] = 0
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=width)
    ero = cv2.erode(m.astype(np.uint8), kernel, iterations=width)
    return (dil != ero).astype(np.uint8)

def boundary_f1(pred: np.ndarray, gt: np.ndarray, width: int = 3, ignore_index: int = 255) -> float:
    pb = mask_to_boundary(pred, width=1, ignore_index=ignore_index)
    gb = mask_to_boundary(gt,   width=1, ignore_index=ignore_index)

    kernel = np.ones((3, 3), np.uint8)
    pb_d = cv2.dilate(pb, kernel, iterations=width)
    gb_d = cv2.dilate(gb, kernel, iterations=width)

    valid = (gt != ignore_index)

    tp_p = np.logical_and(pb.astype(bool), (gb_d.astype(bool) & valid)).sum()
    fp   = np.logical_and(pb.astype(bool), (~gb_d.astype(bool) & valid)).sum()

    tp_r = np.logical_and(gb.astype(bool), (pb_d.astype(bool) & valid)).sum()
    fn   = np.logical_and(gb.astype(bool), (~pb_d.astype(bool) & valid)).sum()

    precision = tp_p / (tp_p + fp + 1e-6)
    recall    = tp_r / (tp_r + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(f1)

def boundary_iou(pred: np.ndarray, gt: np.ndarray, width: int = 3, ignore_index: int = 255) -> float:
    pb = mask_to_boundary(pred, width=1, ignore_index=ignore_index).astype(bool)
    gb = mask_to_boundary(gt,   width=1, ignore_index=ignore_index).astype(bool)

    kernel = np.ones((3, 3), np.uint8)
    band = cv2.dilate(gb.astype(np.uint8), kernel, iterations=width).astype(bool)

    valid = (gt != ignore_index)
    region = band & valid

    inter = (pb & gb) & region
    union = (pb | gb) & region
    return float(inter.sum() / (union.sum() + 1e-6))

def mean_ci95(x: np.ndarray):
    x = x.astype(np.float64)
    m = x.mean()
    s = x.std(ddof=1)
    n = len(x)
    se = s / np.sqrt(n)
    ci = 1.96 * se
    return m, s, (m - ci), (m + ci)

def paired_t_stat(x: np.ndarray, y: np.ndarray):
    d = (y - x).astype(np.float64)
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = md / (se + 1e-12)
    from math import erf, sqrt
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return float(md), float(sd), float(t), float(p)

# -----------------------------
# Dataloader builder
# -----------------------------
def build_val_dataloader(cfg: Config):
    # Prefer test_dataloader if exists (mmseg standard), fallback to val_dataloader
    dl_cfg = getattr(cfg, "test_dataloader", None) or getattr(cfg, "val_dataloader", None)
    if dl_cfg is None:
        raise RuntimeError("Config has no test_dataloader/val_dataloader.")

    # Build Runner just to build dataloader in a version-stable way
    # Minimal runner config:
    runner_cfg = dict(
        model=cfg.model,
        work_dir="./.tmp_boundary",
        test_dataloader=dl_cfg,
        test_cfg=getattr(cfg, "test_cfg", dict(type="TestLoop")),
        test_evaluator=getattr(cfg, "test_evaluator", None),
        env_cfg=dict(cudnn_benchmark=False),
        default_scope="mmseg",
    )
    runner = Runner.from_cfg(Config(runner_cfg))
    return runner.test_dataloader

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="Text file: name,cfg,ckpt per line")
    ap.add_argument("--out_csv", default="results/boundary/boundary_metrics.csv")
    ap.add_argument("--out_latex", default="results/boundary/boundary_tables.tex")
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    register_all_modules()

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_latex = Path(args.out_latex); out_latex.parent.mkdir(parents=True, exist_ok=True)

    models = []
    with open(args.models, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, cfg_path, ckpt_path = [x.strip() for x in line.split(",", 2)]
            models.append((name, cfg_path, ckpt_path))
    if not models:
        raise RuntimeError("No models in list.")

    # Build dataloader once using first cfg (same split for all)
    cfg0 = Config.fromfile(models[0][1])
    val_loader = build_val_dataloader(cfg0)

    per_model = {}

    for name, cfg_path, ckpt_path in models:
        print(f"\n=== Evaluating {name} ===")
        model = init_model(cfg_path, ckpt_path, device=args.device)
        model.eval()

        f1s, bious = [], []
        seen = 0

        for batch in val_loader:
            if seen >= args.max_images:
                break

            # batch contains inputs + data_samples
            data_samples = batch["data_samples"]
            # mmengine usually returns list of data_samples (batch size may be 1)
            if not isinstance(data_samples, (list, tuple)):
                data_samples = [data_samples]

            for ds in data_samples:
                if seen >= args.max_images:
                    break

                # img_path from metainfo
                img_path = ds.metainfo.get("img_path", None)
                if img_path is None:
                    raise RuntimeError("img_path not found in data_sample metainfo.")

                # GT
                if not hasattr(ds, "gt_sem_seg") or ds.gt_sem_seg is None:
                    raise RuntimeError("GT (gt_sem_seg) not found. Ensure test pipeline includes annotations.")
                gt = ds.gt_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

                with torch.no_grad():
                    res = inference_model(model, img_path)
                pred = res.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

                f1s.append(boundary_f1(pred, gt, width=args.width))
                bious.append(boundary_iou(pred, gt, width=args.width))

                seen += 1

            if seen % 50 == 0:
                print(f"  {seen}/{args.max_images}")

        f1s = np.array(f1s, dtype=np.float64)
        bious = np.array(bious, dtype=np.float64)
        per_model[name] = {"bf1": f1s, "biou": bious}

        m_f1, s_f1, lo_f1, hi_f1 = mean_ci95(f1s)
        m_bi, s_bi, lo_bi, hi_bi = mean_ci95(bious)
        print(f"{name} BF1 mean={m_f1:.4f} std={s_f1:.4f} CI95=[{lo_f1:.4f},{hi_f1:.4f}]")
        print(f"{name} BIoU mean={m_bi:.4f} std={s_bi:.4f} CI95=[{lo_bi:.4f},{hi_bi:.4f}]")

    # Write CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "width_px",
                    "bf1_mean", "bf1_std", "bf1_ci95_lo", "bf1_ci95_hi",
                    "biou_mean", "biou_std", "biou_ci95_lo", "biou_ci95_hi"])
        for name in per_model:
            bf1 = per_model[name]["bf1"]
            biou = per_model[name]["biou"]
            m_f1, s_f1, lo_f1, hi_f1 = mean_ci95(bf1)
            m_bi, s_bi, lo_bi, hi_bi = mean_ci95(biou)
            w.writerow([name, len(bf1), args.width,
                        f"{m_f1:.6f}", f"{s_f1:.6f}", f"{lo_f1:.6f}", f"{hi_f1:.6f}",
                        f"{m_bi:.6f}", f"{s_bi:.6f}", f"{lo_bi:.6f}", f"{hi_bi:.6f}"])
    print(f"\nWrote summary CSV: {out_csv}")

    # Paired comparisons
    pairs = []
    def try_pair(a, b):
        if a in per_model and b in per_model:
            pairs.append((a, b))

    try_pair("B0 Baseline", "B0 Gated")
    try_pair("B2 Baseline", "B2 Gated")
    try_pair("B3 Baseline", "B3 Gated")

    # LaTeX output
    with out_latex.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Metryki jakości granic obiektów na zbiorze walidacyjnym Cityscapes: Boundary F1 oraz Boundary IoU (pas tolerancji $w$).}" + "\n")
        f.write(r"\label{tab:boundary_metrics}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Model & BF1 $\uparrow$ & 95\% CI & BIoU $\uparrow$ & 95\% CI \\" + "\n")
        f.write(r"\midrule" + "\n")

        for name in per_model:
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
            f.write(r"\caption{Porównanie parowane (te same obrazy): różnice $\Delta$ (B - A) oraz statystyka testu parowanego.}" + "\n")
            f.write(r"\label{tab:boundary_paired}" + "\n")
            f.write(r"\begin{tabular}{lccccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Para (A vs B) & $\Delta$BF1 & $t$ & $p$ & $\Delta$BIoU & $t$ \\" + "\n")
            f.write(r"\midrule" + "\n")
            for a, b in pairs:
                d_bf1, _sd, tbf, pbf = paired_t_stat(per_model[a]["bf1"], per_model[b]["bf1"])
                d_bi,  _sd2, tbi, pbi = paired_t_stat(per_model[a]["biou"], per_model[b]["biou"])
                f.write(f"{a} vs {b} & {d_bf1:+.4f} & {tbf:.2f} & {pbf:.3g} & {d_bi:+.4f} & {tbi:.2f} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")

    print(f"Wrote LaTeX tables: {out_latex}")
    print("Done.")

if __name__ == "__main__":
    main()
