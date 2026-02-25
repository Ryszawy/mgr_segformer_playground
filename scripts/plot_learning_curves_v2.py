#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


TRAIN_LINE = re.compile(
    r"Iter\(train\)\s*\[\s*(\d+)\s*/\s*(\d+)\].*?\bloss:\s*([0-9]*\.?[0-9]+).*?\bdecode\.acc_seg:\s*([0-9]*\.?[0-9]+)"
)
TRAIN_LINE_NOACC = re.compile(
    r"Iter\(train\)\s*\[\s*(\d+)\s*/\s*(\d+)\].*?\bloss:\s*([0-9]*\.?[0-9]+)"
)

VAL_ITER_LINE = re.compile(
    r"Iter\(val\)\s*\[\s*(\d+)\s*/\s*(\d+)\]"
)
MIOU_LINE = re.compile(r"\bmIoU:\s*([0-9]*\.?[0-9]+)")
AACC_LINE = re.compile(r"\baAcc:\s*([0-9]*\.?[0-9]+)")
MACC_LINE = re.compile(r"\bmAcc:\s*([0-9]*\.?[0-9]+)")


def find_run_dir(work_dir: Path, run_dir: str | None) -> Path:
    if run_dir:
        p = (work_dir / run_dir).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    # najnowszy katalog w formacie YYYYMMDD_HHMMSS
    candidates = [p for p in work_dir.iterdir() if p.is_dir() and re.match(r"^\d{8}_\d{6}$", p.name)]
    if not candidates:
        raise FileNotFoundError(f"Brak katalogu run w {work_dir} (np. 20260222_124436). Podaj --run-dir.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_log(run_dir: Path) -> Path:
    logs = sorted(run_dir.glob("*.log"))
    if logs:
        return logs[0]
    raise FileNotFoundError(f"Brak *.log w {run_dir}")


def parse_mmengine_log(log_path: Path):
    train = []
    val = []

    pending_val_iter = None
    max_iters = None

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # TRAIN
            m = TRAIN_LINE.search(line)
            if m:
                it = int(m.group(1))
                max_iters = int(m.group(2))
                loss = float(m.group(3))
                acc = float(m.group(4))
                train.append((it, loss, acc))
                continue

            m = TRAIN_LINE_NOACC.search(line)
            if m:
                it = int(m.group(1))
                max_iters = int(m.group(2))
                loss = float(m.group(3))
                train.append((it, loss, None))
                continue

            # VAL: najpierw linia Iter(val) -> zapamiętaj iter
            m = VAL_ITER_LINE.search(line)
            if m:
                pending_val_iter = int(m.group(1))
                max_iters = int(m.group(2))
                continue

            # a potem w bloku wyników: mIoU/aAcc/mAcc
            if pending_val_iter is not None:
                mm = MIOU_LINE.search(line)
                if mm:
                    miou = float(mm.group(1))
                    aacc = float(AACC_LINE.search(line).group(1)) if AACC_LINE.search(line) else None
                    macc = float(MACC_LINE.search(line).group(1)) if MACC_LINE.search(line) else None
                    val.append((pending_val_iter, miou, aacc, macc))
                    pending_val_iter = None  # zużyte
                    continue

    train_df = pd.DataFrame(train, columns=["iter", "loss", "acc_seg"]).drop_duplicates("iter").sort_values("iter")
    val_df = pd.DataFrame(val, columns=["iter", "mIoU", "aAcc", "mAcc"]).drop_duplicates("iter").sort_values("iter")
    return train_df, val_df, max_iters


def smooth(series: pd.Series, window: int):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def plot_single(train_df, val_df, out_dir: Path, title: str, smooth_win: int, wide: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    figsize = (14, 4) if wide else (6, 4)

    # LOSS
    if not train_df.empty:
        plt.figure(figsize=figsize)
        y = smooth(train_df["loss"], smooth_win)
        plt.plot(train_df["iter"], y)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{title} — Training loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "train_loss.png", dpi=200)
        plt.close()

    # ACC
    if not train_df.empty and train_df["acc_seg"].notna().any():
        plt.figure(figsize=figsize)
        y = smooth(train_df["acc_seg"].astype(float), smooth_win)
        plt.plot(train_df["iter"], y)
        plt.xlabel("Iteration")
        plt.ylabel("decode.acc_seg")
        plt.title(f"{title} — Training decode.acc_seg")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "train_acc_seg.png", dpi=200)
        plt.close()

    # VAL mIoU
    if not val_df.empty:
        plt.figure(figsize=figsize)
        plt.plot(val_df["iter"], val_df["mIoU"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("mIoU")
        plt.title(f"{title} — Validation mIoU")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "val_miou.png", dpi=200)
        plt.close()


def plot_with_errorbars(val_dfs: list[pd.DataFrame], labels: list[str], out_dir: Path, title: str, wide: bool):
    """
    Mean ± std mIoU po wielu krzywych (dopasowanie po iter).
    Uwaga: statystycznie sensowne dla wielu seedów tego samego eksperymentu,
    ale może być użyte też jako wizualizacja rozrzutu między wariantami.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figsize = (14, 4) if wide else (6, 4)

    merged = None
    for i, df in enumerate(val_dfs):
        if df is None or df.empty:
            continue
        tmp = df[["iter", "mIoU"]].copy()

        # Wymuś typy numeryczne (kluczowe dla fill_between)
        tmp["iter"] = pd.to_numeric(tmp["iter"], errors="coerce")
        tmp["mIoU"] = pd.to_numeric(tmp["mIoU"], errors="coerce")
        tmp = tmp.dropna(subset=["iter", "mIoU"])

        tmp = tmp.rename(columns={"mIoU": f"mIoU_{i}"})
        merged = tmp if merged is None else merged.merge(tmp, on="iter", how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("Brak danych walidacyjnych do porównania (val_dfs puste lub bez mIoU).")

    merged = merged.sort_values("iter")
    miou_cols = [c for c in merged.columns if c.startswith("mIoU_")]

    # mean/std po wierszach (iter)
    mean = merged[miou_cols].astype(float).mean(axis=1, skipna=True)
    std = merged[miou_cols].astype(float).std(axis=1, skipna=True)

    # Wymuś floaty dla matplotlib
    x = pd.to_numeric(merged["iter"], errors="coerce").astype(float)
    mean = pd.to_numeric(mean, errors="coerce").astype(float)
    std = pd.to_numeric(std, errors="coerce").astype(float)

    # Usuń NaN
    mask = x.notna() & mean.notna() & std.notna()
    x = x[mask]
    mean = mean[mask]
    std = std[mask]

    plt.figure(figsize=figsize)
    plt.plot(x, mean, marker="o")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("mIoU")
    plt.title(f"{title} — Validation mIoU (mean ± std)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "val_miou_mean_std.png", dpi=200)
    plt.close()

    merged_out = merged.copy()
    merged_out["mIoU_mean"] = mean.reindex(merged.index)
    merged_out["mIoU_std"] = std.reindex(merged.index)
    merged_out.to_csv(out_dir / "val_miou_mean_std.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", action="append", required=True,
                    help="Podaj jeden lub wiele work_dir. Możesz użyć kilka razy: --work-dir A --work-dir B ...")
    ap.add_argument("--run-dir", default=None, help="Opcjonalnie: konkretny katalog run (np. 20260222_124436).")
    ap.add_argument("--results-root", default="../results/learning_curves",
                    help="Gdzie zapisywać wyniki. Domyślnie: ../results/learning_curves")
    ap.add_argument("--name", default=None, help="Nazwa do tytułów/wykresów (domyślnie: nazwa work_dir lub 'comparison').")
    ap.add_argument("--smooth", type=int, default=1, help="Okno wygładzania dla train (rolling mean).")
    ap.add_argument("--wide", action="store_true", help="Szerokie wykresy (14x4).")
    args = ap.parse_args()

    work_dirs = [Path(p).resolve() for p in args.work_dir]
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # pojedynczy workdir -> normalne krzywe
    if len(work_dirs) == 1:
        wd = work_dirs[0]
        run = find_run_dir(wd, args.run_dir)
        log = find_log(run)
        train_df, val_df, _ = parse_mmengine_log(log)

        exp_name = wd.name
        out_dir = results_root / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out_dir / "train.csv", index=False)
        val_df.to_csv(out_dir / "val.csv", index=False)

        title = args.name or exp_name
        plot_single(train_df, val_df, out_dir, title, args.smooth, args.wide)

        print(f"OK: {exp_name}")
        print(f"  log: {log}")
        print(f"  out: {out_dir}")
        print(f"  train pts: {len(train_df)}")
        print(f"  val pts: {len(val_df)}")
        return

    # wiele workdirów -> error bars na walidacji
    val_dfs = []
    labels = []
    for wd in work_dirs:
        run = find_run_dir(wd, args.run_dir)
        log = find_log(run)
        _, val_df, _ = parse_mmengine_log(log)
        val_dfs.append(val_df)
        labels.append(wd.name)

    comp_name = args.name or "comparison"
    out_dir = results_root / comp_name
    plot_with_errorbars(val_dfs, labels, out_dir, comp_name, args.wide)

    print("OK: comparison mean±std")
    print(f"  out: {out_dir}")
    print("  work_dirs:", ", ".join(labels))


if __name__ == "__main__":
    main()
