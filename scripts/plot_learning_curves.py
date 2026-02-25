#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_run_dir(work_dir: Path, run_dir: str | None) -> Path:
    if run_dir:
        p = (work_dir / run_dir).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Run dir not found: {p}")
        return p

    # mmengine zwykle tworzy katalog w formacie YYYYMMDD_HHMMSS
    candidates = [p for p in work_dir.iterdir() if p.is_dir() and re.match(r"^\d{8}_\d{6}$", p.name)]
    if not candidates:
        raise FileNotFoundError(
            f"Nie znalazłem katalogu run (np. 20260222_124436) w {work_dir}. "
            f"Podaj jawnie --run-dir."
        )
    # wybierz najnowszy (po mtime)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_log_file(run_path: Path, prefer_json: bool = True) -> tuple[str, Path]:
    """
    Returns (kind, path), where kind in {"json", "text"}
    """
    if prefer_json:
        json_logs = sorted(run_path.glob("*.log.json"))
        if json_logs:
            return "json", json_logs[0]

    # tekstowe logi
    txt_logs = sorted(run_path.glob("*.log"))
    if txt_logs:
        return "text", txt_logs[0]

    # fallback: cokolwiek z .log / .log.json w rekurencji
    any_json = glob.glob(str(run_path / "**/*.log.json"), recursive=True)
    if prefer_json and any_json:
        return "json", Path(any_json[0])

    any_txt = glob.glob(str(run_path / "**/*.log"), recursive=True)
    if any_txt:
        return "text", Path(any_txt[0])

    raise FileNotFoundError(f"Nie znalazłem .log ani .log.json w {run_path}")


def parse_text_log(log_path: Path):
    """
    Parsuje Twój format mmengine:
      Iter(train) [   50/160000] ... loss: 2.4098 ... decode.acc_seg: 21.7527
      Iter(val)   [16000/160000] ... mIoU: 0.7345 ...
    """
    train_pat = re.compile(
        r"Iter\(train\)\s*\[\s*(\d+)\s*/\s*(\d+)\].*?\bloss:\s*([0-9]*\.?[0-9]+)"
    )
    acc_pat = re.compile(r"\bdecode\.acc_seg:\s*([0-9]*\.?[0-9]+)")
    lr_pat = re.compile(r"\blr:\s*([0-9.eE+-]+)")
    val_pat = re.compile(
        r"Iter\(val\)\s*\[\s*(\d+)\s*/\s*(\d+)\].*?\bmIoU:\s*([0-9]*\.?[0-9]+)"
    )
    aacc_pat = re.compile(r"\baAcc:\s*([0-9]*\.?[0-9]+)")
    macc_pat = re.compile(r"\bmAcc:\s*([0-9]*\.?[0-9]+)")

    train_rows = []
    val_rows = []
    max_iters = None

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = train_pat.search(line)
            if m:
                it = int(m.group(1))
                max_iters = int(m.group(2))
                loss = float(m.group(3))

                acc = None
                macc = acc_pat.search(line)
                if macc:
                    acc = float(macc.group(1))

                lr = None
                mlr = lr_pat.search(line)
                if mlr:
                    try:
                        lr = float(mlr.group(1))
                    except Exception:
                        lr = None

                train_rows.append((it, loss, acc, lr))
                continue

            m = val_pat.search(line)
            if m:
                it = int(m.group(1))
                max_iters = int(m.group(2))
                miou = float(m.group(3))

                aacc = None
                mm = aacc_pat.search(line)
                if mm:
                    aacc = float(mm.group(1))

                macc = None
                mm2 = macc_pat.search(line)
                if mm2:
                    macc = float(mm2.group(1))

                val_rows.append((it, miou, aacc, macc))
                continue

    train_df = pd.DataFrame(train_rows, columns=["iter", "loss", "acc_seg", "lr"]).drop_duplicates("iter").sort_values("iter")
    val_df = pd.DataFrame(val_rows, columns=["iter", "mIoU", "aAcc", "mAcc"]).drop_duplicates("iter").sort_values("iter")
    meta = {"max_iters": max_iters}
    return train_df, val_df, meta


def parse_json_log(log_path: Path):
    rows = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)

    # Spróbuj standardowych kolumn
    train_df = df[df.get("loss", pd.Series([None]*len(df))).notna()].copy()
    val_df = df[df.get("mIoU", pd.Series([None]*len(df))).notna()].copy()

    # normalizacja kolumn
    if "iter" not in train_df.columns and "iteration" in train_df.columns:
        train_df["iter"] = train_df["iteration"]
    if "iter" not in val_df.columns and "iteration" in val_df.columns:
        val_df["iter"] = val_df["iteration"]

    keep_train = [c for c in ["iter", "loss", "lr", "decode.acc_seg", "acc_seg"] if c in train_df.columns]
    keep_val = [c for c in ["iter", "mIoU", "aAcc", "mAcc"] if c in val_df.columns]

    train_df = train_df[keep_train].drop_duplicates("iter").sort_values("iter")
    val_df = val_df[keep_val].drop_duplicates("iter").sort_values("iter")

    meta = {}
    return train_df, val_df, meta


def save_plots(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, title_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # LOSS
    if not train_df.empty and "iter" in train_df.columns and "loss" in train_df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(train_df["iter"], train_df["loss"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix} — Training loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "train_loss.png", dpi=200)
        plt.close()

    # ACC (opcjonalnie)
    acc_col = None
    for c in ["acc_seg", "decode.acc_seg"]:
        if c in train_df.columns:
            acc_col = c
            break
    if not train_df.empty and acc_col:
        plt.figure(figsize=(6, 4))
        plt.plot(train_df["iter"], train_df[acc_col])
        plt.xlabel("Iteration")
        plt.ylabel("decode.acc_seg")
        plt.title(f"{title_prefix} — Training decode.acc_seg")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "train_acc_seg.png", dpi=200)
        plt.close()

    # mIoU
    if not val_df.empty and "iter" in val_df.columns and "mIoU" in val_df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(val_df["iter"], val_df["mIoU"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("mIoU")
        plt.title(f"{title_prefix} — Validation mIoU")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "val_miou.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot learning curves (loss + mIoU) from mmengine/mmseg logs.")
    ap.add_argument("--work-dir", required=True, help="Path to a single experiment work_dir (e.g. .../work_dirs/b2_baseline_160k)")
    ap.add_argument("--run-dir", default=None, help="Optional run subdir (e.g. 20260222_124436). If omitted, newest run is used.")
    ap.add_argument("--out-dir", default=None, help="Output directory. Default: <work_dir>/learning_curves")
    ap.add_argument("--name", default=None, help="Title prefix for plots. Default: work_dir name.")
    ap.add_argument("--prefer-json", action="store_true", help="Prefer *.log.json if present (otherwise uses *.log).")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    run_path = find_run_dir(work_dir, args.run_dir)
    kind, log_path = find_log_file(run_path, prefer_json=args.prefer_json)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (work_dir / "learning_curves")
    title_prefix = args.name if args.name else work_dir.name

    if kind == "text":
        train_df, val_df, meta = parse_text_log(log_path)
    else:
        train_df, val_df, meta = parse_json_log(log_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    save_plots(train_df, val_df, out_dir, title_prefix)

    print("OK")
    print(f"  work_dir : {work_dir}")
    print(f"  run_dir  : {run_path.name}")
    print(f"  log      : {log_path} ({kind})")
    print(f"  out_dir  : {out_dir}")
    print(f"  train pts: {len(train_df)}")
    print(f"  val pts  : {len(val_df)}")
    if meta.get("max_iters"):
        print(f"  max_iters: {meta['max_iters']}")


if __name__ == "__main__":
    main()
