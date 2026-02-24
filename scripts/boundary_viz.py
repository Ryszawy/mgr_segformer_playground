import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
from mmseg.registry import DATASETS

def build_val_dataset(cfg: Config):
    dl = getattr(cfg, "test_dataloader", None) or getattr(cfg, "val_dataloader", None)
    ds_cfg = dl["dataset"]
    return DATASETS.build(ds_cfg)

def mask_to_boundary(mask: np.ndarray, width: int = 1, ignore_index: int = 255) -> np.ndarray:
    m = mask.copy()
    m[m == ignore_index] = 0
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=width)
    ero = cv2.erode(m.astype(np.uint8), kernel, iterations=width)
    return (dil != ero).astype(np.uint8)

def overlay_edges(image_bgr, edge, color=(0, 0, 255)):
    out = image_bgr.copy()
    ys, xs = np.where(edge > 0)
    out[ys, xs] = color
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="results/boundary/viz")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    register_all_modules()

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.cfg)
    dataset = build_val_dataset(cfg)
    model = init_model(args.cfg, args.ckpt, device=args.device)
    model.eval()

    for idx in range(min(args.k, len(dataset))):
        data_info = dataset.data_list[idx]
        img_path = data_info["img_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        with torch.no_grad():
            res = inference_model(model, img_path)

        pred = res.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

        # GT
        data = dataset[idx]
        dsamp = data["data_samples"]
        gt = dsamp.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8)

        pb = mask_to_boundary(pred, width=1)
        gb = mask_to_boundary(gt, width=1)

        # tolerance band for error visualization
        kernel = np.ones((3, 3), np.uint8)
        gb_band = cv2.dilate(gb, kernel, iterations=args.width)

        # boundary errors:
        fp = (pb > 0) & (gb_band == 0)  # pred edge but not near GT edge
        fn = (gb > 0) & (cv2.dilate(pb, kernel, iterations=args.width) == 0)  # missed GT edge

        # overlays
        overlay_gt = overlay_edges(img, gb, color=(0, 255, 0))      # GT edges green
        overlay_pr = overlay_edges(img, pb, color=(0, 0, 255))      # Pred edges red
        overlay_fp = overlay_edges(img, fp.astype(np.uint8), color=(0, 0, 255))  # FP red
        overlay_fn = overlay_edges(img, fn.astype(np.uint8), color=(255, 0, 0))  # FN blue

        stem = Path(img_path).stem
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_img.png"), img)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_gt_edges.png"), overlay_gt)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_pred_edges.png"), overlay_pr)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_fp.png"), overlay_fp)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_fn.png"), overlay_fn)

    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
