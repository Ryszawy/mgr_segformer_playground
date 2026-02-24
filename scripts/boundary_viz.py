import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model


# -----------------------------
# Boundary utils
# -----------------------------
def mask_to_boundary(mask, width=1):
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=width)
    ero = cv2.erode(mask.astype(np.uint8), kernel, iterations=width)
    return (dil != ero).astype(np.uint8)


def overlay_edges(img, edges, color):
    out = img.copy()
    out[edges == 1] = color
    return out


# -----------------------------
def build_dataloader(cfg):
    dl_cfg = getattr(cfg, "test_dataloader", None) or getattr(cfg, "val_dataloader", None)
    runner_cfg = dict(
        model=cfg.model,
        work_dir="./.tmp_boundary",
        test_dataloader=dl_cfg,
        test_cfg=getattr(cfg, "test_cfg", dict(type="TestLoop")),
        default_scope="mmseg",
    )
    runner = Runner.from_cfg(Config(runner_cfg))
    return runner.test_dataloader


# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.cfg)
    model = init_model(args.cfg, args.ckpt, device=args.device)
    model.eval()

    dataloader = build_dataloader(cfg)

    out_dir = Path("../results/boundary/viz") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for batch in dataloader:
        if count >= args.k:
            break

        data_samples = batch["data_samples"]
        if not isinstance(data_samples, (list, tuple)):
            data_samples = [data_samples]

        for ds in data_samples:
            if count >= args.k:
                break

            img_path = ds.metainfo["img_path"]

            # load original image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gt = ds.gt_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

            with torch.no_grad():
                res = inference_model(model, img_path)

            pred = res.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

            gt_edges = mask_to_boundary(gt, width=1)
            pred_edges = mask_to_boundary(pred, width=1)

            # FP / FN
            fp = np.logical_and(pred_edges == 1, gt_edges == 0)
            fn = np.logical_and(pred_edges == 0, gt_edges == 1)

            # overlays
            gt_overlay = overlay_edges(img, gt_edges, [0, 255, 0])     # green
            pred_overlay = overlay_edges(img, pred_edges, [0, 0, 255]) # blue
            fp_overlay = overlay_edges(img, fp.astype(np.uint8), [255, 0, 0])  # red
            fn_overlay = overlay_edges(img, fn.astype(np.uint8), [255, 255, 0]) # yellow

            cv2.imwrite(str(out_dir / f"{count:03d}_img.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{count:03d}_gt_edges.png"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{count:03d}_pred_edges.png"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{count:03d}_fp.png"), cv2.cvtColor(fp_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"{count:03d}_fn.png"), cv2.cvtColor(fn_overlay, cv2.COLOR_RGB2BGR))

            count += 1

        print(f"{count}/{args.k}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
