import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules

# Cityscapes palette
PALETTE = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
    [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
    [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
    [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

def colorize(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask != 255
    color[valid] = PALETTE[mask[valid]]
    return color

def overlay(img, mask_rgb, alpha=0.5):
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img, 1-alpha, mask_bgr, alpha, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    register_all_modules()

    cfg = Config.fromfile(args.cfg)
    dataset_cfg = cfg.test_dataloader["dataset"]
    dataset = DATASETS.build(dataset_cfg)

    model = init_model(args.cfg, args.ckpt, device=args.device)
    model.eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.n, len(dataset))):
        data_info = dataset.get_data_info(i)
        img_path = data_info["img_path"]
        img = cv2.imread(img_path)

        # GT
        gt = dataset[i]["data_samples"].gt_sem_seg.data.squeeze(0).numpy()

        # Pred
        with torch.no_grad():
            result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data[0].cpu().numpy()

        gt_color = colorize(gt)
        pred_color = colorize(pred)

        gt_overlay = overlay(img, gt_color, args.alpha)
        pred_overlay = overlay(img, pred_color, args.alpha)

        combined = np.hstack([gt_overlay, pred_overlay])

        cv2.imwrite(str(outdir / f"{i:04d}_gt_pred.png"), combined)

    print(f"Saved to {outdir}")

if __name__ == "__main__":
    main()
