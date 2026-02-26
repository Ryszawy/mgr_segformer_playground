#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# --- FORCE local repo mmseg + initialize registries ---
REPO_ROOT = Path(__file__).resolve().parents[1] / "mmsegmentation"
sys.path.insert(0, str(REPO_ROOT))

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.registry import MODELS
from torchinfo import summary


MIT_IN_CHANNELS = {
    "b0": [32, 64, 160, 256],
    "b1": [64, 128, 320, 512],
    "b2": [64, 128, 320, 512],
    "b3": [64, 128, 320, 512],
    "b4": [64, 128, 320, 512],
    "b5": [64, 128, 320, 512],
}


def _try_get_in_channels_from_cfg(cfg: Config) -> List[int] | None:
    head = cfg.model.get("decode_head", None)
    if not head:
        return None
    in_ch = head.get("in_channels", None)
    if isinstance(in_ch, (list, tuple)) and len(in_ch) == 4:
        return list(map(int, in_ch))
    return None


def _make_head_inputs(in_channels: List[int], hw: Tuple[int, int], batch: int = 1):
    H, W = hw
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(f"--hw musi być podzielne przez 32, dostałem {H}x{W}")
    return [
        torch.randn(batch, in_channels[0], H // 4,  W // 4),
        torch.randn(batch, in_channels[1], H // 8,  W // 8),
        torch.randn(batch, in_channels[2], H // 16, W // 16),
        torch.randn(batch, in_channels[3], H // 32, W // 32),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--hw", nargs=2, type=int, default=[1024, 1024])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--part", choices=["head", "model"], default="head")
    ap.add_argument("--in-ch", nargs=4, type=int, default=None)
    ap.add_argument("--mit", choices=["b0", "b1", "b2", "b3", "b4", "b5"], default="b0")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)

    # jeśli masz własne heady i config ma custom_imports, to je dociągnij
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)

    model = MODELS.build(cfg.model).eval()

    in_channels = list(args.in_ch) if args.in_ch else (_try_get_in_channels_from_cfg(cfg) or MIT_IN_CHANNELS[args.mit])
    inputs_4 = _make_head_inputs(in_channels, tuple(args.hw), batch=args.batch)

    if args.part == "head":
        print(f"\n[torchinfo] cfg={args.cfg}")
        print(f"[torchinfo] part=decode_head, in_channels={in_channels}, hw={args.hw}, batch={args.batch}\n")
        print(summary(model.decode_head, input_data=[inputs_4], depth=args.depth, verbose=1))
    else:
        H, W = args.hw
        dummy_img = torch.randn(args.batch, 3, H, W)
        print(f"\n[torchinfo] cfg={args.cfg}")
        print(f"[torchinfo] part=model, dummy_img=({args.batch},3,{H},{W})\n")
        print(summary(model, input_data=[dummy_img], depth=args.depth, verbose=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
