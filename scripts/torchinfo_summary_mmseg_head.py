#!/usr/bin/env python3
"""
torchinfo_summary_mmseg_head.py

Użycie:
  python torchinfo_summary_mmseg_head.py \
    --cfg configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py \
    --hw 1024 1024 \
    --depth 6

Opcje:
  --part head|model     (domyślnie: head)
  --in-ch 32 64 160 256 (opcjonalnie: ręczne kanały wejściowe 4 skal)
  --mit b0|b1|b2|b3|b4|b5 (fallback, gdy nie da się wyczytać z cfg)
"""

import argparse
import sys
from typing import List, Tuple

import torch
from mmengine.config import Config
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
    """Spróbuj wyciągnąć listę in_channels z cfg.model.decode_head.in_channels."""
    try:
        head = cfg.model.get("decode_head", None)
        if head is None:
            return None
        in_ch = head.get("in_channels", None)
        if in_ch is None:
            return None
        if isinstance(in_ch, (list, tuple)) and len(in_ch) == 4:
            return list(map(int, in_ch))
        # czasem bywa int (gdy head nie jest multi-scale) – wtedy nie pasuje do SegFormer
        return None
    except Exception:
        return None


def _make_head_inputs(in_channels: List[int], hw: Tuple[int, int], batch: int = 1):
    """Zbuduj 4 feature mapy (1/4,1/8,1/16,1/32) jak w SegFormer."""
    H, W = hw
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(f"--hw musi być podzielne przez 32, dostałem {H}x{W}")

    x1 = torch.randn(batch, in_channels[0], H // 4,  W // 4)
    x2 = torch.randn(batch, in_channels[1], H // 8,  W // 8)
    x3 = torch.randn(batch, in_channels[2], H // 16, W // 16)
    x4 = torch.randn(batch, in_channels[3], H // 32, W // 32)
    return [x1, x2, x3, x4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Ścieżka do configa MMSeg")
    ap.add_argument("--hw", nargs=2, type=int, default=[1024, 1024], help="H W (domyślnie 1024 1024)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--part", choices=["head", "model"], default="head", help="Co podsumować torchinfo")
    ap.add_argument("--in-ch", nargs=4, type=int, default=None, help="Ręczne in_channels: 4 liczby")
    ap.add_argument("--mit", choices=["b0", "b1", "b2", "b3", "b4", "b5"], default="b0",
                    help="Fallback kanałów jeśli nie da się wyczytać z configa")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)

    # build model
    model = MODELS.build(cfg.model).eval()

    # wybierz kanały
    in_channels = None
    if args.in_ch is not None:
        in_channels = list(args.in_ch)
    else:
        in_channels = _try_get_in_channels_from_cfg(cfg)
        if in_channels is None:
            in_channels = MIT_IN_CHANNELS[args.mit]

    # przygotuj wejścia dla heada/modelu
    inputs_4 = _make_head_inputs(in_channels, tuple(args.hw), batch=args.batch)

    if args.part == "head":
        # decode_head w SegFormer przyjmuje listę 4 tensorów jako jeden argument
        print(f"\n[torchinfo] cfg={args.cfg}")
        print(f"[torchinfo] part=decode_head, in_channels={in_channels}, hw={args.hw}, batch={args.batch}\n")
        print(summary(model.decode_head, input_data=[inputs_4], depth=args.depth, verbose=1))
    else:
        # cały model w MMSeg często oczekuje 'inputs' jako obraz, a nie listę feature map.
        # Najbezpieczniej zrobić summary na forward dummy image.
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
