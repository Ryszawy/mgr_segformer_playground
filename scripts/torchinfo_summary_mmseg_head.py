#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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


def _try_get_in_channels_from_cfg(cfg: Config) -> Optional[List[int]]:
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


def _shape_of(x: Any):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape_of(t) for t in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return str(type(x))


def _tuple2(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(x)
    return (x, x)


def inspect_ops_io(module: torch.nn.Module, inputs, title: str):
    """
    Pojedynczy forward + hooks:
      - Conv2d/ConvTranspose2d: kernel/stride/padding/dilation/groups + ch in/out
      - Pool: kernel/stride/padding
      - AdaptiveAvgPool2d: output_size
      - Linear: in/out features
      - GRU/LSTM: input_size/hidden_size/layers/bidirectional (+ shapes)
    Dodatkowo numeruje wielokrotne wywołania tego samego modułu (np. gap#1..gap#4).
    """
    records: List[Dict[str, Any]] = []
    hooks = []

    wanted = (
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.MaxPool2d,
        torch.nn.AvgPool2d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.Linear,
        torch.nn.GRU,
        torch.nn.LSTM,
    )

    # id(module) -> name (unikalne ścieżki z named_modules)
    name_by_id = {id(m): n for n, m in module.named_modules()}

    # żeby numerować wywołania tego samego modułu
    call_count_by_id: Dict[int, int] = {}

    def hook_fn(m, inp, out):
        mid = id(m)
        call_count_by_id[mid] = call_count_by_id.get(mid, 0) + 1
        call_idx = call_count_by_id[mid]

        base_name = name_by_id.get(mid, m.__class__.__name__)
        name = f"{base_name}#{call_idx}" if call_idx > 1 else base_name

        # input może być krotką - bierz pierwszy argument jeśli jest jeden,
        # a jeśli jest wiele, loguj całość (rzadkie przypadki)
        in_obj = inp[0] if len(inp) == 1 else inp

        rec: Dict[str, Any] = {
            "name": name,
            "type": m.__class__.__name__,
            "in_shape": _shape_of(in_obj),
            "out_shape": _shape_of(out),
        }

        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            rec.update({
                "in_ch": m.in_channels,
                "out_ch": m.out_channels,
                "kernel": _tuple2(m.kernel_size),
                "stride": _tuple2(m.stride),
                "padding": _tuple2(m.padding),
                "dilation": _tuple2(m.dilation),
                "groups": getattr(m, "groups", 1),
            })

        elif isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            rec.update({
                "kernel": _tuple2(m.kernel_size),
                "stride": _tuple2(m.stride),
                "padding": _tuple2(m.padding),
            })

        elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
            rec.update({"output_size": m.output_size})

        elif isinstance(m, torch.nn.Linear):
            rec.update({
                "in_features": m.in_features,
                "out_features": m.out_features,
            })

        elif isinstance(m, (torch.nn.GRU, torch.nn.LSTM)):
            rec.update({
                "input_size": m.input_size,
                "hidden_size": m.hidden_size,
                "num_layers": m.num_layers,
                "batch_first": m.batch_first,
                "bidirectional": m.bidirectional,
            })

        records.append(rec)

    # register hooks
    for n, m in module.named_modules():
        if isinstance(m, wanted):
            hooks.append(m.register_forward_hook(hook_fn))

    # run forward
    module.eval()
    with torch.no_grad():
        if isinstance(inputs, dict):
            module(**inputs)
        else:
            module(*inputs) if isinstance(inputs, (list, tuple)) else module(inputs)

    for h in hooks:
        h.remove()

    # print
    print(f"\n=== {title}: ops details (Conv/Pool/Linear/RNN) with input/output shapes ===")
    for r in records:
        line = f"- {r['name']} [{r['type']}]: in={r['in_shape']} -> out={r['out_shape']}"

        if "kernel" in r:
            line += f", k={r.get('kernel')}, s={r.get('stride')}, p={r.get('padding')}"
        if "dilation" in r:
            line += f", d={r.get('dilation')}, g={r.get('groups')}"
        if "in_ch" in r:
            line += f", ch={r.get('in_ch')}→{r.get('out_ch')}"
        if "output_size" in r:
            line += f", output_size={r['output_size']}"
        if "in_features" in r:
            line += f", feats={r['in_features']}→{r['out_features']}"
        if "hidden_size" in r:
            bi = "bi" if r.get("bidirectional") else "uni"
            line += f", rnn({bi}, in={r['input_size']}, h={r['hidden_size']}, layers={r['num_layers']})"

        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--hw", nargs=2, type=int, default=[1024, 1024])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--part", choices=["head", "model"], default="head")
    ap.add_argument("--in-ch", nargs=4, type=int, default=None)
    ap.add_argument("--mit", choices=["b0", "b1", "b2", "b3", "b4", "b5"], default="b0")
    ap.add_argument(
        "--details",
        action="store_true",
        help="Dodatkowo wypisz kernel/stride/padding + input/output shapes dla Conv/Pool/Linear/RNN",
    )
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    if "custom_imports" in cfg:
        import_modules_from_strings(**cfg.custom_imports)

    model = MODELS.build(cfg.model).eval()

    in_channels = list(args.in_ch) if args.in_ch else (_try_get_in_channels_from_cfg(cfg) or MIT_IN_CHANNELS[args.mit])

    if args.part == "head":
        inputs_4 = _make_head_inputs(in_channels, tuple(args.hw), batch=args.batch)

        print(f"\n[torchinfo] cfg={args.cfg}")
        print(f"[torchinfo] part=decode_head, in_channels={in_channels}, hw={args.hw}, batch={args.batch}\n")

        # torchinfo
        summary(model.decode_head, input_data=[inputs_4], depth=args.depth, verbose=1)

        # extra details
        if args.details:
            # decode_head przyjmuje listę 4 tensorów jako 1 argument
            inspect_ops_io(model.decode_head, inputs=(inputs_4,), title="DECODE_HEAD")

    else:
        H, W = args.hw
        dummy_img = torch.randn(args.batch, 3, H, W)

        print(f"\n[torchinfo] cfg={args.cfg}")
        print(f"[torchinfo] part=model, dummy_img=({args.batch},3,{H},{W})\n")

        summary(model, input_data=[dummy_img], depth=args.depth, verbose=1)

        if args.details:
            inspect_ops_io(model, inputs=(dummy_img,), title="MODEL")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
