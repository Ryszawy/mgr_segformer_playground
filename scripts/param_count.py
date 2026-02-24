import sys
import csv
from pathlib import Path

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def main():
    if len(sys.argv) < 2:
        print("Usage: python param_count.py <path_to_config> [out_csv]")
        sys.exit(1)

    cfg_path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else None

    import torch
    import mmseg  # rejestracje
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules()

    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval()

    with torch.no_grad():
        total, trainable = count_params(model)
        head_total, head_trainable = count_params(model.decode_head)

    print(f"Config: {cfg_path}")
    print(f"Model params total: {total:,} ({total/1e6:.2f}M)")
    print(f"Model params trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"Decode head params total: {head_total:,} ({head_total/1e6:.4f}M)")
    print(f"Decode head params trainable: {head_trainable:,} ({head_trainable/1e6:.4f}M)")

    # Dodatkowo: policz parametry tylko “naszych nowych warstw”, jeśli istnieją
    new_names = ["fuse_conv1", "fuse_dwconv", "fuse_conv2"]
    extras = {}
    for name in new_names:
        if hasattr(model.decode_head, name):
            m = getattr(model.decode_head, name)
            t, tr = count_params(m)
            extras[name] = (t, tr)
            print(f"{name} params: {t:,} ({t/1e6:.4f}M), trainable: {tr:,}")

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()

        row = {
            "config": cfg_path,
            "model_params": total,
            "model_params_m": total / 1e6,
            "head_params": head_total,
            "head_params_m": head_total / 1e6,
        }
        # dorzuć extra warstwy (jeśli są)
        for k, (t, tr) in extras.items():
            row[f"{k}_params"] = t
            row[f"{k}_params_m"] = t / 1e6

        fieldnames = list(row.keys())

        with out_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

        print(f"Wrote CSV row -> {out_path}")

if __name__ == "__main__":
    main()
