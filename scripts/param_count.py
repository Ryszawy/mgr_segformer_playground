import sys
from pathlib import Path

# Uruchamiaj z katalogu mmsegmentation (żeby ścieżki configów były proste)
# Example:
#   python ../scripts/param_count.py configs/segformer_exp/segformer_mit-b0_convhead_cityscapes.py

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def main():
    if len(sys.argv) < 2:
        print("Usage: python param_count.py <path_to_config>")
        sys.exit(1)

    cfg_path = sys.argv[1]

    import mmseg  # ważne: rejestracje
    from mmengine.config import Config
    from mmseg.registry import MODELS

    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)

    total, trainable = count_params(model)
    head_total, head_trainable = count_params(model.decode_head)

    print(f"Config: {cfg_path}")
    print(f"Model params total: {total:,} ({total/1e6:.2f}M)")
    print(f"Model params trainable: {trainable:,} ({trainable/1e6:.2f}M)")

    print(f"Decode head params total: {head_total:,} ({head_total/1e6:.2f}M)")
    print(f"Decode head params trainable: {head_trainable:,} ({head_trainable/1e6:.2f}M)")

    # Dodatkowo: policz parametry tylko “naszych nowych warstw”, jeśli istnieją
    new_names = ["fuse_conv1", "fuse_dwconv", "fuse_conv2"]
    for name in new_names:
        if hasattr(model.decode_head, name):
            m = getattr(model.decode_head, name)
            t, tr = count_params(m)
            print(f"{name} params: {t:,} ({t/1e6:.4f}M), trainable: {tr:,}")

if __name__ == "__main__":
    main()
