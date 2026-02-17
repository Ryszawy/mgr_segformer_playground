import sys
from pathlib import Path

vis_dir = Path(sys.argv[1])
n = int(sys.argv[2])

imgs = sorted([p for p in vis_dir.rglob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
for p in imgs[n:]:
    try:
        p.unlink()
    except Exception:
        pass
print(f"Kept {min(n, len(imgs))} images in {vis_dir}")
