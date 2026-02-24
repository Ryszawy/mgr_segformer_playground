import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=200)
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close()
    print("saved", name)

# -------------------------
# Dane globalne (Twoje)
# -------------------------
iters = np.array([20000, 40000, 80000, 160000])

b0_miou = {
    "Baseline": [58.02, 66.03, 71.22, 73.15],
    "Conv":     [58.10, 65.44, 69.69, 72.63],
    "RNN":      [58.11, 58.11, 69.70, 72.70],
    "Gated":    [60.36, 65.49, 70.49, 73.62],
}
b0_fps = {
    "Baseline": [19.5084, 20.0602, 18.6567, 18.9789],
    "Conv":     [18.5908, 19.1681, 17.8126, 19.1681],
    "RNN":      [17.3974, 17.3974, 17.1497, 17.2592],
    "Gated":    [18.4196, 18.5082, 18.2916, 19.7122],
}

# Final points @160k for trade-off
tradeoff = [
    ("B0 Baseline", 73.15, 18.9789),
    ("B0 Conv",     72.63, 19.1681),
    ("B0 RNN",      72.70, 17.2592),
    ("B0 Gated",    73.62, 19.7122),
    ("B2 Baseline", 79.26, 10.9854),
    ("B2 Gated",    79.37, 11.0779),
    ("B3 Baseline", 80.86,  8.0991),
    ("B3 Gated",    80.79,  7.9745),
]

# -------------------------
# 1) mIoU vs iteracje (B0)
# -------------------------
plt.figure()
for k, v in b0_miou.items():
    plt.plot(iters, v, marker="o", label=k)
plt.xticks(iters, [f"{i//1000}k" for i in iters])
plt.xlabel("Iteracje")
plt.ylabel("mIoU [%]")
plt.title("B0: mIoU vs iteracje")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
savefig("b0_miou_vs_iters")

# -------------------------
# 2) FPS vs iteracje (B0)
# -------------------------
plt.figure()
for k, v in b0_fps.items():
    plt.plot(iters, v, marker="o", label=k)
plt.xticks(iters, [f"{i//1000}k" for i in iters])
plt.xlabel("Iteracje")
plt.ylabel("FPS (batch=1)")
plt.title("B0: FPS vs iteracje")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
savefig("b0_fps_vs_iters")

# -------------------------
# 3) mIoU vs FPS (trade-off)
# -------------------------
plt.figure()
for name, miou, fps in tradeoff:
    plt.scatter(fps, miou)
    plt.text(fps + 0.05, miou + 0.05, name, fontsize=8)
plt.xlabel("FPS (batch=1)")
plt.ylabel("mIoU [%]")
plt.title("Trade-off: mIoU vs FPS (160k)")
plt.grid(True, linestyle="--", linewidth=0.5)
savefig("tradeoff_miou_vs_fps")

# -------------------------
# 4) ΔIoU per klasa (B0@160k)
# -------------------------
classes = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain",
    "sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
b0_iou_160k = {
    "Baseline": [97.80,82.36,91.12,58.56,51.76,57.27,63.67,72.69,92.04,62.59,94.31,77.35,50.94,93.30,79.66,75.67,58.32,57.56,72.82],
    "Conv":     [97.78,82.30,91.13,53.95,50.68,58.43,64.90,74.24,91.93,61.27,94.47,77.82,51.59,93.51,76.16,75.62,56.35,54.72,73.03],
    "RNN":      [97.41,80.90,90.71,59.44,54.43,49.40,60.98,71.91,91.12,62.21,93.37,75.65,52.26,92.84,73.04,79.87,68.37,56.03,71.35],
    "Gated":    [97.87,82.79,91.10,60.57,51.02,56.75,63.59,72.88,92.07,62.90,94.51,77.68,53.36,93.50,78.47,77.96,61.50,57.21,73.09],
}
base = np.array(b0_iou_160k["Baseline"])
d_conv  = np.array(b0_iou_160k["Conv"])  - base
d_rnn   = np.array(b0_iou_160k["RNN"])   - base
d_gated = np.array(b0_iou_160k["Gated"]) - base

x = np.arange(len(classes))
w = 0.28

plt.figure(figsize=(14,5))
plt.bar(x - w, d_conv,  w, label="Conv - Baseline")
plt.bar(x,     d_rnn,   w, label="RNN - Baseline")
plt.bar(x + w, d_gated, w, label="Gated - Baseline")
plt.axhline(0, linewidth=1)
plt.xticks(x, classes, rotation=45, ha="right")
plt.ylabel(r"$\Delta$IoU [pp]")
plt.title("B0@160k: różnice IoU per klasa względem baseline")
plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
plt.legend()
savefig("b0_delta_iou_per_class")

print("done")
