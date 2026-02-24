import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14,4))

data = {
    "B0": ([3.72, 4.05, 3.90, 3.79],
           [73.15, 72.63, 72.70, 73.62],
           ["Baseline","Conv","RNN","Gated"]),
    "B2": ([24.73, 24.79],
           [79.26, 79.37],
           ["Baseline","Gated"]),
    "B3": ([44.60, 44.67],
           [80.86, 80.79],
           ["Baseline","Gated"])
}

for ax, (backbone, (params, miou, labels)) in zip(axes, data.items()):
    ax.scatter(params, miou)
    for x, y, lbl in zip(params, miou, labels):
        ax.text(x + 0.1, y + 0.05, lbl, fontsize=8)

    ax.set_title(backbone)
    ax.set_xlabel("Parametry [M]")
    ax.set_ylabel("mIoU [%]")
    ax.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("results/plots/miou_vs_params_subplots.png", dpi=200)
plt.close()

print("Saved: results/plots/miou_vs_params_subplots.png")
