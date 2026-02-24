import csv
import os
import matplotlib.pyplot as plt

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# mIoU @160k (Twoje)
miou = {
    "B0 Baseline": 73.15,
    "B0 Conv": 72.63,
    "B0 RNN": 72.70,
    "B0 Gated": 73.62,
    "B2 Baseline": 79.26,
    "B2 Gated": 79.37,
    "B3 Baseline": 80.86,
    "B3 Gated": 80.79,
}

x, y, labels = [], [], []
with open("results/param_counts.csv", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        name = row["model"]
        if name not in miou:
            continue
        x.append(float(row["params_m"]))
        y.append(miou[name])
        labels.append(name)

plt.figure()
for xi, yi, name in zip(x, y, labels):
    plt.scatter(xi, yi)
    plt.text(xi + 0.05, yi + 0.05, name, fontsize=8)
plt.xlabel("Parametry [M]")
plt.ylabel("mIoU [%]")
plt.title("mIoU vs liczba parametrów (160k)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "miou_vs_params.pdf"))
plt.savefig(os.path.join(OUT_DIR, "miou_vs_params.png"), dpi=200)
plt.close()
print("saved results/plots/miou_vs_params.*")
