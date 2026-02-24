import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]  # /workspace/mgr_segformer_playground
PARAM_CSV = BASE_DIR / "results" / "param_counts.csv"
OUT_DIR = BASE_DIR / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.savefig(OUT_DIR / f"{name}.pdf")
    plt.close()
    print(f"Saved: {OUT_DIR / (name + '.png')}")
    print(f"Saved: {OUT_DIR / (name + '.pdf')}")

def pretty_name(cfg_path: str) -> str:
    # mapuj config -> czytelna nazwa jak w pracy
    p = cfg_path.replace("\\", "/")
    if "mit-b0" in p and "segformer_exp" not in p and "segformer_rnn" not in p and "segformer_gate" not in p:
        return "B0 Baseline"
    if "mit-b0" in p and "segformer_exp" in p:
        return "B0 Conv"
    if "mit-b0" in p and "segformer_rnn" in p:
        return "B0 RNN"
    if "mit-b0" in p and "segformer_gate" in p:
        return "B0 Gated"
    if "mit-b2" in p and "segformer_gate" not in p:
        return "B2 Baseline"
    if "mit-b2" in p and "segformer_gate" in p:
        return "B2 Gated"
    if "mit-b3" in p and "segformer_gate" not in p:
        return "B3 Baseline"
    if "mit-b3" in p and "segformer_gate" in p:
        return "B3 Gated"
    return Path(p).stem

# mIoU @160k (Twoje finalne)
mIoU_160k = {
    "B0 Baseline": 73.15,
    "B0 Conv": 72.63,
    "B0 RNN": 72.70,
    "B0 Gated": 73.62,
    "B2 Baseline": 79.26,
    "B2 Gated": 79.37,
    "B3 Baseline": 80.86,
    "B3 Gated": 80.79,
}

# FPS @160k (Twoje finalne)
fps_160k = {
    "B0 Baseline": 18.978933383943822,
    "B0 Conv": 19.168104274487252,
    "B0 RNN": 17.259233690024164,
    "B0 Gated": 19.712201852946976,
    "B2 Baseline": 10.985389432055367,
    "B2 Gated": 11.077877478675086,
    "B3 Baseline": 8.099133392726978,
    "B3 Gated": 7.974481658692186,
}

def load_param_counts():
    rows = []
    with PARAM_CSV.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = pretty_name(row["config"])
            if name not in mIoU_160k:
                # ignoruj nieznane
                continue

            # w Twoim CSV są czasem dodatkowe kolumny (conv extras) — DictReader i tak je wczyta,
            # ale nas interesują podstawowe pola:
            model_m = float(row["model_params_m"])
            head_m = float(row["head_params_m"])
            rows.append({
                "name": name,
                "model_params_m": model_m,
                "head_params_m": head_m,
                "miou": float(mIoU_160k[name]),
                "fps": float(fps_160k[name]),
            })
    # sort stabilny: B0 -> B2 -> B3, baseline->others
    order = {
        "B0 Baseline": 0, "B0 Conv": 1, "B0 RNN": 2, "B0 Gated": 3,
        "B2 Baseline": 4, "B2 Gated": 5,
        "B3 Baseline": 6, "B3 Gated": 7,
    }
    rows.sort(key=lambda x: order.get(x["name"], 999))
    return rows

def plot_miou_vs_params(rows, which="model"):
    assert which in ("model", "head")
    key = "model_params_m" if which == "model" else "head_params_m"
    xlabel = "Parametry modelu [M]" if which == "model" else "Parametry dekodera [M]"
    title = "mIoU vs liczba parametrów (160k)" if which == "model" else "mIoU vs liczba parametrów dekodera (160k)"
    fname = "miou_vs_model_params" if which == "model" else "miou_vs_head_params"

    plt.figure()
    for r in rows:
        x = r[key]
        y = r["miou"]
        plt.scatter(x, y)
        plt.text(x + 0.15, y + 0.05, r["name"], fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel("mIoU [%]")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    savefig(fname)

def plot_tradeoff_miou_vs_fps(rows):
    plt.figure()
    for r in rows:
        x = r["fps"]
        y = r["miou"]
        plt.scatter(x, y)
        plt.text(x + 0.08, y + 0.05, r["name"], fontsize=8)
    plt.xlabel("FPS (batch=1)")
    plt.ylabel("mIoU [%]")
    plt.title("Trade-off: mIoU vs FPS (160k)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    savefig("miou_vs_fps_tradeoff")

def main():
    if not PARAM_CSV.exists():
        raise FileNotFoundError(f"Missing: {PARAM_CSV}")

    rows = load_param_counts()
    if not rows:
        raise RuntimeError("No rows loaded (check config name mapping).")

    plot_miou_vs_params(rows, which="model")
    plot_miou_vs_params(rows, which="head")
    plot_tradeoff_miou_vs_fps(rows)

    print("Done.")

if __name__ == "__main__":
    main()
