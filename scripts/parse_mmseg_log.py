import json
import csv
import re
import sys
from pathlib import Path
from statistics import mean, median

log_path = Path(sys.argv[1])
json_out = Path(sys.argv[2])
csv_out = Path(sys.argv[3])

lines = log_path.read_text(errors="ignore").splitlines()

# Final metrics line (we still use the last occurrence as "official")
final_re = re.compile(
    r"aAcc:\s*([0-9.]+)\s+mIoU:\s*([0-9.]+)\s+mAcc:\s*([0-9.]+)"
)
aAcc = mIoU = mAcc = None
for line in reversed(lines):
    m = final_re.search(line)
    if m:
        aAcc, mIoU, mAcc = m.groups()
        break

# Collect per-iter timing + memory from progress lines:
# Example:
# Iter(test) [ 50/500] eta: ... time: 0.0505 data_time: 0.0072 memory: 1135
iter_re = re.compile(
    r"Iter\(test\).*?\btime:\s*([0-9.]+).*?\bdata_time:\s*([0-9.]+).*?\bmemory:\s*([0-9]+)"
)
times = []
data_times = []
mems = []

for line in lines:
    m = iter_re.search(line)
    if m:
        t, dt, mem = m.groups()
        times.append(float(t))
        data_times.append(float(dt))
        mems.append(int(mem))

# Peak memory (MB-ish as logged by mmengine)
peak_mem_mb = max(mems) if mems else None

# Robust timing aggregates
avg_time = mean(times) if times else None
med_time = median(times) if times else None
avg_data_time = mean(data_times) if data_times else None

# FPS for batch=1: fps ~ 1 / avg_time
avg_fps = (1.0 / avg_time) if (avg_time and avg_time > 0) else None
med_fps = (1.0 / med_time) if (med_time and med_time > 0) else None

# Per-class table
row_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
per_class = []
in_table = False
for line in lines:
    if "per class results" in line:
        in_table = True
        continue
    if in_table:
        m = row_re.match(line.strip())
        if m:
            cls, iou, acc = m.groups()
            per_class.append({"class": cls.strip(), "IoU": float(iou), "Acc": float(acc)})

result = {
    "log": str(log_path),
    "aAcc": float(aAcc) if aAcc else None,
    "mIoU": float(mIoU) if mIoU else None,
    "mAcc": float(mAcc) if mAcc else None,
    "avg_time_per_iter_s": avg_time,
    "median_time_per_iter_s": med_time,
    "avg_data_time_s": avg_data_time,
    "avg_fps_batch1": avg_fps,
    "median_fps_batch1": med_fps,
    "peak_mem_mb": peak_mem_mb,
    "per_class": per_class,
}

json_out.write_text(json.dumps(result, indent=2))

with csv_out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for k in [
        "mIoU",
        "mAcc",
        "aAcc",
        "avg_time_per_iter_s",
        "median_time_per_iter_s",
        "avg_fps_batch1",
        "median_fps_batch1",
        "peak_mem_mb",
    ]:
        w.writerow([k, result.get(k)])

    w.writerow([])
    w.writerow(["class", "IoU", "Acc"])
    for r in per_class:
        w.writerow([r["class"], r["IoU"], r["Acc"]])

print(f"Wrote {json_out}")
print(f"Wrote {csv_out}")
