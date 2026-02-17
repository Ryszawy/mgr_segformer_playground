import json, csv, re, sys
from pathlib import Path

log_path = Path(sys.argv[1])
json_out = Path(sys.argv[2])
csv_out  = Path(sys.argv[3])

text = log_path.read_text(errors="ignore").splitlines()

# Final metrics line
final_re = re.compile(r"aAcc:\s*([0-9.]+)\s+mIoU:\s*([0-9.]+)\s+mAcc:\s*([0-9.]+).*?time:\s*([0-9.]+)")
aAcc=mIoU=mAcc=time_it=None
for line in reversed(text):
    m = final_re.search(line)
    if m:
        aAcc, mIoU, mAcc, time_it = m.groups()
        break

# Peak memory from progress lines (e.g. "memory: 953")
mem_re = re.compile(r"\bmemory:\s*([0-9]+)\b")
mem_vals = []
for line in text:
    m = mem_re.search(line)
    if m:
        mem_vals.append(int(m.group(1)))
max_mem_mb = max(mem_vals) if mem_vals else None

# Per-class table
row_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
per_class = []
in_table = False
for line in text:
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
    "time_per_iter_s": float(time_it) if time_it else None,
    "approx_fps_batch1": (1.0/float(time_it)) if time_it else None,
    "peak_mem_mb": max_mem_mb,
    "per_class": per_class,
}

json_out.write_text(json.dumps(result, indent=2))

with csv_out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for k in ["mIoU","mAcc","aAcc","time_per_iter_s","approx_fps_batch1","peak_mem_mb"]:
        w.writerow([k, result.get(k)])
    w.writerow([])
    w.writerow(["class","IoU","Acc"])
    for r in per_class:
        w.writerow([r["class"], r["IoU"], r["Acc"]])

print(f"Wrote {json_out}")
print(f"Wrote {csv_out}")
