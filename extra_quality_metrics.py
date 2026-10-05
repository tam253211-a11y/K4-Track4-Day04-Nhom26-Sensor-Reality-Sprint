"""Additional image-quality proxies requested by the original T1 handout."""

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
rows = []
for entry in manifest:
    for severity in range(6):
        path = OUT / "conditions" / f"severity_{severity}" / "images" / "val" / f'{entry["image_id"]}.png'
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Proxy for near-white clipping, not calibrated sensor well saturation.
        saturation = float(np.mean(np.max(rgb, axis=2) >= 250))
        counts = np.bincount(gray.ravel(), minlength=256).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        rows.append({"image_id": entry["image_id"], "severity": severity, "saturation_ratio": saturation, "entropy_bits": entropy})

with (OUT / "quality_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_id", "severity", "saturation_ratio", "entropy_bits"])
    writer.writeheader()
    writer.writerows(rows)

summary = []
for severity in range(6):
    group = [r for r in rows if r["severity"] == severity]
    summary.append({"severity": severity, "n_images": len(group), "median_saturation_ratio": float(np.median([r["saturation_ratio"] for r in group])), "median_entropy_bits": float(np.median([r["entropy_bits"] for r in group]))})
with (OUT / "quality_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(range(6), [r["median_saturation_ratio"] for r in summary], "o-")
axes[0].set(xlabel="Severity", ylabel="Median near-white pixel fraction", title="Saturation proxy")
axes[1].plot(range(6), [r["median_entropy_bits"] for r in summary], "o-")
axes[1].set(xlabel="Severity", ylabel="Median grayscale entropy (bits)", title="Entropy proxy")
for ax in axes:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "plots" / "additional_quality_metrics.png", dpi=160)
plt.close(fig)
print("Wrote 768 quality metric rows, summary and plot")
