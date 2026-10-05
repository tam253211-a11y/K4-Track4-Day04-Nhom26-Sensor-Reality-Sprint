"""Check final artifact consistency without rerunning the model."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
image_rows = list(csv.DictReader((OUT / "image_metrics.csv").open(encoding="utf-8", newline="")))
summary = list(csv.DictReader((OUT / "condition_summary.csv").open(encoding="utf-8", newline="")))
predictions = json.loads((OUT / "predictions.json").read_text(encoding="utf-8"))
quality_rows = list(csv.DictReader((OUT / "quality_metrics.csv").open(encoding="utf-8", newline="")))
quality_summary = list(csv.DictReader((OUT / "quality_summary.csv").open(encoding="utf-8", newline="")))
n = len(manifest)
assert n == 128, n
assert len(image_rows) == 6*n
assert len(summary) == 6
assert len(quality_rows) == 6*n and len(quality_summary) == 6
assert len({(r["image_id"], r["severity"]) for r in image_rows}) == 6*n
assert {int(r["severity"]) for r in image_rows} == set(range(6))
assert all(math.isfinite(float(r["laplacian_var"])) and float(r["laplacian_var"]) >= 0 for r in image_rows)
for severity in range(6):
    folder = OUT / "conditions" / f"severity_{severity}"
    images = list((folder / "images" / "val").glob("*.png"))
    labels = list((folder / "labels" / "val").glob("*.txt"))
    assert len(images) == len(labels) == n, (severity, len(images), len(labels))
    assert not list((folder / "images" / "val").glob("*.jpg")), severity
    rows = [r for r in image_rows if int(r["severity"]) == severity]
    item = summary[severity]
    assert int(item["n_images"]) == n
    assert sum(int(r["n_gt"]) for r in rows) == int(item["total_gt"]) == 929
    assert sum(int(r["n_pred"]) for r in rows) == int(item["total_pred"])
    assert all(int(r["tp"])+int(r["fp"]) == int(r["n_pred"]) for r in rows)
    assert all(int(r["tp"])+int(r["fn"]) == int(r["n_gt"]) for r in rows)
    total_tp = sum(int(r["tp"]) for r in rows)
    assert math.isclose(total_tp/929, float(item["recall_at_025"]), abs_tol=1e-12)
    assert 0 <= float(item["map50"]) <= 1
    assert sum(p["severity"] == severity for p in predictions) == int(item["total_pred"])
    assert sum(int(r["severity"]) == severity for r in quality_rows) == n
for name in ["blur_panel.png", "laplacian_by_severity.png", "detection_by_severity.png", "failure_case.png", "score_limitation.png", "additional_quality_metrics.png"]:
    assert (OUT / "plots" / name).is_file(), name
for name in ["REPORT.md", "PITCH.md", "run_metadata.json", "validation_log.json"]:
    assert (OUT / name).is_file(), name
assert (ROOT / "output" / "pdf" / "T1_One_Page.pdf").is_file()
print(f"PASS: {n} images, {6*n} rows, six conditions, 929 GT/condition, CSV/prediction/plots/report consistent")
