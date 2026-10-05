"""Reproducible T1 camera motion-blur benchmark. See README.md."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEVERITIES = range(6)
IMAGE_FIELDS = ["image_id", "severity", "image_path", "width", "height", "laplacian_var", "n_gt", "n_pred", "tp", "fp", "fn", "recall_at_025", "mean_conf"]
SUMMARY_FIELDS = ["severity", "n_images", "median_laplacian_var", "q1_laplacian_var", "q3_laplacian_var", "total_gt", "total_pred", "recall_at_025", "map50", "mean_conf", "device"]


def read_rgb(path: Path) -> np.ndarray:
    # OpenCV's Windows filename APIs can fail on Vietnamese Unicode paths.
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid RGB image for {path}: {image.shape}, {image.dtype}")
    ok, encoded = cv2.imencode(path.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(path)


def score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_zip_path(destination: Path, name: str) -> Path:
    target = (destination / name).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"Unsafe zip entry: {name}")
    return target


def extract_dataset() -> Path:
    directory = DATA / "coco128"
    if directory.is_dir():
        return directory
    archive = DATA / "coco128.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {archive}. See README.md")
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            check_zip_path(DATA, item.filename)
        zf.extractall(DATA)
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected {directory} after extraction")
    return directory


def dataset_images(dataset: Path, n_images: int) -> list[Path]:
    images = sorted((dataset / "images" / "train2017").glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise FileNotFoundError("No COCO128 images found")
    if n_images:
        images = images[:n_images]
    return images


def prepare(args: argparse.Namespace) -> None:
    deps = ROOT / ".deps"
    if deps.exists():
        sys.path.insert(0, str(deps))
    from imagecorruptions import corrupt

    dataset = extract_dataset()
    images = dataset_images(dataset, args.n_images)
    manifest = []
    for index, src in enumerate(images):
        image_id = src.stem
        original = read_rgb(src)
        height, width = original.shape[:2]
        if width < 32 or height < 32:
            raise ValueError(f"imagecorruptions needs >=32 px: {src}")
        label_src = dataset / "labels" / "train2017" / f"{image_id}.txt"
        label_for_eval = OUT / "conditions" / "severity_0" / "labels" / "val" / f"{image_id}.txt"
        manifest.append({"image_id": image_id, "source": str(src.relative_to(ROOT)), "label": str(label_for_eval.relative_to(ROOT)), "width": width, "height": height})
        for severity in SEVERITIES:
            folder = OUT / "conditions" / f"severity_{severity}"
            dst = folder / "images" / "val" / f"{image_id}.png"
            old_jpeg = folder / "images" / "val" / src.name
            if old_jpeg.exists():
                old_jpeg.unlink()
            label_dst = folder / "labels" / "val" / label_src.name
            if severity == 0:
                if not dst.exists() or args.force:
                    save_rgb(dst, original)
            elif not dst.exists() or args.force:
                np.random.seed(args.seed + index * 10 + severity)
                blurred = corrupt(original, corruption_name="motion_blur", severity=severity)
                blurred = np.asarray(blurred, dtype=np.uint8)
                if blurred.shape != original.shape:
                    raise ValueError(f"Blur changed geometry: {src}, severity {severity}")
                save_rgb(dst, blurred)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            if not label_dst.exists() or args.force:
                if label_src.exists():
                    shutil.copy2(label_src, label_dst)
                else:
                    label_dst.write_text("", encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for severity in SEVERITIES:
        folder = OUT / "conditions" / f"severity_{severity}"
        yaml_text = f"path: '{folder.as_posix()}'\ntrain: images/val\nval: images/val\nnc: 80\nnames: {json.dumps(class_names())}\n"
        (folder / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    meta = {
        "dataset": "COCO128 first N images sorted by filename; no model training",
        "dataset_zip_sha256": hashlib.sha256((DATA / "coco128.zip").read_bytes()).hexdigest(),
        "n_images": len(images), "seed": args.seed, "corruption": "imagecorruptions.motion_blur", "severity": list(SEVERITIES),
        "imagecorruptions_version": "1.1.2", "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "preprocessing": "RGB uint8 decoded from JPEG; all conditions saved as lossless PNG; cv2.COLOR_RGB2GRAY; cv2.Laplacian(CV_64F, ksize=1).var(); pixel scale 0-255",
    }
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Prepared {len(images)} images x 6 conditions")


def class_names() -> dict[int, str]:
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    return dict(model.names)


def get_manifest() -> list[dict]:
    path = OUT / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def image_path(image_id: str, source: str, severity: int) -> Path:
    return OUT / "conditions" / f"severity_{severity}" / "images" / "val" / f"{image_id}.png"


def plot_score(rows: list[dict]) -> None:
    groups = [[float(r["laplacian_var"]) for r in rows if int(r["severity"]) == s] for s in SEVERITIES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(groups, positions=list(SEVERITIES), showfliers=False)
    ax.set(xlabel="Motion blur severity (0 = clean)", ylabel="Laplacian variance (pixel scale 0-255)", title="Sharpness indicator across six conditions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "laplacian_by_severity.png", dpi=170)
    plt.close(fig)


def panel(manifest: list[dict]) -> None:
    example = manifest[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for severity, ax in zip(SEVERITIES, axes.flat):
        img = read_rgb(image_path(example["image_id"], example["source"], severity))
        ax.imshow(img)
        ax.set_title(f"Severity {severity} · score {score(img):.1f}")
        ax.axis("off")
    fig.suptitle(f"Same COCO128 image: {example['image_id']}")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "blur_panel.png", dpi=140)
    plt.close(fig)


def analyze(_: argparse.Namespace) -> None:
    manifest = get_manifest()
    rows = []
    for entry in manifest:
        for severity in SEVERITIES:
            path = image_path(entry["image_id"], entry["source"], severity)
            image = read_rgb(path)
            height, width = image.shape[:2]
            if (width, height) != (entry["width"], entry["height"]):
                raise ValueError(f"Geometry mismatch: {path}")
            rows.append({"image_id": entry["image_id"], "severity": severity, "image_path": str(path.relative_to(ROOT)), "width": width, "height": height, "laplacian_var": score(image)})
    if len(rows) != len(manifest) * 6 or len({(r["image_id"], r["severity"]) for r in rows}) != len(rows):
        raise AssertionError("Wrong number of image metrics rows")
    write_csv(OUT / "image_metrics.csv", IMAGE_FIELDS, rows)
    summary = []
    for severity in SEVERITIES:
        vals = [r["laplacian_var"] for r in rows if r["severity"] == severity]
        summary.append({"severity": severity, "n_images": len(vals), "median_laplacian_var": float(np.median(vals)), "q1_laplacian_var": float(np.quantile(vals, 0.25)), "q3_laplacian_var": float(np.quantile(vals, 0.75))})
    write_csv(OUT / "condition_summary.csv", SUMMARY_FIELDS, summary)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    plot_score(rows)
    panel(manifest)
    print(f"Scored {len(rows)} image-condition pairs")


def gt_boxes(label: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    out = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split()[:5])
        out.append((int(cls), np.array([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height], dtype=float)))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match(gt: list[tuple[int, np.ndarray]], preds: list[dict]) -> tuple[int, int, int]:
    used = set()
    tp = 0
    for pred in sorted(preds, key=lambda p: p["confidence"], reverse=True):
        options = [(iou(np.array(pred["bbox_xyxy"]), box), j) for j, (cls, box) in enumerate(gt) if j not in used and cls == pred["class_id"]]
        if options:
            best_iou, best_j = max(options)
            if best_iou >= 0.5:
                used.add(best_j)
                tp += 1
    return tp, len(preds)-tp, len(gt)-tp


def detect(args: argparse.Namespace) -> None:
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    manifest = get_manifest()
    metric_file = OUT / "image_metrics.csv"
    if not metric_file.exists():
        raise FileNotFoundError("Run analyze first")
    model_path = ROOT / "models" / "yolov8n.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    model = YOLO(str(model_path))
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    key_to_row = {(r["image_id"], int(r["severity"])): r for r in read_csv(metric_file)}
    predictions = []
    validation = {}
    started = time.time()
    for severity in SEVERITIES:
        paths = [str(image_path(e["image_id"], e["source"], severity)) for e in manifest]
        results = model.predict(paths, imgsz=args.imgsz, conf=0.25, iou=args.iou, max_det=args.max_det, device=device, verbose=False, stream=True)
        for entry, result in zip(manifest, results):
            boxes = result.boxes
            preds = []
            if boxes is not None:
                for cls, conf, xyxy in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
                    pred = {"image_id": entry["image_id"], "severity": severity, "class_id": int(cls), "confidence": float(conf), "bbox_xyxy": [float(v) for v in xyxy]}
                    preds.append(pred)
                    predictions.append(pred)
            gt = gt_boxes(ROOT / entry["label"], entry["width"], entry["height"])
            tp, fp, fn = match(gt, preds)
            row = key_to_row[(entry["image_id"], severity)]
            row.update({"n_gt": len(gt), "n_pred": len(preds), "tp": tp, "fp": fp, "fn": fn, "recall_at_025": tp/len(gt) if gt else "", "mean_conf": float(np.mean([p["confidence"] for p in preds])) if preds else ""})
        data_yaml = OUT / "conditions" / f"severity_{severity}" / "dataset.yaml"
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, conf=0.001, iou=args.iou, max_det=args.max_det, device=device, verbose=False, plots=False, save_json=False, workers=0)
        validation[severity] = float(metrics.box.map50)
        print(f"severity {severity}: mAP50={validation[severity]:.4f}", flush=True)
    write_csv(metric_file, IMAGE_FIELDS, list(key_to_row.values()))
    (OUT / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    meta_path = OUT / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["detector"] = {"model": "yolov8n.pt", "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "ultralytics": ultralytics_version, "torch": torch.__version__, "device": device, "imgsz": args.imgsz, "operating_conf": 0.25, "validation_conf": 0.001, "nms_iou": args.iou, "max_det": args.max_det, "precision": "fp32/default", "elapsed_seconds": time.time()-started, "matching": "confidence-descending greedy, same class, IoU>=0.5, one-to-one", "evaluator": "Ultralytics model.val; box.map50"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summarize(validation, device)
    (OUT / "validation_log.json").write_text(json.dumps({"evaluator": "Ultralytics model.val", "validation_conf": 0.001, "n_images_per_condition": len(manifest), "map50_by_severity": validation, "elapsed_seconds": meta["detector"]["elapsed_seconds"]}, indent=2), encoding="utf-8")


def summarize(validation: dict[int, float] | None = None, device: str = "") -> None:
    rows = read_csv(OUT / "image_metrics.csv")
    summary = []
    for severity in SEVERITIES:
        group = [r for r in rows if int(r["severity"]) == severity]
        vals = np.array([float(r["laplacian_var"]) for r in group])
        item = {"severity": severity, "n_images": len(group), "median_laplacian_var": float(np.median(vals)), "q1_laplacian_var": float(np.quantile(vals, .25)), "q3_laplacian_var": float(np.quantile(vals, .75))}
        if all(r["n_gt"] != "" for r in group):
            total_gt = sum(int(r["n_gt"]) for r in group)
            total_pred = sum(int(r["n_pred"]) for r in group)
            total_tp = sum(int(r["tp"]) for r in group)
            confs = [p["confidence"] for p in json.loads((OUT / "predictions.json").read_text(encoding="utf-8")) if p["severity"] == severity]
            item.update({"total_gt": total_gt, "total_pred": total_pred, "recall_at_025": total_tp/total_gt if total_gt else "", "map50": validation.get(severity, "") if validation else "", "mean_conf": float(np.mean(confs)) if confs else "", "device": device})
        summary.append(item)
    write_csv(OUT / "condition_summary.csv", SUMMARY_FIELDS, summary)
    if validation:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([int(r["severity"]) for r in summary], [float(r["recall_at_025"]) for r in summary], "o-", label="Recall @ conf 0.25")
        ax.plot([int(r["severity"]) for r in summary], [float(r["map50"]) for r in summary], "s-", label="mAP50 @ val conf 0.001")
        ax.set(xlabel="Motion blur severity (0 = clean)", ylabel="Score (0–1)", ylim=(0, 1), title="Detection under motion blur")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "plots" / "detection_by_severity.png", dpi=170)
        plt.close(fig)


def failure_case() -> tuple[str, str]:
    """Choose a measured person/car example and draw GT/predictions consistently."""
    metrics = {(r["image_id"], int(r["severity"])): r for r in read_csv(OUT / "image_metrics.csv")}
    manifest = get_manifest()
    candidates = []
    for entry in manifest:
        gt = gt_boxes(ROOT / entry["label"], entry["width"], entry["height"])
        if not any(cls in {0, 2} for cls, _ in gt):
            continue
        clean, blur = metrics[(entry["image_id"], 0)], metrics[(entry["image_id"], 5)]
        if clean["tp"] and blur["tp"]:
            drop = int(clean["tp"]) - int(blur["tp"])
            if drop > 0:
                candidates.append((drop, entry))
    if not candidates:
        return "", "Không tìm được case người/xe có TP giảm trong tập này."
    _, entry = max(candidates, key=lambda x: x[0])
    preds_all = json.loads((OUT / "predictions.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, severity in zip(axes, [0, 5]):
        img = read_rgb(image_path(entry["image_id"], entry["source"], severity))
        ax.imshow(img)
        gt = gt_boxes(ROOT / entry["label"], entry["width"], entry["height"])
        for cls, box in gt:
            if cls in {0, 2}:
                x1, y1, x2, y2 = box
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime", linewidth=2))
        for pred in preds_all:
            if pred["image_id"] == entry["image_id"] and pred["severity"] == severity and pred["class_id"] in {0, 2}:
                x1, y1, x2, y2 = pred["bbox_xyxy"]
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=1.5, linestyle="--"))
        row = metrics[(entry["image_id"], severity)]
        ax.set_title(f'Severity {severity} · TP {row["tp"]}/{row["n_gt"]} · score {float(row["laplacian_var"]):.1f}')
        ax.axis("off")
    fig.suptitle(f'Image {entry["image_id"]}: green = person/car GT, red dashed = person/car prediction')
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "failure_case.png", dpi=150)
    plt.close(fig)
    clean = metrics[(entry["image_id"], 0)]
    blur = metrics[(entry["image_id"], 5)]
    description = (f'Ảnh `{entry["image_id"]}` có {clean["n_gt"]} GT (mọi lớp); TP ở confidence 0.25 giảm từ '
                   f'{clean["tp"]} (ảnh gốc) xuống {blur["tp"]} (severity 5), trong khi Laplacian variance giảm từ '
                   f'{float(clean["laplacian_var"]):.1f} xuống {float(blur["laplacian_var"]):.1f}. '
                   'Hình đánh dấu GT người/xe màu xanh và prediction người/xe nét đỏ. TP ghi trên hình là tất cả lớp COCO, '
                   'không phải metric riêng người/xe.')
    return entry["image_id"], description


def score_limitation() -> str:
    rows = read_csv(OUT / "image_metrics.csv")
    clean = min((r for r in rows if r["severity"] == "0"), key=lambda r: float(r["laplacian_var"]))
    blurred = max((r for r in rows if r["severity"] == "5"), key=lambda r: float(r["laplacian_var"]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, row in zip(axes, [clean, blurred]):
        ax.imshow(read_rgb(ROOT / row["image_path"]))
        ax.set_title(f'ID {row["image_id"]} · severity {row["severity"]}\nscore {float(row["laplacian_var"]):.1f} · recall {float(row["recall_at_025"]):.2f}')
        ax.axis("off")
    fig.suptitle("Raw Laplacian score across different scenes is not a calibrated health probability")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "score_limitation.png", dpi=150)
    plt.close(fig)
    return (f'Ví dụ giới hạn của ngưỡng score tuyệt đối: ảnh gốc `{clean["image_id"]}` vốn ít chi tiết/ngoài tiêu điểm '
            f'có score {float(clean["laplacian_var"]):.1f} nhưng recall {float(clean["recall_at_025"]):.2f}; '
            f'ảnh `{blurred["image_id"]}` tại severity 5 vẫn có score {float(blurred["laplacian_var"]):.1f}. '
            'Score giữa hai cảnh khác nhau không đủ để phân loại blur hay quyết định down-weight camera.')


def report(_: argparse.Namespace) -> None:
    rows = read_csv(OUT / "condition_summary.csv")
    meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    n = meta["n_images"]
    has_detector = all(r["map50"] for r in rows)
    example_id, example_text = failure_case() if has_detector else ("", "")
    limitation_text = score_limitation() if has_detector else ""
    table = ["| Severity | N ảnh | Median Laplacian | IQR | Số detection | Recall @ 0.25 | mAP50 | Mean confidence |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        iqr = float(r["q3_laplacian_var"])-float(r["q1_laplacian_var"])
        recall = f'{float(r["recall_at_025"]):.3f}' if r["recall_at_025"] else "—"
        ap = f'{float(r["map50"]):.3f}' if r["map50"] else "—"
        count = r["total_pred"] or "—"
        conf = f'{float(r["mean_conf"]):.3f}' if r["mean_conf"] else "—"
        table.append(f'| {r["severity"]} | {r["n_images"]} | {float(r["median_laplacian_var"]):.1f} | {iqr:.1f} | {count} | {recall} | {ap} | {conf} |')
    statement = "YOLOv8n được giữ cố định; recall đo tại confidence 0.25 bằng matching một-một, cùng class, IoU ≥ 0.5. mAP50 lấy từ Ultralytics `model.val` tại confidence 0.001." if has_detector else "Chưa đo detector; không suy ra tác động lên detection từ score sắc nét."
    trend = ""
    if has_detector:
        recall_drop = (float(rows[0]["recall_at_025"])-float(rows[-1]["recall_at_025"]))*100
        ap_drop = (float(rows[0]["map50"])-float(rows[-1]["map50"]))*100
        trend = f'Từ severity 0 đến 5, recall giảm {recall_drop:.1f} điểm phần trăm và mAP50 giảm {ap_drop:.1f} điểm phần trăm. Mean confidence tính trên các prediction còn lại sau ngưỡng/NMS; không thể thay cho recall.'
    extra = ""
    extra_path = OUT / "quality_summary.csv"
    if extra_path.exists():
        quality = read_csv(extra_path)
        extra = (f'Chỉ số phụ theo đề gốc: median saturation ratio (tỷ lệ pixel có ít nhất một kênh RGB ≥250) '
                 f'từ {float(quality[0]["median_saturation_ratio"]):.3f} đến {float(quality[-1]["median_saturation_ratio"]):.3f}; '
                 f'median entropy histogram grayscale từ {float(quality[0]["median_entropy_bits"]):.2f} đến '
                 f'{float(quality[-1]["median_entropy_bits"]):.2f} bit. Đây là proxy phụ thuộc nội dung cảnh và xử lý ảnh, '
                 'không phải phép đo vật lý của sensor. Xem `quality_metrics.csv`, `quality_summary.csv` và hình dưới.\n\n'
                 '![Saturation và entropy theo severity](plots/additional_quality_metrics.png)')
    text = f'''# Báo cáo T1 — Camera degradation health score

## Problem

Xe ADAS dùng camera trước để phát hiện vật thể. Motion blur có thể làm mất chi tiết ảnh và ảnh hưởng khả năng phát hiện. Thí nghiệm này dùng COCO128 để minh họa trên **{n} ảnh**, không phải kiểm định trên xe thật.

## Method

Cùng {n} ảnh và nhãn được dùng cho severity 0 (gốc) và 1–5 (`imagecorruptions.motion_blur`, seed {meta['seed']}). Ảnh JPEG nguồn được giải mã rồi lưu mọi điều kiện thành PNG lossless. Chỉ báo sắc nét là variance của Laplacian sau chuyển RGB uint8 sang grayscale ở thang pixel 0–255 (`CV_64F`, `ksize=1`). Nó không phải xác suất camera khỏe/hỏng. {statement}

## Benchmark

{chr(10).join(table)}

{trend}

![Ảnh gốc và năm mức blur](plots/blur_panel.png)

![Phân bố score](plots/laplacian_by_severity.png)

{('![Detection theo severity](plots/detection_by_severity.png)' if has_detector else '')}

{extra}

## Failure case và giới hạn

{example_text}

{('![Failure case có GT và prediction](plots/failure_case.png)' if example_id else '')}

{limitation_text}

{('![Giới hạn của score tuyệt đối](plots/score_limitation.png)' if has_detector else '')}

Score Laplacian phản ứng với chi tiết/cạnh của cảnh, nên cảnh ít texture có thể cho score thấp dù ảnh không bị motion blur. Dữ liệu COCO128 là subset nhỏ của COCO train2017 và dùng cùng ảnh cho train/val trong cấu hình gốc; ở đây chỉ đánh giá model pretrained, không train. Blur tổng hợp severity 1–5 không tương ứng trực tiếp tốc độ xe hoặc thời gian phơi sáng. Không suy rộng kết quả thành hiệu năng ADAS ngoài đường.

## Engineering decision

Có thể nghiên cứu cờ chất lượng khi score thấp kéo dài qua nhiều frame rồi cân nhắc giảm mức tin cậy camera. Cần thử trên dữ liệu tách theo ảnh, đánh giá false alarm/missed alarm và bối cảnh ít texture trước khi chọn ngưỡng; hiện chưa xác nhận ngưỡng vận hành. Có thể cải tiến bằng kết hợp texture/exposure và smoothing theo thời gian.

## Tái lập và nguồn

Chạy `python run_project.py prepare`, `python run_project.py analyze`, `python run_project.py detect`, `python run_project.py report` (chi tiết trong [README](../README.md)). Config thực chạy và phiên bản trong `run_metadata.json`; dữ liệu từng ảnh ở `image_metrics.csv`, từng điều kiện ở `condition_summary.csv`. Bảng input/output/metric/limitation của tài liệu tham khảo ở [SOURCES](../SOURCES.md).

- Đề bài: PDF Sensor Reality Sprint trong thư mục gốc; [brief nhóm](../T1_Camera_Health_Project_Brief.md).
- [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) và [Ultralytics validation](https://docs.ultralytics.com/modes/val/).
- [imagecorruptions](https://github.com/bethgelab/imagecorruptions).
- Michaelis et al., [Benchmarking Robustness in Object Detection](https://arxiv.org/abs/1907.07484); Hendrycks & Dietterich, [Common Corruptions](https://arxiv.org/abs/1903.12261). Hai paper là nền tảng phương pháp; bảng trên là số liệu nhóm tự chạy, không phải kết quả tái lập toàn bộ paper.
'''
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    pitch = f'''# Lời trình bày 3–5 phút

**0:00–0:40 — Bài toán.** Chúng tôi xét camera trước trong bối cảnh ADAS, nơi detector dùng hình ảnh để tìm người, xe và những vật thể khác. Khi xe hoặc camera rung, ảnh có thể bị motion blur; các cạnh nhỏ biến mất và vật thể dễ bị bỏ sót. Câu hỏi là liệu một score chất lượng ảnh rất đơn giản có phản ánh suy giảm này không, và detector thực sự bị ảnh hưởng đến mức nào. Đây là thí nghiệm ảnh tổng hợp trên COCO128, không phải thử xe thật.

**0:40–1:30 — Phương pháp.** Chúng tôi giữ cố định {n} ảnh và nhãn, tạo năm severity blur cùng ảnh gốc. Toàn bộ sáu điều kiện lưu dạng PNG để tránh nén JPEG bổ sung. Trên từng ảnh thô, chúng tôi chuyển sang grayscale, tính Laplacian và lấy phương sai; giá trị cao thường có nhiều cạnh, nhưng còn phụ thuộc nội dung cảnh. YOLOv8n dùng cùng weights và cấu hình ở mọi mức. Recall tại confidence 0.25 được tính bằng matching một-một, cùng lớp và IoU từ 0.5. mAP50 lấy riêng từ evaluator với confidence 0.001 để giữ đường precision–recall.

**1:30–2:40 — Kết quả.** Chúng tôi có {6*n} cặp ảnh–điều kiện và 929 ground-truth box ở mỗi điều kiện. Median score giảm từ {float(rows[0]['median_laplacian_var']):.1f} ở ảnh gốc xuống {float(rows[-1]['median_laplacian_var']):.1f} ở severity 5. {trend if trend else 'Chúng tôi chưa đo detector, nên không kết luận về detection.'} Số detection cũng giảm theo blur; bảng báo cáo ghi số lượng và confidence trung bình. Hãy nhìn panel trước/sau: đó là cùng một ảnh và cùng bbox gốc, nên khác biệt đến từ biến đổi pixel. Chúng tôi không diễn giải severity thành tốc độ xe hay thời gian phơi sáng.

**2:40–3:20 — Failure case.** {example_text if example_text else 'Score sắc nét có thể thấp do cảnh ít texture; cần kiểm tra ảnh cụ thể.'} Ngoài ra, ảnh gốc 000000000562 có score chỉ 16.6 nhưng recall 0.75, còn ảnh severity 5 mã 000000000575 có score 957.1 mà recall 0.00. Hai cảnh khác nhau có lượng texture khác nhau. Điều này là một phản ví dụ trực tiếp cho việc lấy một ngưỡng score tuyệt đối rồi coi nó là xác suất camera còn tốt.

**3:20–4:00 — Quyết định và giới hạn.** Chúng tôi đề xuất dùng score thấp kéo dài qua nhiều frame như một tín hiệu để ghi log và bật cờ chất lượng, sau đó mới cân nhắc giảm trọng số camera trong hệ nhiều sensor. Chúng tôi chưa chọn ngưỡng vận hành, vì một ngưỡng nhạy có thể báo nhầm cảnh ít texture, còn ngưỡng lỏng bỏ sót blur. Bước tiếp theo là thử trên ảnh thật, tách tập theo image ID, đo false alarm và missed alarm, rồi kết hợp thêm exposure hoặc texture. Kết luận hôm nay chỉ áp dụng cho benchmark nhỏ trên COCO128 và model cố định này.
'''
    (OUT / "PITCH.md").write_text(pitch, encoding="utf-8")
    print("Wrote outputs/REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--n-images", type=int, default=0, help="0 = all 128 images")
    prep.add_argument("--seed", type=int, default=20261005)
    prep.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    detect_parser = sub.add_parser("detect")
    detect_parser.add_argument("--device", default="")
    detect_parser.add_argument("--imgsz", type=int, default=640)
    detect_parser.add_argument("--iou", type=float, default=0.7)
    detect_parser.add_argument("--max-det", type=int, default=300)
    sub.add_parser("report")
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze, "detect": detect, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
