"""Create the required one-page PDF summary from measured artifacts."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".deps"))
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

OUT = ROOT / "outputs"
PDF = ROOT / "output" / "pdf" / "T1_One_Page.pdf"
PDF.parent.mkdir(parents=True, exist_ok=True)
font_path = Path("C:/Windows/Fonts/arial.ttf")
if not font_path.exists():
    raise FileNotFoundError("Arial Unicode font required at C:/Windows/Fonts/arial.ttf")
pdfmetrics.registerFont(TTFont("Arial", str(font_path)))
rows = list(csv.DictReader((OUT / "condition_summary.csv").open(encoding="utf-8", newline="")))
images = list(csv.DictReader((OUT / "image_metrics.csv").open(encoding="utf-8", newline="")))
by_key = {(r["image_id"], int(r["severity"])): r for r in images}
case_id = "000000000257"
if (case_id, 0) not in by_key:
    case_id = images[0]["image_id"]
case_clean, case_blur = by_key[(case_id, 0)], by_key[(case_id, 5)]
lowest_clean = min((r for r in images if r["severity"] == "0" and r["recall_at_025"]), key=lambda r: float(r["laplacian_var"]))
recall_drop = (float(rows[0]["recall_at_025"])-float(rows[-1]["recall_at_025"]))*100
ap_drop = (float(rows[0]["map50"])-float(rows[-1]["map50"]))*100
w, h = landscape(A4)
c = canvas.Canvas(str(PDF), pagesize=(w, h))
c.setTitle("T1 Camera Degradation Health Score")


def txt(x, y, content, size=9, color="#222222", font="Arial"):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, content)


def rule(x1, y, x2, color="#CBD5E1"):
    c.setStrokeColor(color)
    c.setLineWidth(.6)
    c.line(x1, y, x2, y)


txt(30, h-36, "T1 | CAMERA DEGRADATION HEALTH SCORE", 16, "#123A5A")
txt(30, h-54, "COCO128 · 128 ảnh × 6 điều kiện · motion blur tổng hợp · YOLOv8n cố định", 9, "#4B5563")
rule(30, h-64, w-30)

x = 30
txt(x, 510, "BÀI TOÁN", 10, "#176B87")
txt(x, 495, "Camera trước ADAS có thể mất chi tiết khi motion blur; detector có thể bỏ sót vật thể.", 8.5)
txt(x, 482, "Thử cùng 128 ảnh và nhãn ở severity 0–5; đây là demo, không phải kiểm định xe thật.", 8.5)

txt(x, 458, "PHƯƠNG PHÁP", 10, "#176B87")
txt(x, 443, "Ảnh nguồn giải mã rồi lưu PNG lossless. Score = variance của Laplacian trên grayscale.", 8.5)
txt(x, 430, "Recall: conf 0.25, cùng class, IoU ≥ 0.5, matching một-một; mAP50: val conf 0.001.", 8.5)

txt(x, 406, "BENCHMARK", 10, "#176B87")
headers = [("Blur", 30), ("Median score", 82), ("Detection", 175), ("Recall", 255), ("mAP50", 320)]
for label, col_x in headers:
    txt(col_x, 390, label, 8.5, "#123A5A")
rule(30, 384, 398)
for i, row in enumerate(rows):
    y = 367-i*19
    vals = [row["severity"], f'{float(row["median_laplacian_var"]):.1f}', row["total_pred"], f'{float(row["recall_at_025"]):.3f}', f'{float(row["map50"]):.3f}']
    for val, (_, col_x) in zip(vals, headers):
        txt(col_x, y, val, 8.5)
    rule(30, y-6, 398, "#E5E7EB")

txt(x, 236, "PHÁT HIỆN VÀ GIỚI HẠN", 10, "#176B87")
txt(x, 220, f"Severity 0 → 5: recall giảm {recall_drop:.1f} điểm %, mAP50 giảm {ap_drop:.1f} điểm %.", 8.5)
txt(x, 206, f"Ảnh {int(case_id)}: TP {case_clean['tp']}/{case_clean['n_gt']} → {case_blur['tp']}/{case_blur['n_gt']}; score {float(case_clean['laplacian_var']):.1f} → {float(case_blur['laplacian_var']):.1f}.", 8.5)
txt(x, 192, f"Một ảnh gốc score {float(lowest_clean['laplacian_var']):.1f} vẫn recall {float(lowest_clean['recall_at_025']):.2f}; ngưỡng tuyệt đối có thể báo nhầm.", 8.5)

txt(x, 166, "QUYẾT ĐỊNH KỸ THUẬT", 10, "#176B87")
txt(x, 150, "Score thấp nhiều frame có thể bật cờ chất lượng và ghi log để xem xét down-weight.", 8.5)
txt(x, 136, "Cần dữ liệu thật, split theo ảnh, false/missed alarm và kiểm tra cảnh ít texture.", 8.5)
txt(x, 122, "Chưa có ngưỡng vận hành; severity không tương ứng tốc độ xe/phơi sáng.", 8.5)

right_x = 420
panel = OUT / "plots" / "failure_case.png"
chart = OUT / "plots" / "detection_by_severity.png"
txt(right_x, 510, "FAILURE CASE · GT xanh, prediction đỏ", 9.5, "#176B87")
c.drawImage(ImageReader(str(panel)), right_x, 296, width=390, height=205, preserveAspectRatio=True, anchor="c", mask="auto")
txt(right_x, 275, "DETECTION THEO MỨC BLUR", 9.5, "#176B87")
c.drawImage(ImageReader(str(chart)), right_x, 82, width=390, height=185, preserveAspectRatio=True, anchor="c", mask="auto")

rule(30, 68, w-30)
txt(30, 52, "Nguồn: COCO128, imagecorruptions 1.1.2, Ultralytics 8.4.21. Số đo chi tiết: outputs/REPORT.md", 7.5, "#4B5563")
txt(30, 40, "Score chỉ là proxy độ sắc nét; COCO128 không đại diện cho toàn bộ điều kiện ADAS.", 7.5, "#4B5563")
c.showPage()
c.save()
print(PDF)
