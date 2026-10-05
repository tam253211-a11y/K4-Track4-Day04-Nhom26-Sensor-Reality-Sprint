# T1 — Camera degradation health score

Thí nghiệm có thể chạy lại từ [brief](T1_Camera_Health_Project_Brief.md) và [flow](T1_EXECUTION_FLOW.md): COCO128 → motion blur severity 0–5 → Laplacian variance → YOLOv8n cố định → CSV, plot và report. Đây là benchmark lớp học trên ảnh tổng hợp, không phải kiểm định ADAS trên xe.

Repo lưu code, số đo, biểu đồ và bản nộp. COCO128, weights, thư viện cài tại chỗ và 768 ảnh PNG sinh ra được loại khỏi Git; tải đầu vào theo mục dưới rồi chạy các lệnh để tái tạo. Bảng nguồn và giới hạn được cập nhật trong [SOURCES.md](SOURCES.md) và [báo cáo kết quả](outputs/REPORT.md).

## Đầu vào

- Python 3.10+ với `numpy`, `opencv-python`, `matplotlib`, `ultralytics`, `torch`, `Pillow`, `scipy`, `PyYAML`.
- `imagecorruptions==1.1.2` cài vào `.deps` bằng `python -m pip install --target .deps --no-deps imagecorruptions==1.1.2`.
- [COCO128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip) đặt ở `data/coco128.zip`. Script giải nén an toàn khi chạy.
- [YOLOv8n weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) đặt ở `models/yolov8n.pt`.

## Chạy

Từ thư mục này:

```powershell
python run_project.py prepare
python run_project.py analyze
python run_project.py detect
python extra_quality_metrics.py
python run_project.py report
python make_one_page.py
python verify_project.py
```

`prepare --n-images 8` dùng để smoke test. Mặc định `prepare` dùng toàn bộ 128 ảnh, sắp theo filename; `--force` tạo lại ảnh. `detect --device cpu` chạy CPU, còn mặc định chọn GPU khi CUDA khả dụng. Nếu thay subset, chạy lại các lệnh theo thứ tự; không trộn CSV cũ với ảnh mới.

## Đầu ra

- `outputs/manifest.json`: danh sách ảnh và nhãn cố định.
- `outputs/conditions/severity_0..5/`: ảnh PNG lossless, nhãn và YAML cho evaluator.
- `outputs/image_metrics.csv`: một dòng cho mỗi `(image_id, severity)`.
- `outputs/quality_metrics.csv` và `quality_summary.csv`: saturation ratio và grayscale entropy phụ theo đề gốc.
- `outputs/condition_summary.csv`: median/IQR score và metric detector theo severity.
- `outputs/predictions.json`: prediction ở `conf=0.25`.
- `outputs/plots/`: panel ảnh và đồ thị.
- `outputs/run_metadata.json`: phiên bản, seed, preprocessing, cấu hình detector.
- `outputs/validation_log.json`: mAP50 từng điều kiện và thời gian chạy evaluator.
- `outputs/REPORT.md`: báo cáo tự điền từ CSV.
- `outputs/PITCH.md`: lời trình bày 3–5 phút lấy từ số liệu trong CSV.
- `output/pdf/T1_One_Page.pdf`: bản tóm tắt một trang để trình bày/nộp.
- [SOURCES.md](SOURCES.md): bảng nguồn, input/output/metric và giới hạn.

Để tạo PDF cần thêm `reportlab` trong `.deps` (`python -m pip install --target .deps --no-deps reportlab`) và font Arial tại `C:/Windows/Fonts/arial.ttf`.

## Protocol

Ảnh blur được tạo bằng `imagecorruptions.motion_blur`, seed cố định cho mỗi ảnh và severity. Ảnh JPEG nguồn được giải mã một lần; mọi điều kiện lưu PNG lossless. Nhãn bbox giữ nguyên vì phép blur không đổi hình học. Ảnh thiếu file nhãn được xem là ảnh có **0 GT**; recall từng ảnh đó để trống. Score dùng ảnh thô, không dùng ảnh có box/text. Score = variance của `cv2.Laplacian(gray, cv2.CV_64F, ksize=1)`; grayscale lấy từ RGB uint8, thang 0–255.

Prediction vận hành ở `conf=0.25`. Matching TP dùng class giống nhau, IoU ≥ 0.5, một prediction khớp tối đa một GT, duyệt prediction theo confidence giảm dần. Recall toàn điều kiện = tổng TP/tổng GT. mAP50 lấy từ Ultralytics `model.val` với confidence 0.001 để giữ prediction cho đường precision–recall. Cùng weights, `imgsz`, NMS IoU, `max_det` và danh sách lớp được dùng ở mọi severity. COCO128 là tập sanity check nhỏ, không phải test độc lập hay tập lái xe.

## Kiểm tra nhanh

```powershell
python -m py_compile run_project.py
python -c "from run_project import match; import numpy as np; gt=[(0,np.array([0,0,10,10]))]; pred=[{'class_id':0,'confidence':.9,'bbox_xyxy':[0,0,10,10]}]; assert match(gt,pred)==(1,0,0)"
```

## Nguồn

- [COCO128 official documentation](https://docs.ultralytics.com/datasets/detect/coco128/)
- [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/)
- [imagecorruptions repository](https://github.com/bethgelab/imagecorruptions)
- [Michaelis et al. 2019](https://arxiv.org/abs/1907.07484); [Hendrycks & Dietterich 2019](https://arxiv.org/abs/1903.12261)
