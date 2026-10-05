# Nguồn phương pháp và giới hạn áp dụng

| Nguồn | Input → output | Metric/đóng góp liên quan | Giới hạn khi áp dụng vào bài này |
|---|---|---|---|
| [Michaelis et al., *Benchmarking Robustness in Object Detection* (2019/2020)](https://arxiv.org/abs/1907.07484) | Ảnh của Pascal/COCO/Cityscapes với nhiều corruption → detection | Đưa ra Pascal-C, Coco-C, Cityscapes-C để đánh giá độ bền detector khi chất lượng ảnh suy giảm | Nhóm chỉ thử **một** corruption trên **COCO128**, không tái lập toàn bộ benchmark và không dùng tập lái xe để kiểm định. |
| [Hendrycks & Dietterich, *Common Corruptions and Perturbations* (2019)](https://arxiv.org/abs/1903.12261) | Ảnh phân loại với corruption/perturbation → dự đoán lớp | Nền tảng cho ý tưởng severity 1–5 và kiểm tra robustness với corruption thông dụng | Nghiên cứu gốc là **classification**, không trực tiếp chứng minh metric object detection của nhóm. |
| [bethgelab/imagecorruptions](https://github.com/bethgelab/imagecorruptions) | RGB `uint8` H×W×3, severity 1–5 → ảnh bị corruption cùng hình dạng | API `corrupt(..., corruption_name='motion_blur', severity=s)`; mức severity là mức của thư viện | Blur tổng hợp không ánh xạ sang vận tốc xe hoặc phơi sáng vật lý. |
| [Ultralytics COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) | 128 ảnh đầu từ COCO train2017 + bbox → dữ liệu YOLO | Tập nhỏ để thử pipeline và kiểm tra sanity | Cấu hình gốc dùng cùng ảnh cho train và val; không phải tập test độc lập hay tập lái xe. Nhóm chỉ đánh giá pretrained YOLOv8n, không train. |
| [Ultralytics validation](https://docs.ultralytics.com/modes/val/) | Dataset YAML + model cố định → precision/recall/mAP | `model.val`, dùng `metrics.box.map50` trong bài | mAP qua đường precision–recall khác recall vận hành tại confidence 0.25. |
| [Noki et al., *Deblurring in the Wild* (2025)](https://arxiv.org/abs/2506.19445) | Video tốc độ cao từ smartphone → cặp ảnh blur/sharp | Nguồn gần đây về độ đa dạng của blur thật | Dataset deblurring từ smartphone, không phải kết quả detector hoặc benchmark ADAS của nhóm; chỉ hỗ trợ thảo luận khoảng cách giữa blur tổng hợp và blur thực tế. |

**Ranh giới kết luận:** Số liệu trong `outputs/REPORT.md` và hai CSV là kết quả chạy của dự án này. Các nguồn trên cung cấp bối cảnh, cách làm và giới hạn; không có kết quả từ paper nào được gán thành số đo của nhóm.
