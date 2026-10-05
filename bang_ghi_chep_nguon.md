# BẢNG GHI CHÉP NGUỒN (PAPER ĐỌC)

## 1. Paper 1: Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming
* **Tác giả:** Michaelis et al. (2019) - arXiv: 1907.07484
* **Tóm tắt:** Đánh giá tính bền vững (robustness) của các mô hình nhận diện vật thể (Object Detection) dưới các điều kiện hỏng hóc hình ảnh (corruptions), đặc biệt ứng dụng cho xe tự lái.

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input → Output** | **Input:** Ảnh gốc từ các bộ dữ liệu MS-COCO, PASCAL VOC, Cityscapes bị làm hỏng bởi 15 loại corruptions (trong đó có Motion Blur) ở 5 mức độ nghiêm trọng (severity 1-5).<br>**Output:** Bounding boxes, class labels và confidence scores từ các mô hình detector (Faster R-CNN, Mask R-CNN, RetinaNet, YOLO, v.v.). |
| **Metric (Đo lường)** | - **mPC (mean Performance under Corruption):** Trung bình mAP trên tất cả các loại hỏng hóc và mức độ.<br>- **rPC (relative Performance under Corruption):** Hiệu năng tương đối so với mAP trên dữ liệu sạch (rPC = mPC / mAP_clean). |
| **Giới hạn (Limitations)** | - Các lỗi (corruptions) được tạo ra bằng thuật toán (synthetic), không hoàn toàn giống 100% với lỗi vật lý/quang học thực tế của camera ngoài đời.<br>- Chưa đánh giá sâu vào chuỗi thời gian (video), chủ yếu áp dụng trên ảnh tĩnh (frame-by-frame). |

## 2. Paper 2: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
* **Tác giả:** Hendrycks & Dietterich (2019) - arXiv: 1903.12261 (ICLR 2019)
* **Tóm tắt:** Bài báo gốc định nghĩa ImageNet-C và 15 loại corruptions chuẩn mực (chia làm 4 nhóm: Noise, Blur, Weather, Digital) để test độ bền vững của mạng neural.

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Input → Output** | **Input:** Ảnh phân loại (ImageNet) bị áp dụng thuật toán hỏng hóc hình ảnh.<br>**Output:** Kết quả phân loại (Classification label) từ các model. |
| **Metric (Đo lường)** | - **CE (Corruption Error):** Lỗi phân loại đối với một loại hỏng hóc cụ thể, chuẩn hóa so với baseline (AlexNet).<br>- **mCE (mean Corruption Error):** Lỗi trung bình trên toàn bộ 15 loại corruptions. |
| **Giới hạn (Limitations)** | - Tập trung hoàn toàn vào bài toán phân loại ảnh (Classification), không trực tiếp đánh giá nhận diện vật thể (Detection) và bounding boxes.<br>- Lỗi tổng hợp (synthetic) có thể tạo ra các nhiễu giả (artifacts) mà ngoài đời ít gặp. |
