# CHI TIẾT FAILURE CASE VÀ ĐỀ XUẤT NGƯỠNG CẢNH BÁO

## 1. Định nghĩa Failure Case: Motion Blur trên ADAS
* **Tình huống thực tế:** Xe di chuyển với tốc độ cao, đi qua ổ gà gây rung lắc camera (vibration), hoặc di chuyển trong môi trường thiếu sáng khiến camera tự động tăng thời gian phơi sáng (exposure time). 
* **Hậu quả:** Các frame ảnh thu được bị vệt mờ chuyển động (Motion Blur).
* **Ảnh hưởng đến Detector:** 
  - Các đặc trưng cạnh (edges), góc (corners) của vật thể (người đi bộ, xe cộ) bị nhòe, mất thông tin không gian.
  - Model YOLO mất khả năng trích xuất đặc trưng (feature extraction), dẫn đến **giảm đột ngột Confidence Score** và **bỏ lót vật thể (tụt Recall)**.

## 2. Giải pháp Cảnh báo Sớm (Early Warning System)
Thay vì chờ đợi model nhận diện sai rồi mới phát hiện, ta cần một chỉ số đánh giá "Sức khỏe Camera" (Camera Health Score) ngay từ đầu vào.

* **Công cụ đề xuất:** **Laplacian Variance** (Phương sai của đạo hàm bậc hai).
  - *Ý nghĩa:* Tính toán độ sắc nét của ảnh dựa trên sự thay đổi cường độ điểm ảnh. Ảnh sắc nét có nhiều cạnh rõ rệt → Variance cao. Ảnh mờ (blur) làm mất cạnh → Variance thấp.
  - *Công thức (Code):* `cv2.Laplacian(image, cv2.CV_64F).var()`

## 3. Khung Đề xuất Ngưỡng Cảnh báo (Threshold Proposal)
Sau khi có Health Score và dữ liệu Recall từ các mức độ mờ (Severity 1-5), ngưỡng cảnh báo sẽ được thiết lập thông qua phân tích tương quan **Spearman ρ**.

* **Nguyên lý thiết lập ngưỡng:**
  1. Xác định mức **Recall tối thiểu chấp nhận được** cho xe ADAS (Ví dụ: Không được dưới 85%).
  2. Ánh xạ từ đồ thị Scatter Plot (Health score vs Recall) do thành viên C vẽ, tìm ra giá trị Health score tương ứng với mốc Recall 85%.
  3. Đặt giá trị đó là **T_warn**.

* **Kịch bản hệ thống Monitor:**
  - Nếu `Laplacian Variance > T_warn`: Camera ổn định, tiếp tục dùng kết quả từ YOLO.
  - Nếu `Laplacian Variance < T_warn`: Kích hoạt cờ cảnh báo (Warning flag).
    - *Hành động của xe ADAS:* Giảm tốc độ, cảnh báo tài xế cầm lái, hoặc chuyển sang lấy dữ liệu tin cậy hơn từ cảm biến khác (Radar/LiDAR).
