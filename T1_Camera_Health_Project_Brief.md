# BRIEF T1 — CAMERA DEGRADATION HEALTH SCORE

**Nhóm:** 3 người, vai trò A/B/C theo phân công đã gửi.
**Phiên bản:** 1.0, ngày 05/10/2026.
**Bài tập:** Sensor Reality Sprint — bài thực hành 120 phút, trình bày 3–5 phút.
**Trạng thái:** Kế hoạch thực nghiệm; chưa có kết quả đo.
**Nguồn yêu cầu:** PDF Sensor Reality Sprint do người dùng cung cấp, đặc biệt trang 1–4 và 8. Đây là bài tập T1 mới; không tiếp tục phạm vi radar hoặc roadmap cá nhân 16 ngày trước đó.

## 1. Đọc nhanh: chúng ta đang làm gì?

Nhóm kiểm tra ảnh camera bị motion blur ảnh hưởng như thế nào đến chất lượng ảnh và khả năng phát hiện vật thể. Dùng cùng một tập ảnh, tạo năm mức blur, tính một chỉ báo độ sắc nét, và nếu đủ thời gian chạy một detector cố định để đo ảnh hưởng.

Ví dụ: trong ảnh gốc, người và xe tương đối rõ. Sau khi thêm blur, cạnh vật thể bị nhòe; detector có thể bỏ sót hoặc dự đoán sai. Nhóm cần cho thấy ảnh trước/sau, kết quả số, và giải thích có nên giảm mức tin cậy vào camera trong tình huống này hay không.

**Không cần train YOLO. Không cần radar, camera thật hoặc xe thật.** Bản demo dùng dữ liệu ảnh và blur tổng hợp. Liên hệ đến ADAS là bối cảnh ứng dụng, không phải bằng chứng hệ thống đã được kiểm định trên xe.

## 2. Yêu cầu từ đề và cách đáp ứng

| Yêu cầu trong PDF | Nhóm đáp ứng bằng gì? |
|---|---|
| Nêu platform, tính năng, sensor | Xe ADAS; phát hiện vật thể; camera phía trước |
| Một failure cụ thể | Motion blur có thể xuất hiện khi chuyển động/rung và thời gian phơi sáng dài |
| T1: tạo 3–5 mức degradation | Tạo motion blur severity 1–5, thêm điều kiện gốc severity 0 |
| Có metric định lượng | Laplacian variance là metric tối thiểu; recall/mAP50 là mở rộng |
| Bảng trước/sau và nhận xét down-weight camera | Bảng health score, ảnh minh họa và quyết định kỹ thuật có giới hạn |
| Code/log/ảnh/plot chứng minh chạy | Script, cấu hình, CSV, log, ảnh gốc/blur |
| Một failure case và đề xuất cải tiến | Chọn ví dụ thật từ lần chạy; đề xuất cảnh báo thích ứng hoặc kết hợp nhiều chỉ báo |
| Giải thích paper/repo: input/output/metric/limitation | A ghi nguồn, tách kết luận nguồn khỏi kết quả nhóm |
| Report/slide ngắn, trình bày 3–5 phút | Dùng khung ở mục 13 |

Rubric: demo/benchmark 40%, hiểu failure 25%, giải thích thuật toán 20%, trade-off 15%. Đề cho phép benchmark mô phỏng nhỏ khi thiếu tài nguyên, nhưng phải giải thích proxy metric. Đề không bắt buộc train model, Spearman hoặc một ngưỡng cảnh báo đã được xác nhận.

## 3. Câu hỏi, giả thuyết và giới hạn kết luận

**Câu hỏi chính:** khi tăng motion blur, chỉ báo độ sắc nét thay đổi ra sao? Nếu có detector, sự thay đổi có liên quan đến giảm chất lượng detection không?

**Giả thuyết:** trên cùng ảnh, blur tăng thường làm Laplacian variance giảm; recall/mAP50 có thể giảm; health score có thể liên hệ cùng chiều với recall. Không giả định mọi ảnh và mọi metric đều giảm đơn điệu.

Trong brief, “health score” là tên gọi vận hành của một **chỉ báo độ sắc nét**, chưa phải xác suất camera khỏe hoặc hỏng. Score thấp có thể do cảnh ít chi tiết, chứ không riêng motion blur. Tương quan với recall không chứng minh dự báo tương lai, cảnh báo sớm hoặc nguyên nhân duy nhất của lỗi detector.

Mục tiêu phù hợp trong lớp là **kiểm tra chỉ báo cảnh báo suy giảm ở frame hiện tại**. Nếu chưa có kiểm tra ngưỡng trên dữ liệu giữ riêng, chỉ đề xuất ngưỡng thăm dò, không khẳng định dùng được trên xe thật.

## 4. Hai mức hoàn thành

### Mức tối thiểu: phải hoàn thành trong 120 phút

- Có một tập ảnh cố định và năm mức motion blur.
- Tính Laplacian variance theo cùng preprocessing.
- Lưu bảng score theo điều kiện, plot và ảnh trước/sau.
- Chỉ ra một failure hoặc limitation của score.
- Nêu nguồn, input/output, một quyết định kỹ thuật và một đề xuất cải tiến.
- Có slide/report ngắn và bằng chứng chạy.

### Mức mở rộng: làm sau khi bản tối thiểu chạy

- Chạy yolov8n pretrained trên sáu điều kiện.
- Đo recall và mAP50 bằng nhãn có sẵn, xuất prediction.
- Phân tích confidence cùng số detection.
- Nếu có matching per-image đúng: tính recall từng ảnh, vẽ scatter và Spearman.
- Nếu còn thời gian: thử ngưỡng cảnh báo trên tập ảnh giữ riêng.

Nếu YOLO hoặc dependency bị lỗi, báo rõ chưa đo tác động lên detector. Vẫn có thể đáp ứng benchmark tối thiểu T1 bằng health score; không tạo số recall/mAP giả và không kết luận về hiệu năng detection khi chưa đo.

## 5. Dữ liệu, model và điều kiện kiểm soát

**Dữ liệu nhóm chọn:** COCO128 có ảnh và bounding box. Theo tài liệu Ultralytics, đây là 128 ảnh đầu từ COCO train2017; cấu hình mặc định dùng cùng ảnh cho train và val. Phù hợp sanity check/demo, không phải benchmark độc lập chứng minh tổng quát hóa. Nhóm chỉ đánh giá model pretrained, không train trên tập này.

**Detector cố định:** yolov8n.pt. Không đổi weights giữa các mức blur. Cấu hình vận hành dự kiến: imgsz=640, confidence=0.25; chốt thêm NMS IoU, max_det, device và precision. Lưu phiên bản thư viện và cấu hình thực sự dùng. Thời gian chạy CPU phải đo thử, không cam kết mỗi bộ mất một phút.

**Phạm vi lớp:** bản đầu có thể báo toàn bộ lớp COCO hiện diện; ghi rõ đây là general-object benchmark liên hệ đến ADAS. Nếu muốn chỉ person/car, cần lọc prediction và nhãn nhất quán, báo số ảnh/đối tượng hỗ trợ; không gọi metric tất cả lớp là metric riêng người/xe.

**Blur:** dùng `imagecorruptions`, corruption_name='motion_blur', severity=1..5. Đây là mức của thư viện, không đại diện trực tiếp cho tốc độ xe hoặc thời gian phơi sáng. Motion blur chỉ thay pixel, không đổi hình học; giữ tên ảnh, kích thước và nhãn bbox.

Kiểm tra RGB uint8 đầu vào, ảnh đầu ra và seed nếu phép tạo blur có yếu tố ngẫu nhiên. Nếu có ảnh kích thước không được thư viện hỗ trợ, xử lý có ghi log; không loại âm thầm hoặc resize ảnh mà quên đổi bbox. Nếu cần resize, dùng cùng phép biến đổi cho cả ảnh gốc và mọi biến thể.

## 6. Luồng xử lý và bàn giao

1. Chốt danh sách ảnh, nhãn và cấu hình dùng chung.
2. B tạo năm bộ blur, kiểm tra tên ảnh/kích thước và giữ nguyên nhãn.
3. C đọc ảnh gốc và blur, tính score cho từng ảnh.
4. B chạy detector/evaluator và xuất prediction cùng metric tổng hợp.
5. C ghép theo `image_id + severity`, phân tích và vẽ plot.
6. A dùng số liệu thật để hoàn thiện slide; cả nhóm kiểm tra kết luận và giới hạn.

**Input B bàn giao cho C:** ảnh hoặc đường dẫn dùng chung; labels; prediction từng ảnh; metric tổng hợp từng điều kiện; cấu hình, phiên bản và log. B gửi vài ảnh mẫu trước, không cần chờ toàn bộ chạy xong.

**Không dùng ảnh có box vẽ sẵn để tính score.** Chữ và cạnh box có thể làm tăng Laplacian variance giả tạo.

## 7. Metric: hiểu đúng và tính đúng

### 7.1. Laplacian variance

Chuyển ảnh sang grayscale, tính Laplacian, lấy phương sai. Baseline có thể dùng OpenCV `cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var()`. Ghi rõ grayscale conversion, kích thước ảnh, ksize và thang pixel 0–255. Score là chỉ số số học phụ thuộc thang cường độ và xử lý ảnh, không có đơn vị vật lý chuẩn.

Giá trị cao thường thể hiện nhiều biến thiên/cạnh; giá trị thấp có thể do blur hoặc cảnh ít texture. Không so sánh score từ hai preprocessing khác nhau. Báo median cùng phân bố hoặc IQR, không chỉ mean.

Trong phân tích cặp, có thể tính tỷ lệ score_blur / score_clean nếu score_clean > epsilon. Tỷ lệ giúp so sánh cùng ảnh nhưng cần ảnh sạch làm tham chiếu; không mặc nhiên sử dụng được khi vận hành mà không có ảnh sạch.

### 7.2. Recall tại ngưỡng cố định

Recall = TP / (TP + FN). Trong phân tích từng ảnh, prediction phải cùng lớp và IoU >=0.5 với ground truth; matching một-một, một prediction không được khớp nhiều GT. Chốt thuật toán matching và kiểm tra bằng ví dụ đơn giản.

Ảnh không có GT thuộc phạm vi đánh giá có recall không xác định: ghi NaN và loại khỏi tương quan recall, không gán 0 hoặc 1 tùy ý. Recall tổng tập tính từ tổng TP/FN; không coi trung bình recall ảnh là cùng một metric. Khi lấy R từ evaluator Ultralytics, ghi protocol của evaluator; không mặc nhiên gọi đó là recall tại conf=0.25 nếu thư viện chọn operating point khác.

### 7.3. mAP50

AP sử dụng precision–recall qua các mức confidence; mAP50 là trung bình AP theo lớp với IoU=0.5. Dùng cùng evaluator và cùng danh sách lớp cho mọi điều kiện.

**Hai đường đo nên tách:** prediction tại conf=0.25 cho recall vận hành/confidence; validation AP dùng ngưỡng confidence thấp phù hợp evaluator để giữ các prediction dùng dựng đường PR. Nếu chỉ chạy val với conf=0.25, ghi rõ đó là đánh giá có cắt prediction theo ngưỡng và có thể làm mất một phần đường PR. Không trộn metric từ hai protocol mà không mô tả.

### 7.4. Mean confidence

Chốt mean trên tất cả prediction được giữ sau threshold/NMS. Có thể báo thêm mean confidence trên TP nếu matching đã có. Nếu không có prediction, mean confidence là NaN và số detection là 0.

Confidence trung bình có thể tăng khi các dự đoán khó biến mất. Không coi confidence trung bình cao là detection tốt nếu recall giảm.

### 7.5. Spearman rho

Chốt đơn vị quan sát trước: per-image hoặc per-severity. Tương quan sáu điểm tổng hợp chỉ là thăm dò rất nhỏ. Nếu dùng nhiều ảnh x nhiều mức, các quan sát cùng ảnh có liên hệ; không coi chúng độc lập để diễn giải p-value hoặc số lượng mẫu.

Nếu chưa kịp matching, chỉ báo tương quan giữa median health score và recall tổng hợp của sáu điều kiện, ghi rõ giới hạn. Nếu score/recall không biến thiên, rho có thể không xác định. Scatter per-image chỉ làm khi thực sự có recall từng ảnh.

## 8. CSV cần chuẩn bị

**image_metrics.csv — một dòng cho một ảnh ở một điều kiện:**

`image_id, severity, image_path, width, height, laplacian_var, n_gt, n_pred, tp, fp, fn, recall_at_025, mean_conf`

C điền health score trước; các cột detection để trống đến khi nhận prediction và tính matching. NaN phải phân biệt với 0. Không điền mAP toàn tập lặp lại vào từng dòng rồi gọi đó là AP của ảnh.

**condition_summary.csv — một dòng cho một điều kiện:**

`severity, n_images, median_laplacian_var, total_gt, total_pred, recall_at_025, map50, mean_conf, device`

Metadata riêng: model, weights, versions, seed, imgsz, conf, NMS IoU, max_det, lớp, preprocessing, evaluator, đường dẫn dataset và command. So sánh suy giảm recall theo điểm phần trăm khi recall ở dạng %, ghi rõ phân biệt với % tương đối.

## 9. Biểu đồ và ảnh demo

1. Plot severity 0–5 so với median Laplacian variance; có thể thêm IQR.
2. Plot severity so với recall và mAP50 nếu đã đo.
3. Scatter health score–recall đúng đơn vị quan sát; ghi rho và số quan sát hợp lệ. Đây là mở rộng, không buộc phải có trong lớp.
4. Một ảnh gốc và các biến thể blur của cùng ảnh, cùng cách vẽ detection.

Không ép các đại lượng khác đơn vị vào một trục y không có nhãn. Không chỉ chọn ảnh ủng hộ giả thuyết: tìm thêm case score giảm nhưng detector còn tốt, hoặc score không thấp nhưng detector thất bại nếu dữ liệu có.

## 10. Phân công và việc bạn làm ngay

| Người | Công việc song song | Sản phẩm |
|---|---|---|
| A — Nguồn/trình bày | Đọc nguồn, ghi input/output/metric/limitation, dựng slide, bổ sung số liệu cuối | Bảng nguồn, slide/report, lời trình bày |
| B — Chạy pipeline | Chuẩn bị dữ liệu, blur, YOLO, export, log và ảnh demo | Script, dataset biến thể, prediction, bảng evaluator |
| C — Bạn, benchmark | Score, CSV schema, matching nếu đủ thời gian, plot, tương quan và nhận xét | CSV, plot, protocol metric, giới hạn |

**C không cần chờ A/B hoàn thành.** Bắt đầu bằng hàm score trên vài ảnh, tạo CSV, viết plot với dữ liệu đo thật sẵn có. Hỏi B một ảnh mẫu, nhãn và prediction để kiểm tra format. Khi B gửi đầy đủ, chạy phân tích cuối. C và B cùng kiểm tra class mapping và tọa độ bbox; A xác nhận các thuật ngữ metric từ nguồn.

## 11. Timeline 120 phút

| Phút | Việc ưu tiên | Checkpoint |
|---|---|---|
| 0–15 | Chốt T1, scope, giả thuyết, metric và format bàn giao | Một bảng thống nhất |
| 15–45 | A đọc nguồn; B setup/blur mẫu; C score/CSV | Ít nhất một ảnh gốc/blur và score |
| 45–70 | Chạy đủ mức trên tập nhỏ trước; xác minh nhãn và ảnh | Benchmark health score chạy được |
| 70–95 | Chạy detector nếu khả thi; C tổng hợp số và plot | Log/CSV/plot thật |
| 95–115 | Chọn failure, nêu limitation và trade-off | Slide/report có kết luận theo dữ liệu |
| 115–120 | Tổng duyệt | Pitch 3–5 phút |

Nếu thiếu thời gian, giảm số ảnh nhưng giữ cùng danh sách ảnh cho sáu điều kiện và báo số thực tế. Bỏ Spearman per-image/ngưỡng trước, không bỏ bằng chứng chạy. Không chờ một phiên YOLO dài đến hết buổi mới tạo slide.

## 12. Ngưỡng cảnh báo và quyết định kỹ thuật

Đề yêu cầu nhận xét khi nào down-weight camera, không yêu cầu nhóm chứng minh một ngưỡng universal. Có thể đề xuất: score thấp kéo dài qua nhiều frame sẽ kích hoạt cờ chất lượng, log frame và cân nhắc giảm mức tin cậy camera; quyết định cụ thể còn cần đánh giá hệ thống và cảm biến khác.

Nếu thực nghiệm ngưỡng, định nghĩa trước thế nào là detection kém, dùng ảnh riêng để chọn ngưỡng, giữ mọi severity của cùng ảnh trong cùng split, rồi kiểm tra trên ảnh giữ riêng. Báo false alarm và missed alarm. Không dùng kết quả cùng tập để vừa chọn vừa xác nhận ngưỡng. Không lấy một giá trị Laplacian tùy ý rồi coi là chuẩn.

Trade-off: ngưỡng nhạy dễ báo nhầm ở cảnh ít texture; ngưỡng lỏng có thể bỏ qua blur. Đề xuất cải tiến: ngưỡng theo bối cảnh, kết hợp exposure/texture hoặc smoothing theo thời gian. Đây là đề xuất, chưa phải kết quả đã triển khai.

## 13. Report/slide và lời trình bày

Một trang/slide hoặc vài slide ngắn, theo mẫu PDF:

| Mục | Nội dung |
|---|---|
| Problem | ADAS camera, motion blur, hệ quả lên perception |
| Method | Blur tổng hợp, Laplacian variance, detector cố định nếu có |
| Benchmark | Số ảnh thực tế, sáu điều kiện, cấu hình, metric và số đo |
| Failure case | Ảnh đo thực tế và giải thích score/detector còn sai ở đâu |
| Engineering decision | Cảnh báo chất lượng/down-weight có điều kiện, dữ liệu cần kiểm tra thêm |

Pitch 3–5 phút: khoảng 40 giây bài toán, 50 giây phương pháp, 70 giây kết quả, 40 giây failure, 40 giây trade-off/đề xuất. Điền số liệu sau khi chạy; nếu chưa đo detector thì nói rõ.

## 14. Nguồn đọc và cách trích

- PDF đề bài do người dùng cung cấp: yêu cầu, rubric và phạm vi T1. Yêu cầu tìm paper/repo mới nhất nằm ở workflow; hai paper 2019 dưới đây là nguồn nền tảng, không được gọi là mới nhất. A nên bổ sung một nguồn gần đây liên quan nếu tìm và đọc được; không cần đổi baseline giữa buổi chỉ để chạy model mới.
- Michaelis et al., Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming: https://arxiv.org/abs/1907.07484
- Hendrycks & Dietterich, Benchmarking Neural Network Robustness to Common Corruptions and Perturbations: https://arxiv.org/abs/1903.12261
- Repo imagecorruptions, xem README/dependency và license: https://github.com/bethgelab/imagecorruptions
- COCO128: https://docs.ultralytics.com/datasets/detect/coco128/
- Validation Ultralytics: https://docs.ultralytics.com/modes/val/

A ghi đúng metric, dữ liệu và số liệu khi đọc paper. Thí nghiệm một corruption trên COCO128 là lấy cảm hứng/tái hiện một phần ý tưởng, không tái lập toàn bộ benchmark paper. Kết luận của nhóm chỉ dùng log/CSV đã chạy; nguồn chỉ dùng để giải thích phương pháp và bối cảnh.

## 15. Checklist trước khi nộp

- [ ] Ghi rõ T1, ADAS, camera và motion blur.
- [ ] Có ảnh gốc cùng 3–5 mức suy giảm, cùng danh sách và nhãn.
- [ ] Có metric thật, config, phiên bản và bằng chứng chạy.
- [ ] Không gọi COCO128 là bộ lái xe hoặc test độc lập.
- [ ] Không khẳng định score/confidence/recall luôn giảm.
- [ ] Phân biệt recall vận hành và mAP theo evaluator.
- [ ] Spearman/scatter dùng đúng đơn vị quan sát nếu có.
- [ ] Có một failure và một đề xuất cải tiến.
- [ ] Phân biệt kết luận paper và kết luận nhóm.
- [ ] Ngưỡng cảnh báo được gọi là đề xuất nếu chưa kiểm định.
- [ ] Report/slide đủ ngắn để trình bày 3–5 phút.

**Việc đầu tiên của bạn:** thống nhất CSV với B, tính score trên một ảnh gốc và một ảnh blur, xác minh số và ảnh đầu ra trước khi chạy hàng loạt.
