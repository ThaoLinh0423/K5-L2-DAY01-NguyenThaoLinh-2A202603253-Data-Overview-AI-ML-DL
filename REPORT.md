# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 03 / Pytorch / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: 
"class_id": 468,
"class_name": "cab",
"score": 0.510915,
"rank" = 1,
"taxonomy_name": "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh . Vì taxonomy_name = "ImageNet-1K" cho biết kết quả thuộc hệ thống phân loại ImageNet-1K, vốn được sử dụng cho bài toán Image Classification.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp đi kèm với checkpoint, danh sách tên lớp của ImageNet-1K được truy xuất thông qua metadata của model khi nạp checkpoint bằng thư viện Ultralytics
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Vì mỗi lớp có 1 class_id riêng, giúp máy tính xử lý, lưu trữ và truy xuất nhanh hơn, giảm việc phụ thuộc vào tên lớp khi xử lý dữ liệu. Còn taxanomy_name giúp xác định lớp đang được hiểu theo bộ nhãn, tránh nhầm lẫn khi kêt hợp nhiều bộ dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Nếu trong ảnh có nhiều vật, nhiều chủ đề, guideline cần nói rõ đối tượng chủ thể của bức ảnh ngoài ra còn dựa trên diện tích, vị trí và mức độ nổi bật hoặc ý nghĩa của bức ảnh hoặc yêu cầu của sản phẩm để người dán nhãn chọn 1 lớp.
6. Vì sao không được lấy `score` của model làm ground truth?
Bởi bản chất và mục ssichs hoàn toàn khác nhau. Score là model tự đánh giá mức độ tin tưởng vào dự đoán được tạo ra bởi AI nhằm xếp hạng hoặc lọc dự đoán. Đối với ground truth là nhãn đúng được con người gán nhãn hoặc được nguồn dữ liệu chuẩn đã được xác minh và dùng để đánh giá độ chính xác của model
## 2. Phát hiện vật thể 

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: 
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box bao quanh đối tượng person (người).
- Góc trên-trái: tọa độ (385.33, 69.24).
- Góc dưới-phải: tọa độ (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
Hộp bắt đầu ở vị trí cách mép trái ảnh khoảng 385 pixel và cách mép trên khoảng 69 pixel. Hộp kéo xuống và sang phải, kết thúc tại tọa độ (498.92, 348.92).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20 giữ lại nhiều 17 vật thể.
Ngưỡng 0.60 chỉ giữ lại 6 vật thể.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, hệ thống giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Reviewer cần:
- Kiểm tra từng bounding box có thực sự bao quanh đúng vật thể không.
- Loại bỏ các hộp dự đoán sai (false positives).
- Kiểm tra các hộp bị trùng lặp.
- Xác nhận class của từng vật thể.
- Chỉnh sửa những bounding box có vị trí hoặc kích thước chưa chính xác.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Bounding box phải bao sát toàn bộ phần vật thể nhìn thấy được trong ảnh, không bao gồm phần nền xung quanh. Góc trên-trái và góc dưới-phải phải được đặt chính xác theo ranh giới vật thể. Nếu vật thể bị che khuất hoặc cắt khỏi khung hình, chỉ đánh dấu phần thực sự nhìn thấy, không tự suy đoán phần bị khuất. Nếu như vật thể bị khuất dưới 30% thì sẽ bỏ qua không cần bounding box vật thể.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trên ảnh kitchen có bát bị căt mép. Tôi sẽ hỏi lại với Lab Coach để biết được bị che khoảng tầm bao nhiêu thì sẽ được tính là box lại vật thể sau đó viết thành quy tắc trong guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,      
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      ...
    ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box chỉ thể hiện vùng chữ nhật bao quanh vật thể. Polygon mô tả đường biên thực tế của vật thể bằng nhiều điểm tọa độ. Vì vậy, polygon thể hiện hình dạng chi tiết hơn, kể cả những vùng lồi lõm hoặc không vuông vắn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng riêng biệt trong ảnh, kể cả khi chúng cùng một class, không đại diện cho không đại diện cho loại vật thể, không dùng để theo dõi cùng một vật thể qua nhiều frame hoặc video. Nó chỉ định danh một instance trong phạm vi dữ liệu segmentation tương ứng.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ polygon theo sát đường biên của phần vật thể nhìn thấy được, đặt các điểm đủ gần để mô tả chính xác hình dạng, không bao gồm nền hoặc vật thể bên cạnh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trên ảnh kitchen, có phần biên của persson bị tô thêm cả biên của chiếc chảo.
Nếu tình huống đã có quy tắc rõ ràng trong guideline thì áp dụng guideline. Nếu không xác định được ranh giới giữa hai vật thể hoặc có nhiều cách diễn giải hợp lý thì escalation với Lab Coach để thống nhất cách gán nhãn.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | nhiều nhãn class cho toàn bộ ảnh. Đơn vị nhãn là ảnh.  | Có thể nhầm class khi ảnh chứa nhiều loại vật thể hoặc ngữ cảnh không rõ ràng. Cần kiểm tra ảnh có thực sự thuộc class được dự đoán không.   |Xem toàn bộ ảnh và gán đúng nhãn theo guideline.  |Kiểm tra class có phù hợp nội dung ảnh không, có thiếu hoặc thừa nhãn không. |
| Phát hiện vật thể |Bounding box dạng [x1, y1, x2, y2], kèm class_name. Đơn vị nhãn là từng vật thể.  |Trên kitchen, threshold 0.20 có 17 prediction, trong khi 0.60 chỉ có 6. Điều này cho thấy số hộp thay đổi theo ngưỡng. Các hộp có thể bị trùng, sai class hoặc bỏ sót vật thể.|Vẽ box sát vật thể, chọn đúng class và tuân thủ quy tắc với vật bị che khuất hoặc cắt mép.  |Kiểm tra vị trí, kích thước box, class, hộp trùng lặp, false positives và false negatives  |
| Instance segmentation |Polygon hoặc mask cho từng instance, kèm instance_id và class_name. Đơn vị nhãn là từng instance/vật thể riêng biệt.  |Khó xác định biên khi hai vật chạm nhau, bị che khuất hoặc đường viền không rõ. Polygon có thể không sát biên thực tế.  |Vẽ polygon theo đường biên phần vật thể nhìn thấy, tách riêng từng instance kể cả khi cùng class.  | Kiểm tra độ chính xác của mask, đường biên, vùng bị bỏ sót, vùng dư và việc phân biệt các instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu được cấp phép trong phạm vi bài lab, không tự ý sao chép, chia sẻ, tải lên nơi không được phép hoặc tiết lộ thông tin nhạy cảm.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Khi gặp dữ liệu không đúng phạm vi: Tôi sẽ dừng xử lý và báo ngay cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn, không tự ý tiếp tục hoặc chia sẻ dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
