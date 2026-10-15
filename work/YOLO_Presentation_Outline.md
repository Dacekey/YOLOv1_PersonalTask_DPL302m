# YOLO: nền tảng YOLOv1 và quá trình phát triển của họ mô hình

**Bản nháp nội dung thuyết trình — DPL302m Personal Task**  
**Ngôn ngữ:** tiếng Việt, giữ các thuật ngữ kỹ thuật cần thiết.  
**Mô hình trọng tâm:** YOLOv1; paper *You Only Look Once: Unified, Real-Time Object Detection*.  
**Khung gợi ý:** 16 slide, khoảng 15–18 phút; slide demo chờ bổ sung.  
**Ngày tổng hợp và tra cứu:** 13/10/2026.

## 1. Mạch thuyết trình

**Bối cảnh phát hiện vật thể: từ đặc trưng thủ công đến CNN** → **YOLOv1: phát hiện thống nhất trong một lần dự đoán** → **YOLO Family: các hướng phát triển từ cùng một nền tảng**.

Câu hỏi dẫn dắt:

> Làm thế nào để xác định vật thể và vị trí của chúng đủ nhanh cho ứng dụng thời gian thực?

YOLOv1 là nội dung chính. Phần bối cảnh giải thích vì sao cách tiếp cận này có ý nghĩa; phần family giải thích các thế hệ sau thay đổi gì để xử lý hạn chế của baseline.

| Phần | Slide | Nội dung |
| --- | --- | --- |
| Mở đầu | 1 | Bài toán và câu hỏi dẫn dắt |
| Bối cảnh | 2–3 | Đặc trưng thủ công, CNN và các hướng detection trước/cùng thời YOLO |
| YOLOv1 | 4–12 | Theo yêu cầu môn học; bổ sung training/loss để giải thích cơ chế |
| YOLO Family | 13–15 | Các mốc kỹ thuật và các nhánh nghiên cứu |
| Kết thúc | 16 | Đóng góp của baseline và bài học xuyên suốt |

## 2. Nội dung rút gọn theo slide

### Slide 1 — YOLO: từ bài toán detection đến xử lý thời gian thực

**Đưa lên slide:**

- Object detection trả lời: **vật thể gì, ở đâu?**
- Đầu ra: bounding box, nhãn lớp và điểm dự đoán.
- Thách thức: cân bằng độ chính xác và tốc độ.
- Trọng tâm: YOLOv1; mở rộng bằng bối cảnh lịch sử và YOLO Family.

**Câu mở đầu:** “Một hệ thống nhìn thấy đúng nhưng phản hồi quá chậm vẫn khó dùng cho video hoặc robot. YOLOv1 nghiên cứu cách đơn giản hóa quá trình dự đoán để giải quyết yêu cầu này.”

### Slide 2 — Detection trước CNN: quét ảnh và thiết kế đặc trưng

**Đưa lên slide:**

| Hướng tiêu biểu | Cách làm | Điểm cần nhớ |
| --- | --- | --- |
| Viola–Jones, 2001 | Đặc trưng kiểu Haar, integral image, AdaBoost và cascade | Loại background sớm; thành công với face detection |
| HOG + SVM, 2005 | Histogram hướng gradient; phân loại cửa sổ trên image pyramid | Mô tả hình dáng, nổi bật trong pedestrian detection |
| DPM, 2008/2010 | Root và các bộ phận có thể dịch chuyển; latent SVM | Mô hình hóa biến dạng và bố cục vật thể |

**Ý nói:** “Detection đã có các giải pháp nhanh trước YOLO, đặc biệt cho bài toán chuyên biệt như khuôn mặt. Thách thức tiếp theo là học biểu diễn mạnh hơn cho nhiều lớp và cảnh phức tạp.”

**Nguồn:** [Viola–Jones, CVPR 2001](https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/papers/viola_cvpr2001.pdf), [Dalal–Triggs, CVPR 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), [DPM — trang tác giả và các paper 2008/2010](https://www.rossgirshick.info/latent/).

### Slide 3 — CNN cải thiện biểu diễn, nhưng pipeline detection còn nhiều bước

**Đưa lên slide:**

| Mốc | Thay đổi chính | Nút thắt còn lại |
| --- | --- | --- |
| R-CNN, 2013/2014 | Region proposals → CNN cho từng vùng → phân loại và hồi quy box | Lặp tính toán trên nhiều crops; training nhiều bước |
| SPP-net, 2014 | Chia sẻ feature map và pooling vùng về kích thước cố định | Vẫn dùng proposals; quy trình training còn phức tạp |
| Fast R-CNN, 04/2015 | Shared CNN + RoI pooling; học classification/box regression cùng nhau | External proposal generation vẫn là nút thắt |

**Nói thêm ngắn:** OverFeat đã dùng CNN với sliding-window đa tỉ lệ; MultiBox đã học dự đoán các box đề xuất. Vì vậy, lịch sử detection có nhiều hướng song song, không chỉ R-CNN. [OverFeat](https://arxiv.org/abs/1312.6229), [MultiBox](https://research.google/pubs/scalable-object-detection-using-deep-neural-networks/).

**Câu chuyển:** “Các giải pháp đang tìm cách giảm tính toán lặp hoặc cải thiện proposal. YOLOv1 đặt câu hỏi khác: liệu có thể dự đoán trực tiếp box và lớp từ toàn ảnh trong một lần forward?”

**Nguồn:** [R-CNN](https://arxiv.org/abs/1311.2524), [SPP-net](https://arxiv.org/abs/1406.4729), [Fast R-CNN](https://arxiv.org/abs/1504.08083).

### Slide 4 — YOLOv1: paper gốc và động lực nghiên cứu

**Requirement:** Read the original paper; Problem motivation.

**Đưa lên slide:**

- Tác giả: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi.
- Preprint năm 2015; công bố tại CVPR 2016.
- Động lực: giảm sự phân mảnh và tính toán lặp của pipeline detection.
- Mục tiêu: một mạng học đồng thời vị trí và lớp, phục vụ real-time detection.

**Ý nói:** “Đóng góp chính của YOLOv1 nằm ở cách tổ chức bài toán detection thành một bài toán dự đoán thống nhất.”

**Nguồn:** [Paper local](../YOLO_Paper/V1.pdf); [trang paper gốc và lịch sử công bố](https://arxiv.org/abs/1506.02640).

### Slide 5 — Ý tưởng chính: một mạng, một lưới trách nhiệm

**Requirement:** Main idea of the model.

**Đưa lên slide:**

- Mạng nhận toàn ảnh kích thước `448 × 448`.
- Chia trách nhiệm theo lưới `7 × 7`; ô chứa **tâm vật thể** chịu trách nhiệm.
- Mỗi ô dự đoán 2 box và một vector 20 xác suất lớp dùng chung.
- Đầu ra VOC: `7 × 7 × (2 × 5 + 20) = 7 × 7 × 30`.

**Ý nói:** “Lưới quyết định ô nào học vật thể, không giới hạn box nằm trong ô đó. Hai box predictors cũng không có hai bộ phân loại độc lập.”

**Hình:** có thể dùng [sơ đồ grid responsibility](../LaTeX/figures/grid_representation.pdf).

### Slide 6 — Kiến trúc YOLOv1

**Requirement:** Architecture diagram.

**Đưa lên slide:**

```text
448 × 448 × 3
      ↓
24 convolutional layers
      ↓
7 × 7 × 1024 feature representation
      ↓
Fully connected: 4096
      ↓
Fully connected: 1470 → reshape 7 × 7 × 30
```

- Convolution `1 × 1`: phối hợp/giảm số kênh trung gian.
- Convolution `3 × 3`: học đặc trưng không gian.
- Leaky ReLU ở hidden layers; output layer tuyến tính.

**Ý nói:** “Feature extraction học biểu diễn ảnh; fully connected layers biến biểu diễn này thành các dự đoán toàn cục. Downsampling giúp trích xuất ngữ nghĩa nhưng làm mất chi tiết định vị.”

**[Chờ bổ sung hình kiến trúc chính thức/phù hợp để trình chiếu.]** Giữ trạng thái này đồng bộ với báo cáo hiện tại.

### Slide 7 — Từ output tensor đến detections

**Bổ trợ cho main idea.**

**Đưa lên slide:**

- Một box: `(x, y, w, h, confidence)`.
- `x, y`: offsets theo ô; `w, h`: chuẩn hóa theo kích thước ảnh.
- Confidence target: `P(Object) × IoU`.
- Điểm lớp khi suy luận: `P(Class | Object) × predicted confidence`.
- 98 box slots → giải mã → lọc điểm → NMS → detections.

**Ý nói:** “IoU với ground truth dùng để tạo mục tiêu khi train. Khi inference, mạng dự đoán confidence; không dùng ground truth. Một forward pass cũng không có nghĩa bỏ NMS.”

**Hình:** [pipeline suy luận](../LaTeX/figures/inference_pipeline.pdf).

### Slide 8 — Mạng được huấn luyện như thế nào?

**Bổ trợ cho main idea và architecture.**

**Đưa lên slide:**

- Pretrain 20 convolutional layers trên ImageNet, input `224 × 224`.
- Thêm 4 convolutional layers và 2 FC layers; train detection ở `448 × 448`.
- Loss kết hợp tọa độ, confidence và lớp; predictor có IoU cao nhất học tọa độ.
- `λ_coord = 5`, `λ_noobj = 0.5`; dùng căn bậc hai kích thước box trong loss.

**Công thức rút gọn:**

\[
L=5L_{xy}+5L_{\sqrt{w},\sqrt{h}}+L_{conf,obj}+0.5L_{conf,noobj}+L_{class}.
\]

**Ý nói:** “Các trọng số giúp coordinate loss đủ ảnh hưởng và tránh để quá nhiều ô background lấn át tín hiệu từ vật thể. Đây vẫn là squared-error loss, chưa trực tiếp tối ưu AP.”

**Hình:** [sơ đồ loss và responsibility](../LaTeX/figures/loss_assignment.pdf). Không cần đọc toàn bộ lịch learning rate khi trình bày ngắn.

### Slide 9 — Đổi mới so với các detector trước đó

**Requirement:** Key innovations compared to previous models.

**Đưa lên slide:**

| Detector | Đơn vị dự đoán | Tính toán đặc trưng |
| --- | --- | --- |
| R-CNN | Region proposal | CNN riêng cho từng crop |
| Fast R-CNN | Region proposal | Shared feature map và RoI pooling |
| YOLOv1 | Grid cell và box predictors | Một CNN trên toàn ảnh; không cần external proposals |

- Box và lớp được học trong một mạng dự đoán thống nhất.
- Dùng ngữ cảnh toàn ảnh và ưu tiên tốc độ.

**Ý nói:** “Không nên nói mọi detector trước YOLO đều chạy CNN riêng cho từng vùng: Fast R-CNN đã chia sẻ đặc trưng. Điểm khác biệt là cách tạo ra predictions.”

**Nguồn đối chiếu:** [R-CNN](https://arxiv.org/abs/1311.2524), [Fast R-CNN](https://arxiv.org/abs/1504.08083), [YOLOv1](../YOLO_Paper/V1.pdf).

### Slide 10 — Kết quả, ưu điểm và giới hạn

**Requirement:** Advantages and limitations.

**Đưa lên slide:**

| Model | VOC 2007 mAP (%) | FPS trong paper |
| --- | --- | --- |
| YOLO | 63.4 | 45 |
| Fast YOLO | 52.7 | 155 |

- **Ưu điểm:** nhanh; pipeline gọn; ít background false positives hơn Fast R-CNN trong phân tích paper.
- **Hạn chế:** localization errors; vật thể nhỏ/clustered; class vector dùng chung; loss chưa khớp trực tiếp AP.
- Fast YOLO là model nhỏ hơn, không phải chế độ chạy nhanh của cùng mạng.

**Ý nói:** “Đánh đổi của v1 là tốc độ tốt nhưng box chưa luôn chính xác. Lưới thô và biểu diễn giới hạn cũng gây khó khăn khi nhiều tâm vật thể nằm gần nhau.”

**Nguồn:** [Paper v1, phần kết quả và limitations](../YOLO_Paper/V1.pdf). Đây là số được tác giả báo cáo, không phải benchmark tự thực hiện; không so trực tiếp với COCO AP của các phiên bản sau.

### Slide 11 — Ứng dụng điển hình

**Requirement:** Typical applications.

**Đưa lên slide:**

- Video/webcam cần cập nhật detections thường xuyên.
- Robot và assistive vision cần phản hồi nhanh.
- Phân tích cảnh giao thông: quan tâm đồng thời tốc độ và localization.
- Với VOC configuration, đầu ra là box cho 20 lớp; tracking, masks hoặc distance estimation cần thành phần bổ sung.

**Ý nói:** “Đây là các hướng ứng dụng phù hợp với động lực real-time, không phải khẳng định paper đã kiểm chứng mọi hệ thống triển khai.”

### Slide 12 — Demo / implementation example

**Requirement:** Simple demo / implementation example.

**[Để trống, bổ sung sau khi có implementation và tài liệu phù hợp.]**

**[Chờ ảnh đầu vào và ảnh inference thực tế.]**

Không đưa kết quả mô phỏng trước đây vào slide này như một demo đã hoàn thành. Nếu trình bày trước khi bổ sung, chỉ nêu trạng thái đang chuẩn bị; chưa có số liệu inference hoặc accuracy tự đo.

### Slide 13 — Từ YOLOv1 đến v7: cải thiện biểu diễn và hiệu quả hệ thống

**Đưa lên slide:**

| Mốc | Cải tiến cần nhớ | Liên hệ với baseline |
| --- | --- | --- |
| v2 | Anchors, dimension clustering, BN | Recall và định vị ổn định hơn |
| v3 | Residual Darknet-53; ba detection scales | Kết hợp ngữ nghĩa và chi tiết ở nhiều kích thước |
| v4 | CSP + SPP + PAN; BoF/BoS | Phối hợp kiến trúc và training |
| v5 | Pipeline PyTorch; nhiều kích thước model | Train/export/deploy thuận tiện |
| v6 | Re-parameterization; anchor-free decoupled head | Hiệu quả triển khai công nghiệp |
| v7 | E-ELAN, scaling, auxiliary supervision | Tăng hiệu quả học và mở rộng model |

**Ý nói:** “Các phiên bản này lần lượt làm rõ rằng chất lượng không chỉ đến từ backbone lớn hơn: representation, training và hardware đều ảnh hưởng kết quả.”

**Nguồn:** [Bản tổng hợp có dẫn paper và mã nguồn](YOLO_DevProcess/YOLO_Development_Process_V1_to_V12.md).

### Slide 14 — Từ v8 đến v12: head, gradient, NMS và attention

**Đưa lên slide:**

| Mốc | Câu hỏi kỹ thuật | Giải pháp nổi bật |
| --- | --- | --- |
| v8 | Bỏ phụ thuộc vào predefined anchor boxes? | Anchor-free split head; C2f và DFL |
| v9 | Làm sao tạo gradient có thông tin tốt hơn? | PGI và GELAN |
| v10 | Làm sao giảm prediction trùng và bỏ NMS? | Consistent dual assignments; one-to-one inference |
| YOLO11 | Cải thiện feature modeling hiệu quả? | C3k2 và C2PSA |
| v12 | Attention có thể đáp ứng real-time? | Area Attention và R-ELAN |

**Ý nói:** “Đây là các câu hỏi khác nhau. Anchor-free không đồng nghĩa NMS-free; YOLO11 và v12 không mặc nhiên kế thừa cách bỏ NMS của v10.”

**Hình tổng quan để minh họa slide 13–14:**

![Tổng quan phát triển YOLO v1–v12](<../resources/yolo/The Evolution of YOLO v1–v12.png>)

**Nguồn kỹ thuật:** [Bản tổng hợp v1–v12](YOLO_DevProcess/YOLO_Development_Process_V1_to_V12.md), [tài liệu chính thức v8](https://docs.ultralytics.com/models/yolov8/), [tài liệu chính thức YOLO11](https://docs.ultralytics.com/models/yolo11/).

### Slide 15 — YOLO Family phát triển theo nhiều nhánh

**Đưa lên slide:**

- **Ultralytics:** v5 → v8 → YOLO11; hệ sinh thái và khả năng triển khai.
- **Wang và cộng sự:** v7 → v9; aggregation, training và gradient.
- **Meituan:** v6/v6 v3.0; hiệu quả công nghiệp và hardware.
- **Các hướng khác:** v10 tối ưu NMS-free; v12 tập trung attention.

**Ý nói:** “Số phiên bản giúp định danh model, nhưng không tạo thành một cây kế thừa duy nhất. Một phiên bản mới có thể phát triển từ một nhánh trước đó thay vì model mang số liền trước.”

![Các nhánh nghiên cứu của YOLO Family](<../resources/yolo/YOLO Family Branching Evolution Roadmap.png>)

**Cách dùng hình:** tập trung vào v1–v12. Các nút YOLOR, YOLOX, YOLO-NAS, YOLO-World, YOLOE và YOLO26 chỉ thể hiện phạm vi rộng của family; không cần triển khai thành phần trình bày mới. Hình là tài liệu minh họa trong workspace; nguồn xác nhận kỹ thuật là paper/documentation được dẫn trong bản tổng hợp.

### Slide 16 — Kết luận

**Đưa lên slide:**

- YOLOv1 chứng minh giá trị của **unified, full-image prediction** cho real-time detection.
- Thiết kế lưới đơn giản tạo ra cả lợi thế tốc độ và giới hạn biểu diễn.
- YOLO Family mở rộng phạm vi tối ưu: features → training → deployment → attention.

**Câu kết:** “Giá trị nền tảng của YOLOv1 là cách nhìn bài toán. Các thế hệ sau giữ mục tiêu cân bằng tốc độ và chất lượng, nhưng thay đổi nhiều cách để đạt được mục tiêu đó.”

## 3. Ghi chú chuẩn bị: survey ngắn về bối cảnh trước YOLO

Phần này là tài liệu tra nhanh cho người trình bày, không cần đưa toàn bộ lên slide.

| Model / phương pháp | Mốc | Cơ chế | Vai trò trong câu chuyện |
| --- | --- | --- | --- |
| Viola–Jones | 2001 | Integral image, boosting, cascade | Real-time detection chuyên biệt đã có trước YOLO |
| HOG + SVM | 2005 | Gradient descriptor + window classifier | Giai đoạn thiết kế đặc trưng thủ công |
| DPM | CVPR 2008; TPAMI 2010 | Root/parts, deformation cost, latent SVM | Tăng khả năng biểu diễn biến dạng |
| Selective Search | IJCV 2013 | Gộp vùng theo cấu trúc ảnh để tạo proposals | Tách bước tìm vị trí ứng viên khỏi nhận dạng; không phải detector CNN hoàn chỉnh |
| R-CNN | Preprint 2013; CVPR 2014 | Proposals + CNN features + SVM + box regression | CNN tăng chất lượng biểu diễn nhưng còn tính toán lặp |
| OverFeat | Preprint 2013; ICLR 2014 | CNN sliding-window đa tỉ lệ; localization regression | Hướng dense scanning song song với proposal-based detection |
| MultiBox | CVPR 2014 | Dự đoán một tập class-agnostic boxes và objectness scores | Box prediction bằng mạng đã được nghiên cứu trước YOLO |
| SPP-net | ECCV 2014 | Shared feature map + spatial pyramid pooling | Giảm lặp convolution và hỗ trợ pooling vùng |
| Fast R-CNN | Preprint 04/2015; ICCV 2015 | RoI pooling + multi-task classification/regression | Tăng hiệu quả nhưng còn external proposals |
| Faster R-CNN | Preprint 04/06/2015; NeurIPS 2015 | RPN chia sẻ features với detector | Học proposal generation; hướng phát triển cùng thời kỳ YOLOv1 |

Nguồn tương ứng: [Viola–Jones](https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/papers/viola_cvpr2001.pdf), [HOG](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), [DPM](https://www.rossgirshick.info/latent/), [Selective Search — trang tác giả](https://www.koen.me/research/pub/2013/UijlingsIJCV2013/), [R-CNN](https://arxiv.org/abs/1311.2524), [OverFeat](https://arxiv.org/abs/1312.6229), [MultiBox — Google Research](https://research.google/pubs/scalable-object-detection-using-deep-neural-networks/), [SPP-net](https://arxiv.org/abs/1406.4729), [Fast R-CNN](https://arxiv.org/abs/1504.08083), [Faster R-CNN](https://arxiv.org/abs/1506.01497).

### Các điểm cần giữ đúng khi kể lịch sử

- **Phân biệt preprint với conference:** YOLOv1 ra preprint ngày **08/06/2015**, rồi công bố tại CVPR 2016. Faster R-CNN ra preprint ngày **04/06/2015**: nên nói cùng thời kỳ, không dựng quan hệ YOLO ra đời để thay thế riêng Faster R-CNN. [YOLOv1](https://arxiv.org/abs/1506.02640), [Faster R-CNN](https://arxiv.org/abs/1506.01497).
- **`Object-Detection.txt` là danh sách gợi ý, không phải timeline:** Mask R-CNN xuất hiện năm **2017**, DETR năm **2020**, nên không nằm trong phần “trước YOLOv1”. [Mask R-CNN](https://arxiv.org/abs/1703.06870), [DETR](https://arxiv.org/abs/2005.12872).
- **Không nói YOLO là detector đầu tiên biết hồi quy box hoặc real-time:** OverFeat/MultiBox đã học định vị; Viola–Jones đã có face detection thời gian thực. Điểm cần nhấn mạnh là cách YOLOv1 kết hợp dự đoán box và lớp trong một mạng toàn ảnh. [OverFeat](https://arxiv.org/abs/1312.6229), [MultiBox](https://research.google/pubs/scalable-object-detection-using-deep-neural-networks/), [Viola–Jones](https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/papers/viola_cvpr2001.pdf), [paper v1](../YOLO_Paper/V1.pdf).
- **Lưu ý hình timeline:** ô v3 trong hình đang ghi scale nhỏ `52 × 32`; với input `416 × 416`, phải là **`52 × 52`**. Khi dùng slide, sửa nhãn này hoặc giải thích đúng bằng lời. Ba scales là `13 × 13`, `26 × 26`, `52 × 52`; chúng thay đổi theo input resolution. [Paper v3](../YOLO_Paper/V3.pdf).

## 4. Đối chiếu requirement và tài liệu làm việc

| Requirement môn học | Slide |
| --- | --- |
| Read the original paper | 4 và trích dẫn xuyên suốt phần v1 |
| Problem motivation | 4, được chuẩn bị bằng 2–3 |
| Main idea of the model | 5, 7 và 8 |
| Architecture diagram | 6 — nội dung có sẵn, hình chờ bổ sung |
| Key innovations compared to previous models | 9 |
| Advantages and limitations | 10 |
| Typical applications | 11 |
| Simple demo / implementation example | 12 — để trống theo trạng thái báo cáo hiện tại |

Tài liệu để mở rộng khi chuẩn bị slide: [requirement](../Subject_Requirement.txt), [ghi chú YOLOv1](YOLOv1_Detailed_Survey_Report_Notes.md), [quá trình v1–v12](YOLO_DevProcess/YOLO_Development_Process_V1_to_V12.md), [report LaTeX hiện tại](../LaTeX/main.tex).

Với thời lượng ngắn, rút gọn slide 8 thành ba ý về pretraining, loss và responsibility; dùng hai hình family để định hướng rồi chỉ nói các mốc chính. Slide 12 vẫn chờ tài liệu thực tế, không tạo số liệu thay thế.
