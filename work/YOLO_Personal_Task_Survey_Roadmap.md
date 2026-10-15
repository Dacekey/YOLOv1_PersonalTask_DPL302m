# YOLO Personal Task — Survey Roadmap & Study Map

> **Mục tiêu:** xây dựng một bài viết theo phong cách **survey** về sự tiến hóa kiến trúc của họ YOLO, đủ chuyên nghiệp để làm nền cho Personal Task và phần thuyết trình môn học.  
> **Phạm vi:** ưu tiên **architectural evolution** và các **milestone quan trọng**, không cố gắng liệt kê mọi version YOLO theo thứ tự số.

---

## 1. Định hướng tổng thể

### 1.1. Câu hỏi trung tâm

Thay vì trình bày theo kiểu:

> YOLOv1 có gì → YOLOv2 có gì → YOLOv3 có gì → ...

nên đặt câu hỏi trung tâm:

> **How has the YOLO architecture evolved through major architectural milestones for real-time object detection?**

Từ đó, bài survey tập trung trả lời:

1. **YOLO ban đầu giải quyết vấn đề gì?**
2. **Kiến trúc YOLOv1 hoạt động như thế nào?**
3. **Những hạn chế nào của các thế hệ trước tạo động lực cho thế hệ sau?**
4. **Backbone, Neck, Head, prediction mechanism và training strategy đã thay đổi ra sao?**
5. **Các xu hướng lớn của YOLO hiện đại là gì?**
6. **Những cải tiến đó đánh đổi gì về accuracy, speed, complexity và deployment?**

---

## 2. Yêu cầu của môn học

Yêu cầu tham khảo từ môn học:

- **Read the original paper**
- **Problem motivation**
- **Main idea of the model**
- **Architecture diagram**
- **Key innovations compared to previous models**
- **Advantages and limitations**
- **Typical applications**
- **Simple demo / implementation example**

### Cách tích hợp vào bài survey

Không nên biến tám mục trên thành tám phần rời rạc cho **mọi** phiên bản YOLO.

Thay vào đó:

- Dùng **YOLOv1** làm paper gốc để đáp ứng sâu yêu cầu *Read the original paper*.
- Với các version sau, tập trung vào:
  - vấn đề còn tồn tại,
  - thay đổi kiến trúc,
  - innovation mới,
  - trade-off mới.
- Gom phần **advantages / limitations / applications / demo** thành các section tổng hợp ở cuối.

Như vậy vừa đáp ứng yêu cầu môn học, vừa giữ được cấu trúc của một bài survey.

---

# 3. Scope của bài survey

## 3.1. Không nên chia YOLO chỉ dựa trên số version

Một điểm rất quan trọng:

> **YOLO không còn là một lineage duy nhất được phát triển tuần tự bởi cùng một nhóm nghiên cứu.**

Ví dụ:

- YOLOv1–v3: Joseph Redmon và cộng sự.
- YOLOv4: Bochkovskiy, Wang, Liao.
- YOLOv5 / YOLOv8 / YOLO11: Ultralytics.
- YOLOv6: Meituan.
- YOLOv7 / YOLOv9: Wang, Bochkovskiy, Liao / WongKinYiu lineage.
- YOLOv10: một nhóm nghiên cứu khác, tập trung vào end-to-end detection.
- YOLOv12: hướng attention-centric.

Vì vậy, **không nên** mặc định:

```text
v1 → v2 → v3 → v4 → v5 → v6 → v7 → ...
```

là một chuỗi kế thừa trực tiếp duy nhất.

### Chiến lược chính

Sử dụng:

> **Architectural Milestones + Representative Versions**

thay vì:

> **Version-by-Version Enumeration**

---

# 4. Map tiến hóa kiến trúc đề xuất

```text
YOLOv1
│
│  Unified one-stage detection
│  Detection as regression
│
▼
YOLOv2 / YOLOv3
│
│  Anchor-based detection
│  Stronger backbone
│  Multi-scale prediction
│
▼
YOLOv4
│
│  CSP-based backbone
│  SPP + PAN
│  Stronger feature aggregation
│
▼
YOLOv5 / YOLOv7
│
│  Efficient implementation
│  Better aggregation / scaling / training
│
▼
YOLOv8
│
│  Anchor-free
│  Decoupled prediction design
│
▼
YOLOv9
│
│  GELAN
│  Programmable Gradient Information
│
▼
YOLOv10
│
│  End-to-end detection
│  NMS-free direction
│
▼
YOLO11 / YOLOv12
   │
   │  Modern block design
   │  Attention-centric direction
   ▼
Modern YOLO
```

> Đây là **conceptual map**, không phải sơ đồ lineage chính thức.

---

# 5. Các milestone chính nên tập trung

## Milestone A — YOLOv1: Foundation

### Paper

**You Only Look Once: Unified, Real-Time Object Detection**  
Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi — CVPR 2016

- Paper: https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html

### Mức ưu tiên

**BẮT BUỘC — đọc sâu**

### Vì sao quan trọng?

Đây là nền tảng của toàn bộ câu chuyện YOLO:

- framing detection thành regression,
- single-stage detector,
- một neural network xử lý ảnh và dự đoán trực tiếp bounding box + class probability,
- end-to-end optimization.

### Những phần nên bóc kỹ

- Problem Motivation
- Unified Detection
- Network Architecture
- Training
- Limitations
- Comparison with R-CNN / DPM

### Vai trò trong survey

YOLOv1 trả lời:

> **YOLO bắt đầu từ đâu và tại sao nó khác các detector trước đó?**

---

# 6. Milestone B — YOLOv2 / YOLOv3: Anchor + Multi-scale Era

## 6.1. YOLOv2

### Paper

**YOLO9000: Better, Faster, Stronger**  
Joseph Redmon, Ali Farhadi — CVPR 2017

- Paper: https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html

### Mức ưu tiên

**Đọc chọn lọc**

### Tập trung vào

- Batch Normalization
- Higher-resolution classifier
- Anchor boxes
- Dimension clusters
- Direct location prediction
- Multi-scale training

### Không cần đào quá sâu

Phần joint training để detect hơn 9000 categories không phải trọng tâm nếu Personal Task tập trung vào architecture evolution.

---

## 6.2. YOLOv3

### Paper

**YOLOv3: An Incremental Improvement**  
Joseph Redmon, Ali Farhadi — 2018

- Paper: https://arxiv.org/abs/1804.02767

### Mức ưu tiên

**CORE PAPER**

### Tập trung vào

- Darknet-53
- Residual connections
- Multi-scale prediction
- Objectness / class prediction
- So sánh với YOLOv2

### Architectural milestone

```text
Earlier YOLO
    ↓
Anchor-based prediction
    ↓
YOLOv3
    ↓
Stronger Backbone
+
Multi-scale Detection
```

### Vai trò trong survey

YOLOv3 trả lời:

> **Làm thế nào YOLO cải thiện representation và detection ở nhiều object scales?**

---

# 7. Milestone C — YOLOv4: Backbone–Neck–Head Era

### Paper

**YOLOv4: Optimal Speed and Accuracy of Object Detection**  
Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao — 2020

- Paper: https://arxiv.org/abs/2004.10934

### Mức ưu tiên

**CORE PAPER**

### Tập trung vào

- CSPDarknet53
- SPP
- PAN
- Bag of Freebies
- Bag of Specials
- Mosaic augmentation
- CIoU loss

### Kiến trúc nên thể hiện

```text
Input
  ↓
Backbone
CSPDarknet53
  ↓
Neck
SPP + PAN
  ↓
Detection Head
  ↓
Predictions
```

### Vì sao đây là milestone lớn?

Từ đây, cách nhìn:

```text
Backbone → Neck → Head
```

trở thành framework rất hữu ích để phân tích các YOLO hiện đại.

### Vai trò trong survey

YOLOv4 trả lời:

> **Làm thế nào YOLO tối ưu feature extraction và multi-scale feature fusion trong một detector real-time?**

---

# 8. Milestone D — YOLOv7: Efficient Aggregation & Training Design

### Paper

**YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**  
Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao — CVPR 2023

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html

### Mức ưu tiên

**CORE PAPER**

### Tập trung vào

- E-ELAN
- Efficient Layer Aggregation
- Model scaling
- Planned re-parameterization
- Trainable bag-of-freebies
- Gradient propagation path

### Vai trò trong survey

YOLOv7 trả lời:

> **Sau khi backbone–neck–head đã khá trưởng thành, làm sao tiếp tục tối ưu network aggregation, scaling và training mà không làm inference nặng lên?**

---

# 9. Milestone E — YOLOv10: End-to-End / NMS-Free Direction

### Paper

**YOLOv10: Real-Time End-to-End Object Detection**  
Ao Wang et al. — NeurIPS 2024

- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html

### Mức ưu tiên

**CORE PAPER**

### Problem chính

YOLO truyền thống thường có:

```text
Network
   ↓
Dense Predictions
   ↓
NMS
   ↓
Final Detections
```

YOLOv10 hướng đến:

```text
Network
   ↓
End-to-End Detection
```

### Tập trung vào

- Consistent dual assignments
- One-to-many vs one-to-one assignment
- NMS-free inference
- Holistic efficiency-accuracy driven model design
- Reduction of redundant computation

### Vai trò trong survey

YOLOv10 trả lời:

> **Có thể đưa YOLO từ detector phụ thuộc post-processing sang end-to-end detector hay không?**

---

# 10. Các paper bridge / optional rất đáng đọc

## 10.1. YOLOv6 — Industrial Deployment Perspective

### Paper

**YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications**

- Paper: https://arxiv.org/abs/2209.02976

### Đọc khi

Bạn muốn bổ sung góc nhìn:

- industrial deployment,
- latency,
- quantization,
- hardware-friendly design,
- engineering trade-off.

### Mức ưu tiên

**OPTIONAL**

Không cần nếu scope bài đang quá lớn.

---

## 10.2. YOLOv9 — Information Preservation

### Paper

**YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information**  
Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao — ECCV 2024

- Paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04462.pdf

### Concepts chính

- Information bottleneck
- Programmable Gradient Information (**PGI**)
- Generalized Efficient Layer Aggregation Network (**GELAN**)

### Mức ưu tiên

**OPTIONAL / ADVANCED**

### Đọc khi

Bạn muốn bài survey có chiều sâu hơn về:

- information flow,
- gradient path,
- training behavior,
- architectural efficiency.

### Lưu ý

PGI không nên được trình bày đơn giản như chỉ là "một block mới". Nó liên quan nhiều đến cách đảm bảo gradient hữu ích trong quá trình training.

---

## 10.3. YOLOv12 — Attention-Centric Direction

### Paper

**YOLOv12: Attention-Centric Real-Time Object Detectors**  
Yunjie Tian, Qixiang Ye, David Doermann — NeurIPS 2025

- Paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html

### Mức ưu tiên

**OPTIONAL — FUTURE DIRECTION**

### Vì sao đáng đọc?

YOLO truyền thống chủ yếu phát triển quanh CNN.

YOLOv12 đặt câu hỏi:

> Có thể sử dụng attention như một thành phần trung tâm mà vẫn giữ real-time speed hay không?

### Vai trò trong survey

Rất phù hợp cho section:

> **Emerging Directions / Future of YOLO**

---

# 11. Các version quan trọng nhưng không có formal original paper

## YOLOv5

Ultralytics không phát hành formal research paper chính thức cho YOLOv5.

- Documentation: https://docs.ultralytics.com/models/yolov5

## YOLOv8

Ultralytics không phát hành formal research paper chính thức cho YOLOv8.

- Documentation: https://docs.ultralytics.com/models/yolov8

## YOLO11

Ultralytics không phát hành formal research paper chính thức cho YOLO11.

- Documentation: https://docs.ultralytics.com/models/yolo11

### Cách sử dụng trong bài survey

Có thể đưa vào:

- timeline,
- architecture comparison,
- implementation discussion,
- demo,
- practical ecosystem.

Nhưng không nên gọi một bài báo bên thứ ba là:

> "the original YOLOv8 paper"

nếu Ultralytics không có paper chính thức.

### Nguồn nên dùng

- Official documentation
- Official repository
- Model YAML / source code
- Technical documentation

---

# 12. Bộ paper đọc chính đề xuất

Nếu thời gian có hạn, ưu tiên:

## Tier 1 — Core

1. **YOLOv1** — foundation
2. **YOLOv3** — stronger backbone + multi-scale detection
3. **YOLOv4** — CSP + SPP + PAN / Backbone–Neck–Head
4. **YOLOv7** — aggregation + scaling + training design
5. **YOLOv10** — end-to-end / NMS-free

Đây là bộ:

```text
YOLOv1
  ↓
YOLOv3
  ↓
YOLOv4
  ↓
YOLOv7
  ↓
YOLOv10
```

đủ để kể một câu chuyện architectural evolution khá hoàn chỉnh.

---

## Tier 2 — Bridge Papers

Đọc sau Tier 1:

1. **YOLOv2**
   - giúp hiểu transition sang anchors và multi-scale training.

2. **YOLOv9**
   - giúp hiểu information preservation, GELAN và PGI.

---

## Tier 3 — Optional / Modern Direction

1. **YOLOv6**
   - industrial deployment.

2. **YOLOv12**
   - attention-centric architecture.

3. **YOLOv5 / YOLOv8 / YOLO11**
   - đọc documentation và source thay vì tìm "original paper".

---

# 13. Thứ tự đọc khuyến nghị

```text
Stage 1
YOLOv1
↓
Hiểu bản chất YOLO

Stage 2
YOLOv2 → YOLOv3
↓
Hiểu anchors + multi-scale

Stage 3
YOLOv4
↓
Hiểu Backbone / Neck / Head

Stage 4
YOLOv7
↓
Hiểu aggregation + scaling + training

Stage 5
YOLOv10
↓
Hiểu end-to-end YOLO

Stage 6 — Optional
YOLOv9
↓
Information / Gradient

Stage 7 — Optional
YOLOv12
↓
Attention-centric YOLO
```

---

# 14. Cấu trúc bài survey đề xuất

## Title gợi ý

> **Architectural Evolution of YOLO for Real-Time Object Detection: From Unified Regression to End-to-End and Attention-Centric Detectors**

Có thể rút gọn:

> **A Survey on the Architectural Evolution of YOLO Object Detectors**

---

## 1. Abstract

Ngắn gọn:

- YOLO là gì.
- Vì sao YOLO quan trọng trong real-time object detection.
- Survey tập trung vào architectural evolution.
- Các milestone được khảo sát.
- Các xu hướng chính rút ra.

Không cần viết Abstract đầu tiên. Viết sau cùng.

---

## 2. Introduction

### 2.1. Object Detection

Giới thiệu:

```text
Input Image
    ↓
Object Detector
    ↓
Bounding Boxes + Classes
```

### 2.2. Motivation for Real-Time Detection

Ví dụ:

- autonomous driving,
- robotics,
- surveillance,
- UAV,
- edge computing.

### 2.3. Motivation of YOLO

So sánh tư duy:

```text
Traditional multi-stage pipeline
             vs
YOLO unified single-stage detector
```

### 2.4. Survey Objective

Nêu rõ:

> Survey không cố liệt kê mọi YOLO implementation mà tập trung vào các architectural milestones.

### 2.5. Contributions of This Survey

Ví dụ:

1. Tổ chức YOLO theo architectural milestone thay vì chỉ theo version number.
2. Phân tích sự tiến hóa của Backbone–Neck–Head.
3. So sánh anchor-based, anchor-free và end-to-end paradigms.
4. Tổng hợp xu hướng từ CNN-centric tới attention-centric real-time detector.

---

# 15. Background

Nên giải thích đủ để người nghe hiểu các section sau.

## 15.1. One-stage vs Two-stage detector

```text
Two-stage
Image
 ↓
Region Proposal
 ↓
Classification / Regression

One-stage
Image
 ↓
Direct Detection
```

## 15.2. Bounding Box

Có thể dùng:

$$
B = (x, y, w, h)
$$

## 15.3. IoU

$$
IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
$$

## 15.4. NMS

Giải thích vai trò của Non-Maximum Suppression trước khi đến YOLOv10.

## 15.5. Backbone / Neck / Head

```text
Image
  ↓
Backbone
Feature Extraction
  ↓
Neck
Feature Fusion
  ↓
Head
Prediction
```

Đây là vocabulary trung tâm cho toàn bài.

---

# 16. Main Survey Section — Architectural Evolution

Một cấu trúc hợp lý:

## 4.1. Foundation: YOLOv1

- Motivation
- Unified regression
- Architecture
- Strength
- Limitations

## 4.2. Anchor-Based & Multi-Scale Detection: YOLOv2–v3

- anchor boxes,
- stronger backbone,
- multi-scale prediction.

## 4.3. Feature Aggregation Era: YOLOv4

- CSP,
- SPP,
- PAN,
- backbone–neck–head.

## 4.4. Efficient Network Design: YOLOv7

- E-ELAN,
- scaling,
- re-parameterization.

## 4.5. Information Preservation: YOLOv9

Có thể là subsection nhỏ nếu không muốn deep dive.

## 4.6. End-to-End Detection: YOLOv10

- consistent dual assignment,
- NMS-free inference.

## 4.7. Emerging Attention-Centric YOLO: YOLOv12

Có thể đặt trong Discussion thay vì Main Survey.

---

# 17. Framework để phân tích mọi version

Mỗi model nên được đọc bằng cùng một template:

## A. Problem

> Model trước đang gặp vấn đề gì?

## B. Main Idea

> Paper đưa ra ý tưởng gì để giải quyết?

## C. Backbone

> Feature extraction thay đổi thế nào?

## D. Neck

> Multi-scale feature fusion thay đổi thế nào?

## E. Head

> Bounding box / classification prediction thay đổi thế nào?

## F. Prediction Paradigm

- Anchor-based?
- Anchor-free?
- One-to-many?
- One-to-one?
- End-to-end?

## G. Loss / Label Assignment

> Loss và assignment có innovation quan trọng không?

## H. Training Strategy

> Innovation là architectural hay chỉ tác động trong training?

## I. Inference

- Có NMS không?
- Có post-processing đặc biệt không?

## J. Advantages

> Cải thiện gì?

## K. Limitations

> Trade-off / vấn đề còn lại?

## L. Contribution to YOLO Evolution

> Milestone thật sự của paper này là gì?

---

# 18. Bảng comparison chính nên có trong survey

Tạo một bảng tương tự:

| Model | Backbone | Neck | Head | Anchor | Multi-scale | NMS | Main Architectural Contribution |
|---|---|---|---|---|---|---|---|
| YOLOv1 | CNN | — | Unified prediction | No modern anchors | Limited | Yes/processing | Unified one-stage regression |
| YOLOv3 | Darknet-53 | Feature pyramid-like | Multi-scale YOLO head | Yes | Yes | Yes | Residual backbone + multi-scale detection |
| YOLOv4 | CSPDarknet53 | SPP + PAN | YOLO head | Yes | Yes | Yes | CSP + stronger feature aggregation |
| YOLOv7 | E-ELAN-based | Multi-scale fusion | YOLO head | Depends on configuration | Yes | Yes | Efficient aggregation and scaling |
| YOLOv9 | GELAN | Multi-scale fusion | YOLO head | Implementation-dependent | Yes | Yes | GELAN + PGI |
| YOLOv10 | Efficient YOLO design | Multi-scale | Dual assignment / end-to-end head | Modern design | Yes | No for end-to-end path | NMS-free end-to-end detection |
| YOLOv12 | Attention-centric design | Multi-scale | Modern YOLO head | Modern design | Yes | Architecture-dependent | Attention-centric real-time YOLO |

> **Lưu ý:** trước khi đưa bảng vào bản cuối, xác minh từng cell trực tiếp từ paper / official source. Không suy diễn chỉ từ các blog.

---

# 19. Một bảng khác rất đáng có: Evolution by Problem

| Problem | Architectural Response |
|---|---|
| Multi-stage detection quá chậm | YOLOv1 unified one-stage detection |
| Localization / scale handling còn yếu | YOLOv2/v3 anchors + multi-scale |
| Feature fusion chưa đủ mạnh | YOLOv4 CSP + SPP + PAN |
| Network aggregation / scaling | YOLOv7 E-ELAN + scaling strategy |
| Information loss / gradient quality | YOLOv9 GELAN + PGI |
| Dependency on NMS | YOLOv10 end-to-end NMS-free |
| Attention thường quá chậm cho real-time | YOLOv12 attention-centric design |

Bảng này rất phù hợp cho slide vì tạo được câu chuyện:

> **Problem → Architectural Response → New Limitation → Next Milestone**

---

# 20. Các hình nên chuẩn bị

## Figure 1 — YOLO Evolution Timeline

```text
v1 → v2/v3 → v4 → v7 → v9 → v10 → v12
```

Dùng milestone, không cố đưa mọi version vào.

---

## Figure 2 — YOLOv1 Original Architecture

Lấy / redraw từ original paper với citation.

---

## Figure 3 — Modern YOLO Generic Architecture

```text
Input
 ↓
Backbone
 ↓
Multi-level Features
 ↓
Neck
 ↓
Feature Fusion
 ↓
Detection Head
 ↓
BBox + Class
```

---

## Figure 4 — Anchor-Based vs Anchor-Free

Dùng để giải thích transition của modern YOLO.

---

## Figure 5 — NMS-Based vs End-to-End

```text
Traditional YOLO

Dense Predictions
      ↓
     NMS
      ↓
Detections
```

so với:

```text
End-to-End YOLO

Network
   ↓
Detections
```

---

# 21. Advantages & Limitations — cách trình bày

Không nên chỉ có:

> Advantages: fast  
> Limitations: small objects

Nên chia theo **evolution**.

## Advantages nổi bật của YOLO family

- Real-time inference.
- Single-stage design.
- End-to-end optimization.
- Scalable model sizes.
- Multi-scale object detection.
- Strong deployment ecosystem.
- Phù hợp edge / robotics / UAV.

## Limitations cần phân tích

Tùy generation:

- YOLOv1:
  - localization errors,
  - khó với small / clustered objects.

- Anchor-based YOLO:
  - anchor design / assignment complexity.

- Modern high-performance YOLO:
  - architecture ngày càng phức tạp,
  - khó so sánh FPS công bằng giữa papers,
  - nhiều branch cùng mang tên YOLO nhưng không cùng lineage.

- NMS-based detector:
  - extra post-processing,
  - latency / end-to-end deployment concern.

---

# 22. Typical Applications

Có thể gom theo:

- Autonomous Driving
- Surveillance
- Industrial Inspection
- Robotics
- Medical Imaging
- Agriculture
- UAV / Remote Sensing
- Edge AI

### Liên hệ với UAV

Một sơ đồ đơn giản:

```text
UAV Camera
    ↓
YOLO Detector
    ↓
Object / Crop / Vehicle / Person Detection
    ↓
Higher-level Decision
```

Không cần biến phần applications thành literature survey riêng nếu scope chính là architecture.

---

# 23. Simple Demo / Implementation Example

## Mục tiêu

Demo để chứng minh hiểu pipeline, **không phải benchmark khoa học**.

### Demo đơn giản nhất

```text
Image / Video
    ↓
Pretrained YOLO
    ↓
Inference
    ↓
Bounding Boxes + Labels + Confidence
```

### Gợi ý implementation

Dùng Ultralytics vì setup đơn giản.

Có thể demo với:

- YOLOv8n
- YOLO11n

Ví dụ:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")

results[0].show()
```

### Nên trình bày thêm

- input size,
- detected objects,
- confidence score,
- inference time nếu framework cung cấp.

### Không nên

Dùng một lần chạy trên laptop rồi kết luận:

> YOLO11 nhanh hơn YOLOv8.

Để benchmark cần kiểm soát:

- hardware,
- runtime,
- precision,
- image size,
- batch size,
- warm-up,
- export backend.

---

# 24. Lưu ý quan trọng khi so sánh kết quả giữa papers

## Không so FPS trực tiếp nếu hardware khác nhau

Ví dụ:

```text
Paper A: Tesla V100
Paper B: T4
Paper C: A100
```

thì FPS không thể coi là comparison công bằng.

## Luôn ghi

- Dataset
- Input resolution
- AP metric
- Hardware
- Precision
- Latency / FPS measurement setup

### Metrics

Có thể gặp:

$$
AP_{50}
$$

và:

$$
AP_{50:95}
$$

Hai metric này **không tương đương**.

---

# 25. Reading Note Template

Mỗi paper tạo một note theo template sau:

```markdown
# Paper Name

## 1. Metadata
- Authors:
- Year:
- Venue:
- YOLO version:

## 2. Problem Motivation
-

## 3. Previous Model / Baseline
-

## 4. Main Idea
-

## 5. Architecture
### Backbone
-

### Neck
-

### Head
-

## 6. Prediction Mechanism
- Anchor-based / Anchor-free:
- Multi-scale:
- Label assignment:
- NMS:

## 7. Training
- Loss:
- Augmentation:
- Other strategies:

## 8. Key Innovations
1.
2.
3.

## 9. Advantages
-

## 10. Limitations
-

## 11. Experimental Setup
- Dataset:
- Image size:
- Hardware:
- Metrics:

## 12. What Changed Compared with Previous Milestone?
-

## 13. Why Is This a YOLO Milestone?
-

## 14. Useful Figures / Tables
-

## 15. Questions / Things to Verify
-
```

---

# 26. Paper Reading Strategy

Không đọc paper từ đầu đến cuối với mức độ tập trung như nhau.

## Pass 1 — 10 phút

Đọc:

- Abstract
- Introduction
- Figures
- Conclusion

Trả lời:

> Paper giải quyết vấn đề gì?

---

## Pass 2 — Architecture

Tập trung:

- architecture diagram,
- proposed modules,
- backbone,
- neck,
- head.

Trả lời:

> Structural change nằm ở đâu?

---

## Pass 3 — Training

Đọc:

- loss,
- label assignment,
- augmentation,
- training-only modules.

Trả lời:

> Improvement là architecture hay training trick?

---

## Pass 4 — Experiments

Tập trung vào:

- ablation study,
- comparison,
- latency,
- parameters,
- FLOPs,
- AP.

Trả lời:

> Paper có bằng chứng gì cho từng innovation?

---

# 27. Survey Writing Strategy

## Không viết theo kiểu

```text
YOLOv1...
YOLOv2...
YOLOv3...
YOLOv4...
...
```

nếu mỗi section chỉ là tóm tắt độc lập.

## Nên viết theo causal story

```text
Limitation
    ↓
New architectural idea
    ↓
Improvement
    ↓
Remaining limitation
    ↓
Next milestone
```

Ví dụ:

```text
YOLOv1
Small-object / localization limitations
        ↓
YOLOv2/v3
Anchors + Multi-scale
        ↓
Need stronger feature aggregation
        ↓
YOLOv4
CSP + SPP + PAN
        ↓
Need more efficient aggregation / scaling
        ↓
YOLOv7
...
```

Đây chính là điểm làm bài viết giống **survey** hơn là **summary collection**.

---

# 28. Suggested Final Paper Structure

```text
1. Abstract

2. Introduction
   2.1 Object Detection
   2.2 Real-Time Detection Motivation
   2.3 Origin of YOLO
   2.4 Scope of This Survey
   2.5 Contributions

3. Background
   3.1 One-stage vs Two-stage Detection
   3.2 Bounding Box and IoU
   3.3 NMS
   3.4 Backbone–Neck–Head

4. Architectural Evolution of YOLO
   4.1 YOLOv1 — Unified Detection
   4.2 YOLOv2/v3 — Anchor & Multi-scale Era
   4.3 YOLOv4 — CSP and Feature Aggregation
   4.4 YOLOv7 — Efficient Aggregation & Scaling
   4.5 YOLOv9 — Information Preservation
   4.6 YOLOv10 — End-to-End Detection
   4.7 YOLOv12 — Attention-Centric Direction

5. Cross-Generation Comparison
   5.1 Backbone Evolution
   5.2 Neck Evolution
   5.3 Head Evolution
   5.4 Anchor-Based vs Anchor-Free
   5.5 NMS-Based vs End-to-End
   5.6 Training Strategy Evolution

6. Advantages and Limitations

7. Typical Applications

8. Simple Implementation Demo

9. Discussion
   9.1 Major Trends
   9.2 Fragmentation of the YOLO Family
   9.3 Deployment Considerations
   9.4 Future Directions

10. Conclusion

References
```

---

# 29. Mapping Paper Structure ↔ Course Requirements

| Course Requirement | Survey Section |
|---|---|
| Read the original paper | YOLOv1 deep dive + primary papers for milestones |
| Problem motivation | Introduction + each milestone |
| Main idea of the model | YOLOv1 + milestone analysis |
| Architecture diagram | YOLOv1 + generic modern YOLO + milestone diagrams |
| Key innovations compared to previous models | Main Architectural Evolution section |
| Advantages and limitations | Dedicated comparative section |
| Typical applications | Applications section |
| Simple demo / implementation | Implementation section |

Như vậy bài survey vẫn đáp ứng đầy đủ yêu cầu của môn học mà không bị gò bó vào format "mỗi yêu cầu = một slide".

---

# 30. Presentation Map

Nếu phần thuyết trình khoảng 10–15 phút:

## Slide 1 — Title

**Architectural Evolution of YOLO**

## Slide 2 — Object Detection & Motivation

## Slide 3 — YOLOv1 Main Idea

## Slide 4 — YOLOv1 Architecture

## Slide 5 — Why YOLO Had to Evolve

## Slide 6 — YOLO Evolution Timeline

## Slide 7 — Backbone / Neck / Head

## Slide 8 — Major Architectural Milestones

Có thể gom:

```text
v1 → v3 → v4 → v7 → v10
```

## Slide 9 — Anchor-Based → Anchor-Free → End-to-End

## Slide 10 — Advantages & Limitations

## Slide 11 — Applications

## Slide 12 — Demo + Conclusion

---

# 31. Những thứ không cần làm

Để tránh scope explosion:

### Không cần

- nghiên cứu sâu mọi version YOLO;
- benchmark tất cả model;
- train YOLO from scratch;
- tự implement detector;
- đọc tất cả derivation toán;
- liệt kê mọi augmentation trick;
- so sánh FPS giữa papers mà không kiểm tra hardware;
- đưa mọi YOLO variant do cộng đồng đặt tên vào timeline.

### Ưu tiên

- hiểu kiến trúc,
- hiểu motivation của thay đổi,
- hiểu trade-off,
- nối được các milestone thành một câu chuyện.

---

# 32. Checklist làm bài

## Phase 1 — Foundation

- [ ] Đọc YOLOv1
- [ ] Viết Problem Motivation
- [ ] Hiểu architecture
- [ ] Hiểu loss ở mức đủ dùng
- [ ] Ghi limitations

## Phase 2 — Classical Evolution

- [ ] Đọc chọn lọc YOLOv2
- [ ] Đọc YOLOv3
- [ ] Xác định anchor / multi-scale milestone
- [ ] Đọc YOLOv4
- [ ] Hiểu Backbone–Neck–Head

## Phase 3 — Efficient Modern YOLO

- [ ] Đọc YOLOv7
- [ ] Hiểu E-ELAN
- [ ] Hiểu model scaling / re-parameterization

## Phase 4 — Modern Paradigm

- [ ] Đọc YOLOv10
- [ ] Hiểu NMS
- [ ] Hiểu one-to-many / one-to-one concept
- [ ] Hiểu end-to-end detection

## Phase 5 — Optional Depth

- [ ] YOLOv9 — GELAN
- [ ] YOLOv9 — PGI
- [ ] YOLOv12 — Attention-centric
- [ ] YOLOv6 — Industrial deployment

## Phase 6 — Survey Synthesis

- [ ] Timeline
- [ ] Architecture comparison table
- [ ] Problem → Innovation table
- [ ] Backbone evolution
- [ ] Neck evolution
- [ ] Head evolution
- [ ] Anchor / Anchor-free / End-to-End comparison

## Phase 7 — Course Requirements

- [ ] Problem motivation
- [ ] Main idea
- [ ] Architecture diagram
- [ ] Innovation comparison
- [ ] Advantages / limitations
- [ ] Applications
- [ ] Demo

---

# 33. Working Thesis của bài survey

Một thesis có thể dùng để định hướng toàn bài:

> **The evolution of YOLO is not merely a sequence of numbered releases, but a progression of architectural responses to recurring problems in real-time object detection: feature representation, multi-scale fusion, prediction design, information preservation, computational efficiency, and end-to-end inference.**

Ý tưởng tương ứng bằng tiếng Việt:

> **Sự phát triển của YOLO không đơn thuần là chuỗi version tăng dần, mà là quá trình các kiến trúc mới liên tục giải quyết những hạn chế của thế hệ trước về feature representation, multi-scale fusion, prediction mechanism, information flow, efficiency và end-to-end inference.**

Đây có thể coi là **xương sống lập luận** của toàn bài.

---

# 34. Short Map — nếu cần nhìn nhanh

```text
PERSONAL TASK
│
├── Requirement from course
│   ├── Original paper
│   ├── Motivation
│   ├── Main idea
│   ├── Architecture
│   ├── Innovations
│   ├── Pros / Cons
│   ├── Applications
│   └── Demo
│
├── Survey Strategy
│   └── Architectural Milestones
│
├── CORE
│   ├── YOLOv1
│   ├── YOLOv3
│   ├── YOLOv4
│   ├── YOLOv7
│   └── YOLOv10
│
├── BRIDGE
│   ├── YOLOv2
│   └── YOLOv9
│
├── OPTIONAL
│   ├── YOLOv6
│   └── YOLOv12
│
├── IMPLEMENTATION / ECOSYSTEM
│   ├── YOLOv5
│   ├── YOLOv8
│   └── YOLO11
│
└── Final Synthesis
    ├── Backbone evolution
    ├── Neck evolution
    ├── Head evolution
    ├── Anchor → Anchor-free
    ├── NMS → End-to-End
    └── CNN → Attention-centric
```

---

# 35. Primary References

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.  
   **You Only Look Once: Unified, Real-Time Object Detection.** CVPR 2016.  
   https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html

2. Redmon, J., & Farhadi, A.  
   **YOLO9000: Better, Faster, Stronger.** CVPR 2017.  
   https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html

3. Redmon, J., & Farhadi, A.  
   **YOLOv3: An Incremental Improvement.** 2018.  
   https://arxiv.org/abs/1804.02767

4. Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M.  
   **YOLOv4: Optimal Speed and Accuracy of Object Detection.** 2020.  
   https://arxiv.org/abs/2004.10934

5. Li, C. et al.  
   **YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications.** 2022.  
   https://arxiv.org/abs/2209.02976

6. Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M.  
   **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors.** CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html

7. Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M.  
   **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.** ECCV 2024.  
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04462.pdf

8. Wang, A. et al.  
   **YOLOv10: Real-Time End-to-End Object Detection.** NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract-Conference.html

9. Tian, Y., Ye, Q., & Doermann, D.  
   **YOLOv12: Attention-Centric Real-Time Object Detectors.** NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html

---

# 36. Official Non-Paper References

### Ultralytics YOLOv5

https://docs.ultralytics.com/models/yolov5

### Ultralytics YOLOv8

https://docs.ultralytics.com/models/yolov8

### Ultralytics YOLO11

https://docs.ultralytics.com/models/yolo11

Các nguồn này dùng để nghiên cứu implementation / architecture / usage của các version không có formal original research paper do Ultralytics phát hành.

---

## Final Direction

Nếu phải ghi nhớ duy nhất một roadmap:

```text
Understand deeply
YOLOv1
   ↓
Track architectural evolution
YOLOv3
   ↓
YOLOv4
   ↓
YOLOv7
   ↓
YOLOv10
   ↓
Optional depth
YOLOv9 / YOLOv12
   ↓
Use YOLOv5 / v8 / YOLO11
for ecosystem & implementation context
```

**Không cần biết tất cả các YOLO.**

Mục tiêu của Personal Task là có thể giải thích rõ:

> **YOLO đã thay đổi kiến trúc như thế nào, vì sao phải thay đổi, và mỗi milestone giải quyết vấn đề gì của thế hệ trước.**
