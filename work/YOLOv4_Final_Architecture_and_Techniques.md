# YOLOv4 — Final Architecture, Final Techniques, Motivation, Advantages, and Limitations

> **Scope:** Tài liệu này tập trung vào **final YOLOv4** — kiến trúc cuối cùng và các kỹ thuật thực sự được sử dụng trong cấu hình YOLOv4 mà paper chốt lại.  
> Phần comparison/ablation chỉ được nhắc ngắn để hiểu **YOLOv4 thực sự làm gì**, không đi sâu vào toàn bộ các phương án mà tác giả từng thử.
>
> **Primary source:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao, *YOLOv4: Optimal Speed and Accuracy of Object Detection*, 2020.

---

# 1. YOLOv4 trong roadmap phát triển YOLO

```text
YOLOv1
│
│  Unified one-stage object detection
│
▼
YOLOv2
│
│  Anchor-based prediction
│  Dimension clustering
│  Multi-scale training
│
▼
YOLOv3
│
│  Darknet-53
│  Residual connections
│  3-scale detection
│  Multi-scale feature fusion
│
▼
YOLOv4
   CSPDarknet53
   + SPP
   + PAN
   + YOLOv3 detection head
   + stronger training recipe
   + improved bbox optimization
   + improved post-processing
```

YOLOv4 **không thay đổi hoàn toàn detection paradigm**.

Nó vẫn giữ:

- one-stage detection;
- anchor-based prediction;
- multi-scale detection;
- YOLOv3-style detection head.

Điểm khác biệt lớn là YOLOv4 tập trung tối ưu **toàn bộ detector system**:

$$
\boxed{
\text{Backbone}
+
\text{Neck}
+
\text{Training recipe}
+
\text{BBox optimization}
+
\text{Post-processing}
}
$$

thay vì phát minh một prediction paradigm hoàn toàn mới.

---

# 2. Motivation của YOLOv4

Mục tiêu cốt lõi của YOLOv4 là cải thiện:

$$
\boxed{\text{Speed–Accuracy Trade-off}}
$$

Tác giả muốn xây một detector:

- có accuracy cao;
- vẫn chạy real-time;
- tối ưu tốt trên GPU thực tế;
- có thể train bằng một GPU phổ thông;
- không chỉ chạy tốt về mặt theoretical FLOPs/BFLOPs.

Điểm quan trọng là paper không xem **low theoretical computation** là mục tiêu duy nhất.

Một model có ít FLOPs hơn chưa chắc chạy nhanh hơn nếu:

- GPU utilization kém;
- memory access không hiệu quả;
- các operation khó parallelize.

Vì vậy YOLOv4 thiên về:

$$
\boxed{
\text{practical GPU efficiency}
}
$$

hơn là chỉ:

$$
\boxed{
\text{minimum theoretical FLOPs}
}
$$

---

# 3. YOLOv4 thực sự làm gì?

YOLOv4 có thể được xem là một nghiên cứu theo hướng:

$$
\boxed{
\text{Selection}
+
\text{Modification}
+
\text{Integration}
}
$$

Tác giả khảo sát nhiều technique đã tồn tại, sau đó:

1. chọn backbone phù hợp cho object detection;
2. chọn feature-aggregation modules;
3. chọn các training techniques có lợi;
4. sửa đổi một số technique để phù hợp hơn với detector;
5. kết hợp chúng thành một detector có speed–accuracy trade-off tốt.

Do đó novelty của YOLOv4 chủ yếu nằm ở **system-level design và empirical integration**, chứ không phải ở một single detection mechanism hoàn toàn mới.

Paper cũng có một số modification/new training components riêng, nhưng không nên nói rằng mọi thành phần trong YOLOv4 đều do YOLOv4 phát minh.

---

# 4. Final YOLOv4 Architecture

Final high-level pipeline:

$$
\boxed{
\text{Input}
\rightarrow
\text{CSPDarknet53}
\rightarrow
\text{SPP}
\rightarrow
\text{PAN}
\rightarrow
\text{YOLOv3 Head}
\rightarrow
\text{DIoU-NMS}
}
$$

Có thể phân thành:

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
  YOLOv3 Head
  ↓
Post-processing
  DIoU-NMS
  ↓
Final detections
```

Paper chốt kiến trúc chính:

- **Backbone:** CSPDarknet53
- **Neck:** SPP + PAN
- **Head:** YOLOv3

Đây là phần quan trọng nhất cần nhớ khi đưa YOLOv4 vào roadmap.

---

# 5. Backbone — CSPDarknet53

YOLOv3 dùng:

$$
\text{Darknet-53}
$$

YOLOv4 chuyển sang:

$$
\boxed{\text{CSPDarknet53}}
$$

Trong quá trình chọn backbone, tác giả nhấn mạnh:

> một backbone tốt cho image classification chưa chắc là backbone tốt nhất cho object detection.

Object detector cần:

- input resolution lớn;
- receptive field đủ lớn;
- model capacity đủ cao;
- khả năng giữ và truyền feature tốt ở nhiều scale.

CSPDarknet53 được chọn làm backbone cuối cùng vì phù hợp hơn với object detection trong thiết kế của YOLOv4.

---

## 5.1 CSP — Cross Stage Partial connections

CSP chia feature flow thành nhiều phần thay vì đưa toàn bộ feature qua cùng một transformation path.

Mental model đơn giản:

```text
Input feature
     |
     +---------------+
     |               |
     v               |
 Deep processing     |
     |               |
     +------ Merge --+
             |
             v
           Output
```

Mục đích:

- cải thiện learning capability;
- cải thiện gradient/information flow;
- hạn chế redundant computation.

CSP là một trong những technique quan trọng nhất làm CSPDarknet53 khác Darknet-53 của YOLOv3.

---

## 5.2 Mish Activation

YOLOv4 sử dụng **Mish** như một activation function trong final design.

Mish có dạng:

$$
f(x)=x\tanh(\ln(1+e^x))
$$

Nó là một smooth, continuously differentiable activation.

Mục tiêu chung:

- hỗ trợ gradient propagation;
- tăng khả năng representation của backbone.

Mish là một **architectural/training technique**, không phải một high-level stage riêng trong pipeline.

---

## 5.3 MiWRC

Paper liệt kê **Multi-input Weighted Residual Connections (MiWRC)** trong nhóm Bag of Specials được dùng cho backbone.

Vai trò khái quát:

- hỗ trợ information flow;
- hỗ trợ residual learning;
- tăng hiệu quả kết hợp feature paths.

Không cần xem MiWRC như một pipeline stage riêng; nó là một technique nằm bên trong backbone design.

---

# 6. Neck — SPP + PAN

YOLOv4 formalize rõ cấu trúc:

$$
\boxed{
\text{Backbone}
\rightarrow
\text{Neck}
\rightarrow
\text{Head}
}
$$

Neck nằm giữa backbone và head.

Mục tiêu:

$$
\boxed{
\text{collect + fuse + aggregate features from multiple levels}
}
$$

---

# 7. SPP — Spatial Pyramid Pooling

YOLOv4 đặt SPP phía sau CSPDarknet53.

SPP dùng nhiều max-pooling kernel sizes:

$$
1\times1,\quad5\times5,\quad9\times9,\quad13\times13
$$

Conceptually:

```text
                 ┌─ MaxPool 1×1 ──┐
                 ├─ MaxPool 5×5 ──┤
Feature map ────►├─ MaxPool 9×9 ──┼─► Concatenate
                 └─ MaxPool13×13 ─┘
```

Mục đích chính:

$$
\boxed{\text{increase receptive field}}
$$

và lấy context ở nhiều spatial ranges.

---

# 8. PAN — Path Aggregation Network

YOLOv3 đã có multi-scale feature fusion kiểu FPN-like.

YOLOv4 chọn **PAN** để aggregate feature tốt hơn giữa các backbone levels và detector levels.

Mental model:

```text
High-level semantic features
             ↓
          top-down
             ↓

Low-level spatial features
             ↑
          bottom-up
```

Mục tiêu là kết hợp tốt hơn:

$$
\boxed{
\text{semantic information}
+
\text{spatial/localization information}
}
$$

YOLOv4 còn dùng một **modified PAN**, trong đó tác giả sửa cách connection để phù hợp với thiết kế của mình.

---

# 9. Modified SAM

YOLOv4 sử dụng một **modified Spatial Attention Module (SAM)**.

Attention giúp network tăng trọng số cho feature hữu ích và giảm ảnh hưởng của feature ít quan trọng.

Mental model:

```text
Feature map
    ↓
Attention weighting
    ↓
Important regions/features receive more emphasis
    ↓
Better feature representation
```

YOLOv4 sửa SAM từ spatial-wise attention sang point-wise attention trong thiết kế của mình.

SAM là một plugin/feature-enhancement technique, không cần xem như một stage ngang hàng với Backbone–Neck–Head.

---

# 10. Detection Head — YOLOv3 Head

Điểm rất quan trọng:

$$
\boxed{
\text{YOLOv4 không thay thế YOLOv3 detection head}
}
$$

Final YOLOv4 tiếp tục sử dụng **YOLOv3 anchor-based head**.

Do đó vẫn giữ các characteristic lớn của YOLOv3:

- anchor-based bounding-box prediction;
- objectness;
- class prediction;
- multi-scale detection.

Evolution từ v3 → v4 vì vậy gần hơn với:

```text
YOLOv3 head
     +
stronger backbone
     +
better neck
     +
better training
     +
better bbox optimization
```

---

# 11. Final Techniques used by YOLOv4

Dưới đây là những technique paper liệt kê là **YOLOv4 uses**, không phải toàn bộ những candidate methods đã từng được thử.

| Khu vực | Technique dùng trong final YOLOv4 | Vai trò |
|---|---|---|
| Backbone | CSPDarknet53 | Feature extraction |
| Backbone | CSP connections | Improve feature/gradient flow |
| Backbone | Mish | Activation |
| Backbone | MiWRC | Improve residual/information propagation |
| Neck | SPP | Increase receptive field |
| Neck | Modified PAN | Multi-level feature aggregation |
| Feature enhancement | Modified SAM | Attention |
| Head | YOLOv3 Head | Anchor-based multi-scale detection |
| BBox training | CIoU Loss | Better bounding-box regression |
| Prediction formulation | Eliminate Grid Sensitivity | Improve center prediction near grid boundaries |
| Assignment | Multiple Anchors for one Ground Truth | Allow more suitable anchors to supervise one GT |
| Post-processing | DIoU-NMS | NMS using center-distance information |
| Augmentation | Mosaic | Mix four training images |
| Augmentation | CutMix | Mix regions from training images |
| Augmentation | SAT | Self-Adversarial Training |
| Regularization | DropBlock | Reduce overfitting |
| Regularization | Class Label Smoothing | Reduce overconfidence |
| Normalization | CmBN | Cross mini-Batch Normalization |
| LR schedule | Cosine Annealing | Learning-rate scheduling |
| Scale training | Random Training Shapes | Train across multiple input resolutions |
| Hyperparameters | Genetic Algorithm / Optimal Hyperparameters | Search suitable training settings |

---

# 12. Bag of Freebies in Final YOLOv4

**Bag of Freebies (BoF)**:

$$
\boxed{
\text{improve training/accuracy without materially increasing inference cost}
}
$$

Final YOLOv4 uses BoF such as:

- Mosaic;
- CutMix;
- DropBlock;
- Class Label Smoothing;
- CIoU Loss;
- CmBN;
- SAT;
- Grid Sensitivity Elimination;
- Multiple Anchors for one GT;
- Cosine Annealing;
- Optimal Hyperparameters;
- Random Training Shapes.

Mental model:

```text
Better training
     ↓
Better learned weights
     ↓
Same / nearly same inference architecture
```

---

# 13. Bag of Specials in Final YOLOv4

**Bag of Specials (BoS)**:

$$
\boxed{
\text{slightly increase inference/model complexity}
\rightarrow
\text{obtain useful accuracy gain}
}
$$

Final YOLOv4 uses important BoS including:

- Mish;
- CSP;
- MiWRC;
- SPP;
- SAM;
- PAN;
- DIoU-NMS.

---

# 14. Mosaic

Mosaic ghép 4 training images thành một sample:

```text
+---------+---------+
| image 1 | image 2 |
+---------+---------+
| image 3 | image 4 |
+---------+---------+
```

Lợi ích:

- tăng diversity của context;
- object xuất hiện ở nhiều relative scales;
- tăng số object/context trong một training sample;
- hỗ trợ training với mini-batch nhỏ hơn.

## Origin note

Paper mô tả Mosaic như một new augmentation method trong YOLOv4, nhưng phần Acknowledgements **credit ý tưởng Mosaic cho Glenn Jocher**.

Vì vậy không nên viết:

> “YOLOv4 authors invented Mosaic.”

Cách an toàn hơn:

> **Mosaic was incorporated and formalized in the YOLOv4 training recipe, with the idea credited by the paper to Glenn Jocher.**

---

# 15. Self-Adversarial Training — SAT

SAT được paper trình bày như một new training/data-augmentation technique.

Nó gồm hai stages:

```text
Stage 1
Network modifies the image
to make detection harder

        ↓

Stage 2
Network is trained normally
on the modified image
```

Có thể hiểu:

$$
\boxed{
\text{model generates a harder sample for itself}
}
$$

rồi học detect trên sample đó.

---

# 16. CIoU Loss

YOLOv4 sử dụng **CIoU** cho bounding-box regression.

CIoU xét đồng thời:

1. bounding-box overlap;
2. center distance;
3. aspect-ratio consistency.

Mental model:

$$
\boxed{
CIoU
=
\text{overlap}
+
\text{center alignment}
+
\text{shape consistency}
}
$$

Điều này giúp bbox optimization gắn với geometry của object tốt hơn việc chỉ tối ưu trực tiếp từng coordinate độc lập.

---

# 17. Eliminate Grid Sensitivity

YOLOv3 sử dụng:

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y
$$

Do sigmoid bị giới hạn trong $(0,1)$, prediction muốn tiến rất sát cell boundary sẽ cần $|t_x|$ hoặc $|t_y|$ rất lớn.

YOLOv4 sửa formulation để giảm **grid sensitivity**, làm center prediction gần boundary dễ hơn.

Đây không phải một pipeline stage độc lập.

Nó là một **bbox prediction formulation improvement**.

---

# 18. Multiple Anchors for One Ground Truth

Trong YOLOv4 training, một ground-truth box có thể supervise nhiều anchors nếu IoU giữa anchor và GT vượt threshold thích hợp.

```text
Ground Truth
     |
     +--> Anchor A  ✓
     +--> Anchor B  ✓
     +--> Anchor C  ✗
```

Mục tiêu:

- tăng positive training signals;
- giúp detector học object assignment linh hoạt hơn.

---

# 19. CmBN — Cross mini-Batch Normalization

YOLOv4 sử dụng **CmBN**, được sửa từ Cross-Iteration Batch Normalization.

CmBN thu thập statistics giữa các mini-batches nằm trong cùng một batch.

Mục tiêu của thiết kế là phù hợp hơn với:

$$
\boxed{\text{single-GPU training}}
$$

và tình huống mini-batch size bị giới hạn bởi GPU memory.

---

# 20. DIoU-NMS

Traditional NMS chủ yếu dùng:

$$
IoU
$$

để xác định các candidate boxes overlap quá nhiều.

DIoU-NMS thêm thông tin:

$$
\boxed{\text{distance between bounding-box centers}}
$$

vào quyết định suppression.

Pipeline cuối:

```text
YOLOv3 Head
    ↓
Candidate boxes
    ↓
DIoU-NMS
    ↓
Final detections
```

---

# 21. Training Pipeline vs Inference Pipeline

## Training

```text
Training image
     ↓
Mosaic / CutMix / SAT / Random Shapes
     ↓
CSPDarknet53
     ↓
SPP
     ↓
PAN + SAM
     ↓
YOLOv3 Head
     ↓
CIoU + other training objectives
     ↓
Backward / optimization
```

Song song với training:

```text
CmBN
DropBlock
Label Smoothing
Cosine Annealing
Optimal Hyperparameters
Multiple-anchor assignment
Grid-sensitivity elimination
```

## Inference

```text
Input image
     ↓
CSPDarknet53
     ↓
SPP
     ↓
PAN
     ↓
YOLOv3 Head
     ↓
DIoU-NMS
     ↓
Final detections
```

Không nên nhầm các training techniques như Mosaic, SAT, CIoU hoặc label smoothing là những sequential blocks trong inference graph.

---

# 22. YOLOv3 → YOLOv4: Core Evolution

| Characteristic | YOLOv3 | YOLOv4 |
|---|---|---|
| Detector type | One-stage | One-stage |
| Prediction | Anchor-based | Anchor-based |
| Detection scales | Multi-scale | Multi-scale |
| Backbone | Darknet-53 | CSPDarknet53 |
| Feature aggregation | FPN-like | SPP + PAN |
| Attention | Không phải characteristic chính | Modified SAM |
| Head | YOLOv3 Head | YOLOv3 Head |
| BBox optimization | Earlier YOLO formulation | CIoU |
| Training recipe | Relatively simpler | Strong BoF/BoS recipe |
| Data augmentation | Conventional | Mosaic, CutMix, SAT, etc. |
| Post-processing | NMS | DIoU-NMS |
| Main design goal | Strong multi-scale detector | Better practical speed–accuracy trade-off |

---

# 23. Comparison/Ablation — chỉ cần hiểu ở mức nào?

Paper thử nhiều candidates như:

- different backbones;
- multiple activation functions;
- multiple augmentation methods;
- different losses;
- different feature-fusion modules;
- attention modules;
- post-processing methods.

Mục đích của phần comparison không phải tạo ra nhiều versions cần học riêng.

Nó chủ yếu giúp tác giả quyết định:

$$
\boxed{
\text{combination nào được giữ lại trong final YOLOv4}
}
$$

Với mục tiêu survey/roadmap, không cần ghi nhớ toàn bộ các phương án bị loại.

Điều cần nhớ là final recipe:

$$
\boxed{
\text{CSPDarknet53}
+
\text{SPP}
+
\text{PAN}
+
\text{YOLOv3 Head}
+
\text{BoF/BoS final recipe}
}
$$

---

# 24. Advantages của YOLOv4

## 24.1 Better speed–accuracy trade-off

Đây là mục tiêu trung tâm.

YOLOv4 cải thiện accuracy đáng kể trong khi vẫn duy trì real-time operation trên GPU.

Paper báo cáo YOLOv4-608 đạt:

$$
43.5\%~AP
$$

$$
65.7\%~AP_{50}
$$

và khoảng:

$$
62\text{–}65~FPS
$$

trên Tesla V100 tùy phần benchmark/reporting của paper.

## 24.2 Stronger feature extraction

CSPDarknet53 mạnh hơn Darknet-53 theo thiết kế detector của YOLOv4.

## 24.3 Better multi-scale feature aggregation

SPP + PAN giúp kết hợp:

$$
\text{context}
+
\text{semantic information}
+
\text{spatial information}
$$

tốt hơn.

## 24.4 Better bbox optimization

CIoU trực tiếp tối ưu geometry của bbox tốt hơn các coordinate-only formulations đơn giản.

## 24.5 Strong training recipe

Mosaic, SAT, DropBlock, Label Smoothing, CmBN và các training strategies giúp model học robust hơn mà không làm inference cost tăng tương ứng.

## 24.6 Practical single-GPU orientation

YOLOv4 được thiết kế để có thể train bằng GPU phổ thông như GTX 1080 Ti / RTX 2080 Ti thay vì yêu cầu large multi-GPU setup.

---

# 25. Limitations của YOLOv4

> **Lưu ý:** Paper YOLOv4 không có một section “Limitations” riêng. Các điểm dưới đây chủ yếu là **architectural implications** từ final design, không phải toàn bộ đều là limitations mà chính tác giả phát biểu trực tiếp.

## 25.1 Final system khá phức tạp

YOLOv4 tích hợp nhiều components:

- CSP;
- SPP;
- PAN;
- SAM;
- Mish;
- multiple BoF;
- CIoU;
- DIoU-NMS.

Điều này làm training recipe và implementation phức tạp hơn YOLOv3.

## 25.2 Vẫn anchor-based

YOLOv4 vẫn kế thừa YOLOv3 anchor-based head.

Do đó vẫn cần:

- anchor configuration;
- anchor assignment;
- IoU thresholds;
- matching strategy.

## 25.3 Vẫn cần NMS

YOLOv4 vẫn cần DIoU-NMS để loại duplicate detections.

Do đó nó chưa phải một end-to-end NMS-free detector.

## 25.4 Accuracy phụ thuộc mạnh vào training recipe

Một phần performance của YOLOv4 đến từ sự kết hợp nhiều:

- augmentations;
- regularization;
- normalization;
- assignment;
- scheduler;
- hyperparameters.

Điều này làm reproduction/tuning trên dataset mới có thể phức tạp hơn một architecture đơn giản.

## 25.5 Hardware dependence

YOLOv4 chủ động ưu tiên real GPU speed và parallel computation hơn low BFLOPs.

Điều này tốt cho GPU deployment nhưng không đồng nghĩa architecture sẽ tối ưu như nhau trên mọi hardware platform.

---

# 26. Main Characteristics của YOLOv4

Nếu cần rút xuống các characteristic quan trọng nhất:

1. **CSPDarknet53 backbone**
2. **SPP for receptive-field expansion**
3. **PAN for multi-scale feature aggregation**
4. **YOLOv3 anchor-based detection head**
5. **Strong Bag-of-Freebies training recipe**
6. **Bag-of-Specials architectural enhancements**
7. **CIoU bounding-box regression**
8. **DIoU-NMS post-processing**
9. **Mosaic + SAT augmentation/training**
10. **Practical single-GPU, real-time speed–accuracy optimization**

---

# 27. YOLOv4 in One Sentence

> **YOLOv4 keeps the anchor-based multi-scale detection foundation of YOLOv3, but substantially strengthens the complete detector through CSPDarknet53, SPP, PAN, attention and an extensive BoF/BoS training recipe, with the primary goal of achieving a better practical speed–accuracy trade-off on conventional GPUs.**

---

# 28. Key Takeaway for the YOLO Roadmap

```text
YOLOv1:
"Can object detection be unified into one network?"

        ↓

YOLOv2:
"Can anchor-based prediction make localization and training stronger?"

        ↓

YOLOv3:
"Can detection operate effectively across multiple feature scales?"

        ↓

YOLOv4:
"Can we optimize the entire detector system
to get better accuracy while staying real-time
on practical GPU hardware?"
```

Conceptual evolution:

$$
\boxed{
\text{Unified Detection}
\rightarrow
\text{Anchor-Based Detection}
\rightarrow
\text{Multi-Scale Detection}
\rightarrow
\text{System-Level Speed–Accuracy Optimization}
}
$$

---

# 29. Suggested LaTeX Section Structure

```text
4. YOLOv4: System-Level Optimization for Speed and Accuracy

4.1 Motivation
    4.1.1 Speed–Accuracy Trade-off
    4.1.2 Practical GPU-Oriented Design

4.2 Final Architecture
    4.2.1 CSPDarknet53 Backbone
    4.2.2 SPP
    4.2.3 PAN
    4.2.4 Modified SAM
    4.2.5 YOLOv3 Detection Head
    4.2.6 DIoU-NMS

4.3 Final Training Techniques
    4.3.1 Bag of Freebies
    4.3.2 Mosaic and CutMix
    4.3.3 Self-Adversarial Training
    4.3.4 DropBlock and Label Smoothing
    4.3.5 CmBN
    4.3.6 CIoU Loss
    4.3.7 Training Scheduling and Assignment

4.4 Evolution from YOLOv3

4.5 Advantages

4.6 Limitations

4.7 Summary
```

---

# References

[1] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, **“YOLOv4: Optimal Speed and Accuracy of Object Detection,”** arXiv:2004.10934, 2020.

[2] J. Redmon and A. Farhadi, **“YOLOv3: An Incremental Improvement,”** arXiv:1804.02767, 2018.

[3] C.-Y. Wang et al., **“CSPNet: A New Backbone that can Enhance Learning Capability of CNN,”** CVPR Workshops, 2020.

---

# Final Summary

YOLOv4 is best understood not as a completely new object-detection paradigm, but as a **carefully engineered final detector configuration** built on top of YOLOv3's one-stage, anchor-based, multi-scale foundation.

Its final architecture is:

$$
\boxed{
\text{CSPDarknet53}
\rightarrow
\text{SPP}
\rightarrow
\text{PAN}
\rightarrow
\text{YOLOv3 Head}
\rightarrow
\text{DIoU-NMS}
}
$$

and its performance comes not only from this architecture, but also from a carefully selected training recipe involving techniques such as Mosaic, SAT, CIoU, CmBN, DropBlock, label smoothing, cosine annealing, grid-sensitivity elimination, and improved anchor assignment.

For roadmap study, the most important message is:

$$
\boxed{
\text{YOLOv4}
=
\text{stronger architecture}
+
\text{stronger training recipe}
+
\text{practical GPU optimization}
}
$$
