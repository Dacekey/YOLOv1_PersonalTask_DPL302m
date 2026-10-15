# YOLOv3 — Roadmap, Key Characteristics, Improvements, Advantages, and Limitations

> **Scope:** Tài liệu này tập trung vào YOLOv3 và đặt nó trong roadmap phát triển **YOLOv1 → YOLOv2/YOLO9000 → YOLOv3**.  
> Nội dung YOLOv1 và YOLOv3 được tổng hợp từ các paper gốc đã nghiên cứu. Phần YOLOv2 được dùng như **cầu nối kiến trúc** để giải thích những thay đổi mà YOLOv3 kế thừa và phát triển tiếp.
>
> **Mục đích sử dụng:** Có thể dùng làm nguồn nội dung để chuyển sang LaTeX report sau này.

---

# 1. Vị trí của YOLOv3 trong roadmap phát triển YOLO

YOLOv1 đặt nền móng cho triết lý:

$$
\boxed{
\text{Full image}
\rightarrow
\text{single neural network}
\rightarrow
\text{bounding boxes + class predictions}
}
$$

Thay vì dùng nhiều stage như R-CNN, YOLOv1 biến object detection thành một bài toán dự đoán thống nhất (*unified detection*) và xử lý toàn ảnh trong một forward pass.

Tuy nhiên, YOLOv1 vẫn còn nhiều hạn chế đáng kể:

- chỉ dự đoán trên **một spatial scale tương đối coarse**;
- mỗi grid cell chỉ có số lượng bounding-box predictors hạn chế;
- mỗi cell chỉ có **một bộ class probabilities dùng chung**;
- localization là nguồn lỗi chính;
- khả năng phát hiện small objects còn yếu;
- architecture còn phụ thuộc vào fully connected layers ở cuối;
- bounding-box regression chưa có priors/anchors để làm điểm tham chiếu.

Roadmap chính có thể nhìn như sau:

```text
YOLOv1
│
│  Unified one-stage detection
│  Direct bbox regression
│  Single coarse prediction scale
│  24 Conv + 2 FC
│
▼
YOLOv2 / YOLO9000
│
│  Batch Normalization
│  Darknet-19
│  Remove FC detection layers
│  Anchor boxes
│  Dimension clusters using k-means
│  Direct location prediction
│  Multi-scale training
│
▼
YOLOv3
   Darknet-53
   Residual connections
   Anchor-based prediction retained
   Objectness per box
   Independent logistic class prediction
   3-scale detection
   Feature fusion using upsampling + concatenation
```

Điểm quan trọng:

> YOLOv3 không phải là một kiến trúc hoàn toàn tách biệt với YOLOv2. Nó **kế thừa nhiều thay đổi quan trọng từ YOLOv2**, sau đó thay backbone, thay class prediction, và đặc biệt bổ sung **multi-scale detection**.

---

# 2. Những limitation trước YOLOv3 dẫn tới motivation phát triển

## 2.1 Localization vẫn là vấn đề lớn từ YOLOv1

YOLOv1 có tốc độ rất cao nhưng paper chỉ ra rằng nguồn lỗi lớn nhất của nó là:

> **incorrect localization**

YOLOv1 dùng một grid $7\times7$ và dự đoán bounding boxes từ representation tương đối coarse.

Điều này tạo ra hai vấn đề:

1. vị trí và biên object có thể không được xác định chính xác;
2. small objects đặc biệt nhạy với localization error.

Một sai lệch nhỏ trên large object có thể không ảnh hưởng nhiều tới IoU, nhưng cùng sai lệch đó trên small object có thể làm IoU giảm mạnh.

## 2.2 Strong spatial constraint của YOLOv1

YOLOv1 áp dụng quy tắc:

> cell chứa tâm object chịu trách nhiệm detect object đó.

Mỗi cell chỉ dự đoán một số lượng box cố định, và class probability lại dùng chung ở mức cell.

Điều này gây khó khăn khi:

- nhiều objects nằm gần nhau;
- nhiều object centers rơi vào cùng một cell;
- small objects xuất hiện thành nhóm.

Ví dụ:

```text
One grid cell

+----------------------+
| bird   bird   bird   |
|   bird      bird     |
+----------------------+
```

Nếu nhiều tâm object cùng rơi vào một cell, prediction capacity của YOLOv1 không đủ để biểu diễn tất cả chúng.

## 2.3 Coarse feature maps làm small-object detection yếu

YOLOv1 downsample ảnh qua nhiều layers và cuối cùng sử dụng feature representation rất coarse.

Vấn đề là:

```text
Large object
-------------------------
chiếm nhiều pixels
→ vẫn còn feature rõ sau downsampling

Small object
----
rất ít pixels
→ thông tin dễ bị mất sau downsampling
```

Do đó một motivation quan trọng của các version tiếp theo là:

$$
\boxed{
\text{giữ semantic information}
+
\text{khôi phục finer spatial information}
}
$$

---

# 3. YOLOv2 — cầu nối từ YOLOv1 đến YOLOv3

YOLOv2/YOLO9000 là bước chuyển rất quan trọng vì nhiều thứ thường được nhìn thấy ở YOLOv3 thực tế đã bắt đầu từ YOLOv2.

## 3.1 Từ direct bbox regression sang anchor-based prediction

YOLOv1 dự đoán trực tiếp:

$$
(x,y,w,h)
$$

YOLOv2 thay đổi bằng cách sử dụng **anchor boxes**.

Anchor box là một bounding-box prior/template có shape đại diện.

Thay vì yêu cầu model tự tạo width và height hoàn toàn từ đầu:

```text
Feature
  ↓
Predict bbox size from scratch
```

model được cho một prior:

```text
Anchor
  ↓
Predict adjustment
  ↓
Final bbox
```

Ý tưởng tổng quát:

$$
\boxed{
\text{bbox prediction}
=
\text{anchor prior}
+
\text{learned transformation}
}
$$

Điều này làm bài toán localization dễ học hơn, đặc biệt khi anchor đã có shape gần với ground-truth object.

## 3.2 Dimension clusters

Một vấn đề của anchor boxes là:

> chọn anchor shape nào?

Thay vì chọn thủ công, YOLOv2 sử dụng **k-means clustering trên ground-truth bounding-box dimensions** để tìm các shape đại diện.

Ví dụ dataset chứa nhiều bbox:

```text
20 x 30
18 x 28
22 x 35
...
100 x 60
110 x 70
...
300 x 250
...
```

K-means sẽ gom chúng thành những cluster có shape gần nhau.

Cluster centroids được dùng làm anchor priors.

Do đó:

$$
\boxed{
\text{Ground-truth boxes}
\rightarrow
\text{k-means}
\rightarrow
\text{representative anchor shapes}
}
$$

Anchor không phải ground truth.

- **Ground truth:** đáp án của một object cụ thể.
- **Anchor:** shape/template dùng làm điểm tham chiếu.

## 3.3 Loại bỏ fully connected detection layers

YOLOv1 kết thúc bằng:

```text
Conv feature
   ↓
FC 4096
   ↓
FC output
```

YOLOv2 chuyển sang prediction dựa trên convolutional feature maps.

Điều này tạo nền tảng tốt hơn cho:

- prediction ở nhiều spatial locations;
- anchor-based detection;
- nhiều input resolutions;
- multi-scale design về sau.

## 3.4 Multi-scale training

YOLOv2 đưa ra cơ chế thay đổi input resolution trong quá trình training.

Ví dụ:

```text
320 x 320
352 x 352
...
416 x 416
...
608 x 608
```

Mục đích:

> cùng một network có thể chạy ở nhiều input sizes, tạo trade-off giữa speed và accuracy.

Cần phân biệt:

### Multi-scale training

$$
\boxed{
\text{thay đổi input resolution khi training}
}
$$

### Multi-scale detection của YOLOv3

$$
\boxed{
\text{predict ở nhiều feature-map resolutions trong cùng forward pass}
}
$$

Hai khái niệm này không giống nhau.

---

# 4. YOLOv3 — main characteristics

YOLOv3 giữ triết lý one-stage của YOLO nhưng thay đổi mạnh architecture và prediction strategy.

Các characteristic nổi bật nhất:

1. **Darknet-53 backbone**
2. **Residual connections**
3. **Anchor-based bounding-box prediction**
4. **Objectness prediction per box**
5. **Independent logistic classifiers**
6. **Binary cross-entropy for class prediction**
7. **Three-scale prediction**
8. **Upsampling + feature concatenation**
9. **Improved small-object detection**
10. **Maintains strong speed–accuracy trade-off**

---

# 5. Darknet-53 Backbone

YOLOv3 dùng **Darknet-53** làm feature extractor.

Darknet-53 có:

$$
53\text{ convolutional layers}
$$

và sử dụng chủ yếu:

$$
1\times1
$$

và:

$$
3\times3
$$

convolution.

Architecture còn sử dụng **shortcut/residual connections**.

Một residual block có thể hình dung:

```text
Input
  │
  ├───────────────────────┐
  ↓                       │
1 x 1 Conv                │
  ↓                       │
3 x 3 Conv                │
  ↓                       │
  + <─────────────────────┘
```

Residual mapping:

$$
H(x)=F(x)+x
$$

giúp gradient truyền qua deep network dễ hơn và cho phép backbone sâu hơn.

## 5.1 Vì sao Darknet-53 được chọn?

YOLO cần hai mục tiêu đồng thời:

$$
\text{strong feature extraction}
$$

và:

$$
\text{real-time efficiency}
$$

Darknet-53 được thiết kế như một điểm cân bằng giữa hai mục tiêu.

Paper YOLOv3 cho thấy Darknet-53 có classification accuracy tương đương những ResNet sâu nhưng vẫn nhanh hơn đáng kể trong setup benchmark của họ.

Do đó Darknet-53 phù hợp với triết lý:

$$
\boxed{
\text{accuracy + speed}
}
$$

của YOLO.

---

# 6. Anchor-based Bounding-box Prediction trong YOLOv3

YOLOv3 tiếp tục dùng anchor mechanism được phát triển từ YOLOv2.

Network không dự đoán bbox cuối trực tiếp.

Nó predict:

$$
t_x,t_y,t_w,t_h
$$

và decode thành:

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y
$$

$$
b_w=p_we^{t_w}
$$

$$
b_h=p_he^{t_h}
$$

Trong đó:

- $(c_x,c_y)$: location của grid cell;
- $(p_w,p_h)$: anchor width/height;
- $(t_x,t_y,t_w,t_h)$: values model học;
- $(b_x,b_y,b_w,b_h)$: predicted bbox.

## 6.1 Anchor không khóa cứng bbox

Anchor chỉ là **reference/template**.

Ví dụ:

$$
p_w=100,\qquad p_h=50.
$$

Nếu:

$$
t_w=\ln2
$$

thì:

$$
b_w=100e^{\ln2}=200.
$$

Do đó model có thể biến đổi anchor rất đáng kể.

Mental model:

```text
Anchor prior
    ↓
learn offsets / scaling
    ↓
Predicted bbox
    ↓
approach ground truth
```

Anchor vẫn nằm trong prediction equation nhưng không phải hard constraint về final shape.

---

# 7. Objectness Prediction

YOLOv3 mỗi bounding-box prediction có một **objectness score**.

Có thể hiểu:

$$
\boxed{
\text{Objectness}
=
\text{box này có object hay không?}
}
$$

Cấu trúc prediction của một box:

```text
One box
|
+-- x, y, w, h
|
+-- objectness
|
+-- class predictions
```

Objectness khác với class probability.

Ví dụ:

- objectness trả lời: **“Có object không?”**
- class prediction trả lời: **“Nếu có thì thuộc class nào?”**

---

# 8. Independent Logistic Classifiers

Đây là một thay đổi quan trọng so với cách biểu diễn class của YOLOv1.

YOLOv3 không dùng softmax cho class predictions.

Mỗi class có một independent logistic classifier:

```text
person  -> sigmoid
dog     -> sigmoid
car     -> sigmoid
woman   -> sigmoid
...
```

Cho class $i$:

$$
p_i=\sigma(z_i).
$$

Mỗi output class được tính độc lập.

Do đó có thể có:

$$
P(Person)=0.95
$$

và:

$$
P(Woman)=0.90
$$

cùng lúc.

Điều này phù hợp với multilabel situations, nơi một object có thể hợp lý thuộc nhiều semantic labels.

## 8.1 Vì sao không softmax?

Softmax ép:

$$
\sum_i p_i=1.
$$

Điều này tạo competition giữa classes và ngầm giả định rằng một prediction chỉ thuộc đúng một mutually-exclusive class.

Independent sigmoid không có constraint này.

## 8.2 Binary Cross-Entropy

YOLOv3 sử dụng Binary Cross-Entropy cho class predictions.

Cho một class:

$$
L_{BCE}
=
-[y\log p+(1-y)\log(1-p)].
$$

Ví dụ:

```text
Target:
dog = 1

Prediction:
dog = 0.8
```

BCE đo sai số giữa prediction và binary target.

Do mỗi class được xử lý độc lập:

```text
Is dog?
yes / no

Is person?
yes / no

Is car?
yes / no
```

thay vì buộc phải chọn đúng một class duy nhất.

---

# 9. Three-scale Detection

Đây là một trong những characteristic quan trọng nhất của YOLOv3.

YOLOv3 predict bounding boxes tại:

$$
\boxed{3\text{ different scales}}
$$

Với input $416\times416$, cách triển khai tiêu chuẩn thường được hình dung bằng:

$$
13\times13
$$

$$
26\times26
$$

$$
52\times52.
$$

Có thể gọi trực giác:

```text
13 x 13 -> coarse
26 x 26 -> medium
52 x 52 -> fine
```

Tên coarse/medium/fine chỉ là cách diễn giải để dễ hiểu; paper gọi chung là predictions across three scales.

## 9.1 Mỗi scale đều thực hiện detection

Đúng:

```text
Scale 1
  ↓
Detection

Scale 2
  ↓
Detection

Scale 3
  ↓
Detection
```

Mỗi scale predict:

- bbox offsets;
- objectness;
- class predictions.

Trên COCO, mỗi location dùng 3 anchors.

Prediction tensor cho một scale có dạng:

$$
N\times N\times
[3(4+1+80)]
$$

với:

- 4 bbox coordinates;
- 1 objectness;
- 80 class predictions.

---

# 10. 9 Anchor Clusters và 3 Prediction Scales

YOLOv3 dùng 9 representative anchor shapes được tạo từ k-means clustering.

Trong paper, các anchors trên COCO là:

$$
(10,13),(16,30),(33,23),
$$

$$
(30,61),(62,45),(59,119),
$$

$$
(116,90),(156,198),(373,326).
$$

Sau đó:

$$
9\text{ anchors}
\div
3\text{ scales}
=
3\text{ anchors per scale}.
$$

Mental model:

```text
9 anchor shapes
|
+-- 3 anchors -> scale 1
|
+-- 3 anchors -> scale 2
|
+-- 3 anchors -> scale 3
```

Trực giác:

```text
small anchors  -> finer prediction scale

medium anchors -> medium prediction scale

large anchors  -> coarser prediction scale
```

Điều này cho phép detector có nhiều prediction templates phù hợp với object sizes khác nhau.

---

# 11. Feature Fusion: Upsampling + Concatenation

YOLOv3 không chỉ lấy deep feature map cuối cùng.

Nó dùng:

1. feature map sâu có strong semantic information;
2. upsample feature đó;
3. concatenate với earlier feature map có spatial detail cao hơn.

Conceptually:

```text
Deep feature
     |
     +-------> Detection
     |
     v
 Upsample x2
     |
     + Earlier feature
     |
 Concatenate
     |
     v
 Process
     |
     +-------> Detection
     |
     v
 Upsample x2
     |
     + Earlier finer feature
     |
 Concatenate
     |
     v
 Detection
```

Điều này tạo sự kết hợp:

$$
\boxed{
\text{deep semantic information}
+
\text{fine spatial information}
}
$$

---

# 12. Multi-scale Detector là gì?

Một **multi-scale detector** trong ngữ cảnh YOLOv3 là detector thực hiện prediction trên nhiều feature-map spatial resolutions.

YOLOv1:

```text
Image
 ↓
deep feature
 ↓
ONE prediction scale
```

YOLOv3:

```text
Image
 ↓
Darknet-53
 ↓
coarse feature ---------> detect
 ↓
upsample + fusion
 ↓
medium feature ---------> detect
 ↓
upsample + fusion
 ↓
fine feature -----------> detect
```

Các scale không phải ba model riêng biệt.

Tất cả nằm trong:

$$
\boxed{
\text{one neural network forward pass}
}
$$

---

# 13. Tại sao 3 scales giúp small-object detection?

Một small object chứa rất ít pixels.

Sau nhiều lần downsampling:

```text
small object
   ↓
downsampling
   ↓
downsampling
   ↓
information becomes weak
```

Finer-resolution feature maps giữ được nhiều location/detail information hơn.

Do đó YOLOv3 có thể dùng:

```text
high-resolution feature
+
semantic information from deep feature
```

để detect small objects tốt hơn.

Paper ghi nhận rằng YOLOv3 cải thiện đáng kể $AP_S$ so với các version YOLO trước.

---

# 14. YOLOv1 → YOLOv2 → YOLOv3: Architectural Evolution

| Characteristic | YOLOv1 | YOLOv2 | YOLOv3 |
|---|---|---|---|
| Detection paradigm | One-stage | One-stage | One-stage |
| BBox prediction | Direct regression | Anchor-based | Anchor-based |
| Anchor generation | None | K-means dimension clusters | K-means dimension clusters |
| Detection FC layers | Yes | Removed | No FC detection head |
| Backbone | Custom 24-conv CNN | Darknet-19 | Darknet-53 |
| Residual connections | No | No | Yes |
| Prediction scales | Single | Mainly single detection feature scale | 3 |
| Feature fusion across scales | No | Limited | Upsample + concatenate |
| Class prediction | Shared per cell in v1 formulation | Per anchor evolution begins | Per box |
| Class activation | Conditional class probabilities | Earlier YOLO formulation | Independent logistic |
| Small-object capability | Weak | Improved | Stronger due to multi-scale prediction |
| Multi-scale training | No | Yes | Yes |
| Objectness | Confidence formulation | Anchor objectness | Objectness per box |

---

# 15. Characteristic nổi bật nhất của YOLOv3

Nếu cần chọn những điểm quan trọng nhất để đưa vào report, có thể tập trung vào năm ý:

## 15.1 Darknet-53

Backbone sâu hơn, có residual connections, mạnh hơn Darknet-19 nhưng vẫn giữ efficiency phù hợp với real-time detection.

## 15.2 Anchor-based Prediction

Kế thừa anchor mechanism và dimension clustering từ YOLOv2.

## 15.3 Three-scale Detection

Prediction ở ba feature resolutions là thay đổi rất quan trọng để xử lý object sizes khác nhau.

## 15.4 Feature Fusion

Upsampling deep features và concatenate với earlier features giúp kết hợp semantic information và spatial details.

## 15.5 Independent Logistic Class Prediction

Mỗi box có class predictions riêng và các class được xử lý bằng independent logistic classifiers với BCE, phù hợp với multilabel settings hơn softmax.

---

# 16. Advantages của YOLOv3

## 16.1 Cải thiện small-object detection

Đây là một improvement rõ ràng so với YOLOv1.

Multi-scale predictions giúp model sử dụng finer-resolution features nên small objects không còn phụ thuộc duy nhất vào coarse deep feature map.

## 16.2 Stronger feature extractor

Darknet-53:

- deeper;
- residual;
- hierarchical;
- strong classification accuracy;
- vẫn có high computational efficiency.

Điều này cho detection head feature representations tốt hơn.

## 16.3 Better handling of different object sizes

Ba prediction scales và nhiều anchor shapes làm detector thích nghi tốt hơn với:

- small objects;
- medium objects;
- large objects.

## 16.4 Class prediction linh hoạt hơn

Independent logistic classifiers:

- không ép các class mutually exclusive;
- hỗ trợ multilabel semantics;
- mỗi bbox có class prediction riêng.

Điều này khắc phục một phần constraint class representation của YOLOv1.

## 16.5 Vẫn giữ tốc độ cao

YOLOv3 lớn hơn các predecessor nhưng vẫn duy trì philosophy:

$$
\boxed{
\text{real-time / high-speed object detection}
}
$$

Paper báo cáo:

- YOLOv3-320: khoảng 22 ms;
- YOLOv3-416: khoảng 29 ms;
- YOLOv3-608: khoảng 51 ms

trên hardware benchmark của paper.

Vì vậy người dùng có thể chọn trade-off:

$$
\text{smaller input}
\rightarrow
\text{faster}
$$

hoặc:

$$
\text{larger input}
\rightarrow
\text{higher accuracy}
$$

## 16.6 Strong AP50 performance

YOLOv3 có performance rất tốt khi evaluation threshold là:

$$
IoU=0.5.
$$

Paper cho thấy YOLOv3 cạnh tranh tốt với các detector mạnh khác theo $AP_{50}$ trong khi inference nhanh hơn đáng kể.

---

# 17. Limitations của YOLOv3

YOLOv3 cải thiện rất nhiều so với v1 nhưng không giải quyết hoàn toàn localization.

## 17.1 Performance giảm khi IoU threshold cao

Paper nhận xét YOLOv3 mạnh ở:

$$
AP_{50}
$$

nhưng performance giảm đáng kể khi evaluation yêu cầu bounding boxes chính xác hơn.

Điều này cho thấy:

> YOLOv3 thường tìm được object và tạo bbox tương đối đúng, nhưng vẫn khó đạt **high-precision box alignment**.

Tức localization problem của YOLOv1 đã được cải thiện nhưng chưa biến mất.

## 17.2 COCO AP@[0.5:0.95] không mạnh bằng một số detector khác

COCO AP trung bình qua nhiều IoU thresholds:

$$
0.50,0.55,\ldots,0.95
$$

đặt yêu cầu cao hơn vào bbox localization.

YOLOv3 không mạnh bằng RetinaNet và một số two-stage detectors trên metric này.

Điều này phản ánh cùng vấn đề:

$$
\boxed{
\text{decent boxes}
\neq
\text{perfectly aligned boxes}
}
$$

## 17.3 Medium/Large-object AP chưa vượt trội

Một kết quả thú vị trong paper là:

- small-object performance được cải thiện rõ;
- nhưng medium và large object performance tương đối kém hơn một số competitors.

Tác giả cũng nói cần nghiên cứu thêm để hiểu nguyên nhân.

Do đó không nên diễn giải:

> multi-scale prediction làm mọi object size đều tốt hơn tuyệt đối.

Điểm được paper nhấn mạnh rõ nhất là **small-object improvement**.

## 17.4 Anchor-based design tạo thêm dependency vào priors

Anchor mechanism giúp localization dễ học hơn, nhưng nó cũng đưa vào một thiết kế thủ công/hyperparameter mới:

- số anchors;
- anchor shapes;
- anchor assignment;
- scale assignment.

YOLOv3 dùng 9 cluster anchors và chia cho 3 scales.

Do đó performance có thể phụ thuộc một phần vào mức độ anchor priors phù hợp với dataset.

Đây là limitation theo góc nhìn thiết kế, dù paper YOLOv3 không trình bày nó như limitation chính.

## 17.5 Vẫn cần NMS

YOLOv3 vẫn tạo nhiều candidate detections.

Do đó duplicate predictions vẫn cần được loại bằng Non-Maximum Suppression.

Pipeline vẫn có:

```text
Network prediction
   ↓
score filtering
   ↓
NMS
   ↓
final detections
```

Vì vậy YOLOv3 chưa phải end-to-end NMS-free detector theo nghĩa của một số YOLO generation về sau.

## 17.6 Architecture phức tạp và lớn hơn predecessor

Darknet-53 sâu hơn Darknet-19 và YOLOv1 backbone.

Three-scale prediction cùng feature fusion cũng làm detection architecture phức tạp hơn.

Trade-off:

$$
\boxed{
\text{better representation and accuracy}
\leftrightarrow
\text{higher computation/model complexity}
}
$$

Tuy vậy YOLOv3 vẫn được thiết kế để giữ high inference speed.

---

# 18. Những thứ YOLOv3 thử nhưng không hiệu quả

Paper có một section khá đặc biệt liệt kê các idea tác giả thử nhưng không cải thiện model.

## 18.1 Linear $x,y$ prediction

Tác giả thử predict $x,y$ offset bằng linear activation thay vì logistic formulation.

Kết quả:

- model stability giảm;
- mAP giảm.

Do đó YOLOv3 giữ:

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y.
$$

## 18.2 Focal Loss

Focal Loss được thử nghiệm nhưng làm mAP giảm khoảng 2 points trong setup của tác giả.

Do đó YOLOv3 không dùng Focal Loss.

Paper chỉ đưa ra hypothesis rằng objectness/class formulation của YOLOv3 có thể đã xử lý phần nào vấn đề mà Focal Loss hướng tới; tác giả cũng thừa nhận họ không hoàn toàn chắc chắn về lý do.

## 18.3 Dual IoU Threshold Assignment

Tác giả cũng thử assignment strategy tương tự Faster R-CNN với multiple IoU thresholds nhưng không đạt kết quả tốt.

Điều này cho thấy YOLOv3 không phải tập hợp tất cả kỹ thuật nổi tiếng lúc đó; paper giữ lại những design choices thực nghiệm phù hợp với framework YOLO.

---

# 19. YOLOv3 cải thiện limitation nào của YOLOv1?

| YOLOv1 Limitation | YOLOv3 Response |
|---|---|
| Coarse single-scale feature | 3-scale prediction |
| Weak small-object detection | Fine-scale detection + feature fusion |
| Simple/shallower feature extractor | Darknet-53 |
| Deep-network training khó hơn nếu chỉ tăng layers | Residual connections |
| Direct bbox regression | Anchor-based parameterization inherited from v2 |
| Shared class probability at cell level | Class predictions per box |
| Mutually constrained class formulation | Independent logistic classifiers |
| Limited box templates | Multiple anchors per spatial location |
| Limited adaptation to object sizes | 9 anchors across 3 scales |

Điều cần nhấn mạnh:

> Không phải tất cả các solution này đều được YOLOv3 phát minh.  
> **Anchor boxes, dimension clustering và một số prediction improvements đã xuất hiện từ YOLOv2.**  
> YOLOv3 kế thừa chúng và bổ sung backbone + multi-scale feature/prediction strategy mạnh hơn.

---

# 20. YOLOv3 vẫn giữ lại điều gì từ YOLOv1?

Dù kiến trúc thay đổi nhiều, philosophy cốt lõi vẫn giữ nguyên.

## 20.1 One-stage philosophy

Không quay lại pipeline:

```text
region proposal
→ separate classifier
→ separate box regressor
```

YOLOv3 vẫn thực hiện dense detection trực tiếp từ full-image neural-network features.

## 20.2 Full-image processing

Network vẫn xử lý toàn ảnh và prediction trong một unified detector.

## 20.3 Speed as a first-class objective

YOLO không chỉ tối ưu accuracy.

Design luôn cân bằng:

$$
\boxed{
\text{accuracy}
\leftrightarrow
\text{inference speed}
}
$$

Đây là characteristic xuyên suốt từ YOLOv1 tới YOLOv3.

---

# 21. Mental Model hoàn chỉnh của YOLOv3

```text
Input image
    |
    v
Darknet-53 backbone
    |
    +-- residual blocks
    |
    +-- hierarchical feature maps
    |
    v
Deep/coarse feature
    |
    +---------------------> Detection scale 1
    |                       3 anchors/location
    |
    v
Upsample
    +
Earlier feature
    |
Concatenate
    |
    v
Medium feature
    |
    +---------------------> Detection scale 2
    |                       3 anchors/location
    |
    v
Upsample
    +
Earlier finer feature
    |
Concatenate
    |
    v
Fine feature
    |
    +---------------------> Detection scale 3
                            3 anchors/location
```

Mỗi anchor prediction:

```text
Anchor prediction
|
+-- tx
+-- ty
+-- tw
+-- th
|
+-- objectness
|
+-- independent class probabilities
```

Final bbox:

$$
b_x=\sigma(t_x)+c_x
$$

$$
b_y=\sigma(t_y)+c_y
$$

$$
b_w=p_we^{t_w}
$$

$$
b_h=p_he^{t_h}.
$$

---

# 22. YOLOv3 in One Sentence

> **YOLOv3 evolves the original YOLO one-stage detector into a deeper anchor-based multi-scale detection system, using Darknet-53 with residual connections and feature fusion to improve feature representation and small-object detection while retaining the high-speed design philosophy of YOLO.**

---

# 23. Report-ready Summary

YOLOv3 represents an important architectural maturation of the YOLO family rather than a completely new detection paradigm. YOLOv1 introduced unified one-stage object detection, while YOLOv2 progressively improved localization and recall through anchor boxes, dimension clustering, convolutional prediction, and multi-scale training. YOLOv3 builds on this foundation by introducing the deeper Darknet-53 backbone with residual connections and by performing detection at three different feature-map scales.

The three-scale prediction design is particularly important because it addresses a major weakness of earlier YOLO versions: poor detection of small objects caused by relying heavily on coarse downsampled features. YOLOv3 combines deep semantic features with higher-resolution earlier features through upsampling and concatenation, allowing prediction heads to operate on coarse, medium, and fine spatial representations.

YOLOv3 also modifies the class prediction mechanism. Each bounding box predicts its own objectness and class outputs, while independent logistic classifiers and binary cross-entropy are used instead of enforcing a softmax-based mutually exclusive class assumption. Together with anchor-based localization and multi-scale feature fusion, these changes make YOLOv3 substantially more flexible than YOLOv1.

Despite these improvements, YOLOv3 still exhibits localization limitations. It performs strongly at $AP_{50}$, indicating that it usually detects objects with reasonably aligned boxes, but its performance decreases as stricter IoU thresholds are used. The model also remains anchor-dependent, requires NMS, and is computationally more complex than earlier versions. Therefore, YOLOv3 can be viewed as a major step toward the architectural structure of modern YOLO detectors, while still retaining several limitations that later generations continue to address.

---

# 24. Suggested LaTeX Section Structure

```text
3. YOLOv3: Multi-Scale Anchor-Based Detection

3.1 Evolution from YOLOv1 and YOLOv2
    3.1.1 Limitations of YOLOv1
    3.1.2 YOLOv2 as the architectural bridge

3.2 Main Characteristics of YOLOv3
    3.2.1 Darknet-53 Backbone
    3.2.2 Residual Connections
    3.2.3 Anchor-Based Bounding Box Prediction
    3.2.4 Objectness Prediction
    3.2.5 Independent Logistic Classifiers

3.3 Multi-Scale Detection
    3.3.1 Three Prediction Scales
    3.3.2 Anchor Distribution Across Scales
    3.3.3 Upsampling and Feature Concatenation
    3.3.4 Small-Object Detection

3.4 Training and Prediction Characteristics
    3.4.1 Bounding Box Parameterization
    3.4.2 Binary Cross-Entropy Classification
    3.4.3 Multi-Scale Training

3.5 Advantages

3.6 Limitations

3.7 Comparison with YOLOv1 and YOLOv2

3.8 Summary
```

---

# References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, **“You Only Look Once: Unified, Real-Time Object Detection,”** CVPR, 2016.

[2] J. Redmon and A. Farhadi, **“YOLO9000: Better, Faster, Stronger,”** CVPR, 2017.

[3] J. Redmon and A. Farhadi, **“YOLOv3: An Incremental Improvement,”** arXiv:1804.02767, 2018.

---

# Key Takeaway for the YOLO Roadmap

```text
YOLOv1:
"What if object detection is one unified neural-network problem?"

        ↓

YOLOv2:
"How can that unified detector localize more reliably
and become easier to train?"

        ↓

YOLOv3:
"How can we make the representation much stronger
and detect objects effectively across different spatial scales?"
```

The most important conceptual evolution is therefore:

$$
\boxed{
\text{Unified Detection}
\rightarrow
\text{Anchor-Based Detection}
\rightarrow
\text{Multi-Scale Anchor-Based Detection}
}
$$
