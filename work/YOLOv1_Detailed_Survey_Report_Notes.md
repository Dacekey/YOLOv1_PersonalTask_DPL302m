# YOLOv1 — Detailed Study Notes for Survey Report

> **Scope of this document:** This document currently covers **YOLOv1 only**, based primarily on the original paper *You Only Look Once: Unified, Real-Time Object Detection* by Redmon et al. Later YOLO versions can be added as separate sections when they are studied.
>
> **Course-oriented structure:** The content is organized around the required elements: original paper, problem motivation, main idea, architecture, key innovations, advantages and limitations, typical applications, and a simple demo / implementation plan.

---

# 1. Original Paper

**Paper:** Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.

YOLOv1 introduces a unified approach to object detection in which a **single convolutional neural network** receives the full image and directly predicts bounding boxes and class probabilities in one network evaluation.

The central idea can be summarized as:

$$
\text{Image pixels}
\rightarrow
\text{Single neural network}
\rightarrow
\text{Bounding boxes + confidence + class probabilities}
$$

Instead of treating object detection as a pipeline of separate region-search, feature-extraction, classification, and bounding-box-refinement stages, YOLO reformulates the task as a **single regression problem** from the input image to spatially separated bounding boxes and associated class probabilities.

The original paper evaluates YOLO primarily on the PASCAL VOC object-detection benchmark. For PASCAL VOC, the model uses:

$$
S=7,\qquad B=2,\qquad C=20
$$

where:

- $S \times S$ is the prediction grid;
- $B$ is the number of bounding-box predictors per grid cell;
- $C$ is the number of object classes.

The final output tensor therefore has shape:

$$
S\times S\times(B\cdot5+C)
=
7\times7\times30.
$$

---

# 2. Problem Motivation

## 2.1 Object detection before YOLO

Before YOLO, many object-detection systems reused image classifiers as part of a larger detection pipeline.

### Sliding-window approaches

A classifier can be evaluated repeatedly at many image locations and scales:

$$
\text{Image}
\rightarrow
\text{many windows}
\rightarrow
\text{classifier for every window}.
$$

A major weakness is the large amount of repeated computation. Different window sizes or image pyramids are also required because objects may appear at different scales.

### R-CNN-style approaches

R-CNN reduces the need to scan every possible window by first generating candidate regions, for example with Selective Search. A simplified R-CNN pipeline is:

$$
\text{Image}
\rightarrow
\text{Region proposals}
\rightarrow
\text{CNN feature extraction}
\rightarrow
\text{SVM classification}
\rightarrow
\text{Bounding-box regression}
\rightarrow
\text{NMS}.
$$

Selective Search proposes regions that *may* contain objects but does not know their semantic classes. Each region must then be processed by later components.

The original YOLO paper emphasizes two major problems with such multi-stage systems:

1. they are computationally expensive and therefore difficult to run in real time;
2. the individual components are often trained separately, making the whole detection system harder to optimize jointly.

## 2.2 Motivation of YOLO

YOLO asks whether the complete detection problem can instead be learned directly:

$$
\boxed{
\text{full image}
\rightarrow
\text{object locations + object classes}
}
$$

The main motivation is therefore not to invent a new convolution operation, but to **reformulate the complete object-detection problem as one unified neural-network prediction problem**.

Regression

This design aims to obtain:

- a simpler detection pipeline;
- real-time inference;
- end-to-end optimization;
- global reasoning over the entire image instead of isolated local regions.

---

# 3. Main Idea: Unified Detection

## 3.1 Full-image reasoning

YOLO processes the **entire image in one network evaluation**.

This is fundamentally different from interpreting the $7\times7$ grid as 49 image crops. The grid is a **prediction structure**, not a mechanism that independently feeds 49 patches through the CNN.

Conceptually:

```text
Full image
    |
    v
Convolutional neural network
    |
    v
7 x 7 prediction grid
    |
    v
Bounding boxes + confidence + classes
```

Because the entire image is visible to the network, YOLO can use global context when making predictions.

## 3.2 Grid responsibility

The image is conceptually divided into an $S\times S$ grid.

For YOLOv1 on PASCAL VOC:

$$
S=7.
$$

If the **center of an object** falls inside grid cell $i$, that cell is responsible for detecting the object.

Important distinction:

> A bounding box is **not restricted to the physical boundary of its grid cell**. The cell only determines responsibility according to the location of the object's center.

Therefore, a predicted box may extend across several neighboring cells.

---

# 4. Output Representation

## 4.1 Predictions from one grid cell

Each grid cell predicts:

- $B=2$ bounding boxes;
- one set of $C=20$ conditional class probabilities.

Each bounding box contains five outputs:

$$
(x,y,w,h,\text{confidence}).
$$

Therefore, for one grid cell:

$$
2\times5+20=30
$$

values are predicted.

Across all $7\times7=49$ cells:

$$
7\times7\times30
$$

is the final output tensor.

## 4.2 Meaning of $x$ and $y$

The variables $x$ and $y$ represent the bounding-box center relative to the bounds of the responsible grid cell.

They are **parameterized as offsets inside the cell**.

For example, if:

$$
x=0.7,\qquad y=0.4,
$$

then the predicted center is approximately 70% across the cell horizontally and 40% vertically.

The motivation is to let the model learn local offsets instead of directly regressing large absolute pixel coordinates.

The training targets are encoded according to this rule. The network does not independently discover the parameterization; the data-processing and loss pipeline defines the representation that the model must predict.

Conceptually:

```text
Ground-truth bbox in pixel coordinates
        |
        v
Encode using YOLO representation
        |
        +-- x,y: offsets inside responsible cell
        +-- w,h: relative to full image
        |
        v
Training targets
```

## 4.3 Meaning of $w$ and $h$

Bounding-box width and height are normalized by the full image dimensions:

$$
w=\frac{W_{\text{box}}}{W_{\text{image}}},
\qquad
h=\frac{H_{\text{box}}}{H_{\text{image}}}.
$$

Thus the ground-truth values are represented relative to the image size.

During inference, these values can be decoded back into image coordinates.

## 4.4 Bounding-box confidence

YOLOv1 defines bounding-box confidence as:

$$
C = P(\text{Object})\times IoU^{truth}_{pred}.
$$

This value does **not** represent the probability of a specific class such as `dog` or `cat`.

Instead, it combines:

1. whether the model believes an object is associated with the prediction;
2. how well the predicted bounding box matches the ground-truth bounding box.

During training:

- if no object is associated with the prediction, the desired confidence is driven toward $0$;
- for the responsible predictor of an object, the desired confidence reflects the IoU between the predicted box and the ground-truth box.

The ground-truth box is part of the supervised training annotation. The training data therefore contains both **class labels** and **bounding-box annotations**.

## 4.5 Conditional class probabilities

Each grid cell predicts:

$$
P(Class_i\mid Object)
$$

for each class $i$.

This means:

> assuming that the cell contains an object, what is the probability that the object belongs to class $i$?

A crucial YOLOv1 design detail is that the cell predicts **one class-probability vector regardless of the number of bounding-box predictors**.

Therefore:

```text
One grid cell
|
+-- Bounding-box predictor 1
|   +-- x1, y1, w1, h1, confidence1
|
+-- Bounding-box predictor 2
|   +-- x2, y2, w2, h2, confidence2
|
+-- One shared 20-class probability vector
```

The two bounding-box predictors are **not two different classes** and are **not two separate algorithms**. They are two prediction slots produced from shared network features.

## 4.6 Class-specific confidence score

At inference time, the conditional class probability is multiplied by the individual bounding-box confidence:

$$
P(Class_i\mid Object)
\times
P(Object)
\times
IoU^{truth}_{pred}.
$$

Using probability factorization:

$$
P(Class_i\mid Object)P(Object)
=
P(Class_i),
$$

the class-specific score can be written as:

$$
\boxed{
Score_i=P(Class_i)\times IoU^{truth}_{pred}
}
$$

for each predicted bounding box.

This is a **multiplication**, not an addition.

A useful interpretation is:

- bounding-box confidence: **objectness + localization quality**;
- class probability: **what class the object is**;
- class-specific confidence score: **semantic confidence combined with objectness and localization quality**.

Because each of the 98 candidate boxes is combined with the class probabilities, each candidate box ultimately obtains a set of class-specific scores.

---

# 5. Network Architecture

## 5.1 Overall architecture

YOLOv1 uses a CNN with:

- 24 convolutional layers;
- 2 fully connected layers.

A high-level representation is:

```text
Input: 448 x 448 x 3
        |
        v
24 convolutional layers
        |
        v
Feature representation: approximately 7 x 7 x 1024
        |
        v
Fully Connected: 4096
        |
        v
Fully Connected: 7 x 7 x 30
        |
        v
Final detection tensor
```

The early convolutional layers extract visual features, while the fully connected layers convert the final representation into class and bounding-box predictions.

## 5.2 GoogLeNet inspiration

The architecture is inspired by GoogLeNet, but YOLOv1 does **not** directly use GoogLeNet's Inception modules.

Instead, it frequently applies the sequence:

$$
1\times1\text{ convolution}
\rightarrow
3\times3\text{ convolution}.
$$

The $1\times1$ convolution is used as a **reduction layer**.

Example:

$$
28\times28\times512
\rightarrow
1\times1\text{ conv}
\rightarrow
28\times28\times256.
$$

The operation does not simply discard half of the channels. Each output channel learns a weighted combination of the original input channels.

The main benefit is to reduce computation before an expensive $3\times3$ convolution.

For example, a direct $3\times3$ convolution from 512 channels to 512 channels requires, per spatial position:

$$
9\times512\times512=2,359,296
$$

multiplications.

Using a bottleneck:

$$
1\times1:\quad512\rightarrow256
$$

followed by:

$$
3\times3:\quad256\rightarrow512
$$

requires approximately:

$$
512\times256+9\times256\times512
=
1,310,720
$$

multiplications per position, substantially reducing the computation.

## 5.3 Spatial downsampling and increasing channels

As the signal passes through the network, spatial resolution is gradually reduced while the number of feature channels generally increases.

Conceptually:

```text
Spatial size:
448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7

Feature channels:
3 -> 64 -> 192 -> ... -> 512 -> 1024
```

This allows the network to move from detailed low-level visual information toward richer high-level features.

However, repeated downsampling also becomes one of the limitations of YOLOv1 because the final localization is based on relatively coarse spatial features.

## 5.4 Fully connected layers

After the convolutional feature extractor, the network uses:

$$
7\times7\times1024
\rightarrow
4096
\rightarrow
7\times7\times30.
$$

The first fully connected layer provides a global hidden representation. The final fully connected layer produces the detection tensor.

This design also means YOLOv1 contains a very large fully connected component compared with later fully convolutional YOLO designs.

---

# 6. Training Strategy

## 6.1 Classification pretraining

The first 20 convolutional layers are first trained on the ImageNet 1000-class classification task with an input resolution of:

$$
224\times224.
$$

The purpose of this stage is to provide useful visual representations before training the harder detection task.

Conceptually, classification pretraining allows the network to learn reusable features such as:

```text
edges
  ->
textures
  ->
shapes
  ->
object parts
  ->
higher-level visual representations
```

The detector therefore does not start with completely random visual features.

## 6.2 Conversion from classification to detection

After classification pretraining:

- the pretrained first 20 convolutional layers are reused;
- 4 new convolutional layers are added;
- 2 new fully connected layers are added with randomly initialized weights;
- input resolution is increased from $224\times224$ to $448\times448$.

Detection requires more **fine-grained visual information** than classification because the network must predict both:

$$
\text{WHAT is the object?}
$$

and:

$$
\text{WHERE exactly is the object?}
$$

Increasing the input resolution provides more detailed spatial information for localization.

The pretrained weights should be understood as a better initialization point. The optimizer does not conceptually know that some numbers are "pretrained"; it simply begins detection training from useful weight values instead of arbitrary random values. During detection training these parameters can continue adapting to the new task.

## 6.3 Bounding-box parameterization in training

The normalization and parameterization rules are encoded into the training target construction.

The model is not expected to invent:

$$
x,y=\text{cell offsets}
$$

or:

$$
w,h=\text{relative image dimensions}.
$$

Instead:

```text
Annotated ground-truth bbox
        |
        v
Target encoding according to YOLO rules
        |
        v
(x, y, w, h)
        |
        v
Loss compares target with network prediction
```

Repeated gradient-based optimization therefore teaches the network to directly output values in this representation.

At inference, the representation is decoded back into image-space coordinates.

---

# 7. Activation Functions

YOLOv1 uses:

- **Leaky ReLU** in the hidden layers;
- **linear activation** in the final layer.

The hidden activation is:

$$
\phi(x)=
\begin{cases}
x, & x>0,\\
0.1x, & \text{otherwise}.
\end{cases}
$$

## 7.1 Why nonlinearity is required

If all hidden layers were linear, stacking them would still be equivalent to a single linear transformation. Nonlinear activations allow the deep network to model complex mappings.

## 7.2 Why Leaky ReLU

Compared with standard ReLU, Leaky ReLU retains a small non-zero slope for negative activations:

$$
\phi'(x)=0.1,\qquad x<0.
$$

This can reduce the risk that a neuron permanently stops receiving useful gradient when it stays in the negative region.

This should **not** be interpreted as proof that Leaky ReLU is universally better than ReLU. The paper reports the design choice but does not establish that it is always optimal.

## 7.3 Why a linear final layer

The final layer performs continuous regression to detection outputs. A linear activation does not introduce an additional nonlinear transformation on those outputs.

An important consequence is that, although the target representation of $x,y,w,h$ is normalized, the raw network predictions are not mathematically hard-bounded to $[0,1]$ by the final activation.

Training encourages them toward the desired target range through the loss.

---

# 8. YOLOv1 Loss Function

This is one of the most important parts of YOLOv1.

The authors still use **sum-squared error (SSE)**, but they redesign it as a multi-part, conditional and re-weighted objective.

A useful conceptual decomposition is:

$$
\boxed{
L=
L_{xy}
+
L_{wh}
+
L_{conf,obj}
+
L_{conf,noobj}
+
L_{class}
}
$$

with different conditions and weights.

The main problems and corresponding design choices are discussed below.

## 8.1 Problem 1: localization error and classification error are treated equally by plain SSE

Plain SSE has the form:

$$
(\hat y-y)^2.
$$

Therefore, a numerical error of $0.1$ in a coordinate and a numerical error of $0.1$ in a class probability would both contribute:

$$
0.1^2=0.01.
$$

However, equal numerical errors in different output types do not necessarily cause equal degradation in detection performance.

For example, a small localization error may reduce IoU sufficiently to turn a correct detection into an incorrect detection, while a similar change in class probability might leave the predicted class unchanged.

The authors therefore assign a larger fixed weight to coordinate loss:

$$
\boxed{\lambda_{coord}=5.}
$$

This does not dynamically determine whether localization or classification is more important for each sample. It simply gives localization errors a larger fixed contribution to the training objective.

The classification-loss term itself is not multiplied by a corresponding special factor.

## 8.2 Problem 2: no-object confidence gradients can dominate training

In most images, many grid cells contain no object.

These predictions are repeatedly encouraged to produce confidence values near zero.

Because there are many more background/no-object predictions than object predictions, their accumulated gradients can overpower the useful training signal from object-containing regions.

Conceptually:

```text
Object-related training signal:
████

No-object confidence signal:
████████████████████████████████
```

This imbalance can make optimization unstable.

YOLOv1 reduces the contribution from no-object confidence error using:

$$
\boxed{\lambda_{noobj}=0.5.}
$$

Thus, no-object confidence terms receive only half the normal weight.

Important distinction:

> This weighting applies to the **confidence component**, not to every output of a background cell.

There is no coordinate or classification target to learn from a cell that contains no object.

## 8.3 Problem 3: equal absolute width/height error is not equally important for small and large boxes

Consider the same absolute width error of 10 pixels.

For a 300-pixel-wide box:

$$
\frac{10}{300}\approx3.3\%.
$$

For a 20-pixel-wide box:

$$
\frac{10}{20}=50\%.
$$

Plain SSE on width would treat both absolute errors similarly, although the effect on localization quality and IoU is much greater for the small box.

YOLOv1 partially addresses this by optimizing:

$$
(\sqrt{w}-\sqrt{\hat w})^2
$$

and:

$$
(\sqrt{h}-\sqrt{\hat h})^2
$$

instead of applying SSE directly to $w$ and $h$.

Because the square-root function is more sensitive at small values and compresses larger values, the loss becomes relatively more sensitive to width/height errors for small boxes.

The paper describes this as a **partial solution**, not a perfect one.

## 8.4 Problem 4: which bounding-box predictor should learn an object?

Each grid cell predicts two bounding boxes, but the model should not force both predictors to learn the same object coordinates.

During training, the predictor whose **current IoU with the ground-truth box is highest** is selected as the **responsible predictor**.

For example:

$$
IoU_1=0.72,\qquad IoU_2=0.35
$$

implies:

$$
\text{Predictor 1 is responsible.}
$$

Responsibility is based on IoU, **not on which predictor has the highest confidence value**.

This process encourages **specialization**. Over training, different predictors may become better at particular sizes, aspect ratios, or object patterns.

Specialization is not manually assigned in advance. It emerges from the repeated mechanism:

$$
\text{higher current IoU}
\rightarrow
\text{responsibility}
\rightarrow
\text{gradient update}
\rightarrow
\text{better prediction of similar boxes}.
$$

## 8.5 Logical view of the loss

A compact mental model is:

```text
ONE GRID CELL
|
+-- If the cell contains an object:
|   |
|   +-- Choose the bbox predictor with highest IoU to ground truth
|   |
|   +-- Responsible bbox:
|   |   +-- x,y coordinate loss          x lambda_coord = 5
|   |   +-- sqrt(w),sqrt(h) loss         x lambda_coord = 5
|   |   +-- object confidence loss       x 1
|   |
|   +-- Non-responsible bbox:
|   |   +-- no coordinate loss
|   |   +-- confidence treated through no-object term
|   |       with lambda_noobj = 0.5
|   |
|   +-- Cell class probabilities:
|       +-- classification loss          x 1
|
+-- If the cell contains no object:
    |
    +-- bbox predictor 1 confidence      x lambda_noobj = 0.5
    +-- bbox predictor 2 confidence      x lambda_noobj = 0.5
    +-- no coordinate loss
    +-- no classification loss
```

This is the easiest conceptual interpretation of the full YOLOv1 loss equation.

## 8.6 Indicator variables

The full loss uses indicator variables such as:

$$
\mathbb{1}^{obj}_{ij}.
$$

These can be interpreted as switches:

$$
\mathbb{1}=
\begin{cases}
1, & \text{condition is true},\\
0, & \text{condition is false}.
\end{cases}
$$

Therefore, if predictor $j$ in cell $i$ is responsible for a ground-truth object:

$$
\mathbb{1}^{obj}_{ij}=1,
$$

and its coordinate loss is active.

Otherwise:

$$
\mathbb{1}^{obj}_{ij}=0,
$$

and the coordinate term contributes nothing.

Similarly, classification error is only penalized when an object is present in the grid cell.

## 8.7 Compact mathematical form

Conceptually, YOLOv1 can be summarized as:

$$
\boxed{
L
=
5L_{xy}
+
5L_{\sqrt{w},\sqrt{h}}
+
L_{conf,obj}
+
0.5L_{conf,noobj}
+
L_{class}
}
$$

with responsibility and object-presence indicators determining which terms are active.

## 8.8 Relation between loss and Average Precision

YOLOv1 is evaluated using object-detection metrics such as Average Precision (AP) and mean Average Precision (mAP), but the training objective is SSE-based.

The paper explicitly recognizes that:

$$
\text{SSE objective}
\not\equiv
\text{AP/mAP objective}.
$$

The loss is easy to optimize, but it is **not perfectly aligned with maximizing Average Precision**.

YOLOv1 does **not** replace SSE with an AP-oriented loss. Instead, the authors retain SSE and redesign its weighting and conditional structure to better match detection needs.

Therefore, the mismatch between the trainable surrogate loss and the final evaluation metric remains a limitation.

This observation is conceptually related to the broader motivation behind later metric-oriented loss functions: the optimization objective and the evaluation metric do not always measure exactly the same thing.

---

# 9. Other Training Techniques

## 9.1 Learning schedule

The original YOLOv1 model is trained for approximately 135 epochs on PASCAL VOC 2007 and 2012 training/validation data.

The paper uses:

- batch size: $64$;
- momentum: $0.9$;
- weight decay: $0.0005$.

The learning rate is gradually increased at the beginning because a high initial learning rate can cause unstable gradients and divergence.

The reported schedule then uses approximately:

- $10^{-2}$ for 75 epochs;
- $10^{-3}$ for 30 epochs;
- $10^{-4}$ for 30 epochs.

## 9.2 Dropout and co-adaptation

To reduce overfitting, YOLOv1 uses dropout with rate:

$$
0.5
$$

after the first fully connected layer.

**Co-adaptation** means that neurons/features become excessively dependent on particular other neurons being present.

Dropout randomly disables a subset of activations during training. A feature therefore cannot always assume that another specific feature will be available, encouraging more robust representations.

This does not mean neurons should never cooperate. Rather, dropout discourages excessive dependence on a fixed set of neighboring activations.

## 9.3 Data augmentation

YOLOv1 applies extensive data augmentation, including:

- random scaling;
- random translation of up to 20% of the original image size;
- random exposure adjustment;
- random saturation adjustment by up to a factor of 1.5 in HSV color space.

These transformations increase training diversity and help reduce overfitting.

---

# 10. Inference

Inference requires only one network evaluation.

For PASCAL VOC:

$$
7\times7\times2=98
$$

candidate bounding boxes are produced.

Therefore:

$$
\boxed{98\text{ candidate bounding boxes per image}.}
$$

Each candidate box is associated with class-specific scores derived from its own confidence and the class-probability vector of its grid cell.

## 10.1 Score filtering

Conceptually, low-scoring box-class predictions can first be removed using a confidence/class-score threshold.

For example, if:

$$
\tau_{score}=0.5
$$

and a candidate has:

$$
Score_{dog}=0.45,\qquad
Score_{cat}=0.43,
$$

then neither class exceeds the threshold.

The threshold value is a post-processing setting and should not be confused with the IoU threshold used by NMS.

## 10.2 Non-Maximum Suppression

Large objects or objects close to grid boundaries may be detected by multiple neighboring predictions.

NMS removes duplicate detections.

Conceptually:

```text
Same-class candidate boxes
        |
        v
Choose highest-scoring box
        |
        v
Compare overlap (IoU) with remaining boxes
        |
        +-- high overlap -> suppress duplicate
        |
        +-- low overlap  -> keep
```

The NMS IoU threshold is distinct from the class-score threshold.

The original paper reports that NMS improves mAP by approximately 2–3%, although it is less critical for YOLO than for R-CNN or DPM.

---

# 11. Key Innovations Compared with Previous Detection Systems

YOLOv1 should not simply be described as "the first one-stage detector in history," because earlier systems such as OverFeat and MultiBox already contained related ideas.

A more accurate statement is:

> **YOLOv1 introduced a highly influential unified single-stage detection framework that directly predicts bounding boxes and class probabilities from full images in a single neural-network evaluation.**

Its major contributions include the following.

## 11.1 Unified detection formulation

YOLO directly predicts:

$$
\text{image}
\rightarrow
\text{boxes + confidence + classes}.
$$

This contrasts with multi-stage systems in which region proposals, feature extraction, classification, box regression, and duplicate-removal components are more explicitly separated.

## 11.2 Single full-image network evaluation

All bounding boxes and classes are predicted simultaneously from full-image features.

This enables global contextual reasoning.

## 11.3 Joint optimization

The main prediction system can be optimized jointly for the detection task rather than training a collection of independent subsystems.

## 11.4 Very small number of candidate boxes

On PASCAL VOC, YOLO predicts:

$$
98
$$

candidate boxes per image.

The paper contrasts this with approximately:

$$
2000
$$

Selective Search proposals used by R-CNN.

## 11.5 Real-time object detection

The base YOLO model runs at approximately:

$$
45\text{ FPS}
$$

in the paper's reported setup, while Fast YOLO reaches approximately:

$$
155\text{ FPS}.
$$

This makes speed a core design property rather than an optimization added to a complex detection pipeline afterward.

---

# 12. Advantages

## 12.1 High inference speed

The unified prediction pipeline requires only one main network evaluation and avoids thousands of independently processed region proposals.

The original paper reports:

- YOLO: 45 FPS;
- Fast YOLO: 155 FPS.

On PASCAL VOC 2007, the paper reports approximately:

- YOLO: 63.4% mAP;
- Fast YOLO: 52.7% mAP.

These numbers illustrate the intended speed–accuracy trade-off.

## 12.2 Global context

Because YOLO sees the entire image, it can use contextual information when deciding whether a region actually represents an object.

The paper reports substantially fewer background false positives than Fast R-CNN.

## 12.3 Generalizable object representation

The authors show that YOLO degrades less than R-CNN when models trained on natural imagery are evaluated on artwork datasets.

This suggests that the learned representation captures useful global properties such as object size, shape, and relationships rather than depending only on local image texture.

## 12.4 Simpler detection pipeline

Compared with traditional multi-component pipelines, YOLO integrates the core detection prediction process into one neural network and is therefore conceptually simpler to optimize.

---

# 13. Limitations of YOLOv1

## 13.1 Strong spatial constraints

Each grid cell predicts:

- only two bounding boxes;
- only one shared class-probability distribution.

Therefore, if several object centers fall into the same grid cell, YOLOv1 has insufficient prediction capacity to represent all of them independently.

This is particularly problematic for small objects that appear close together, such as a flock of birds.

A useful summary is:

> YOLOv1 is limited not only by whether an object is visually recognizable, but also by how many object centers fall into the same grid cell.

## 13.2 One shared class distribution per grid cell

Even though a cell predicts two bounding boxes, both boxes share the same class-probability vector.

Therefore, two different object classes whose centers occupy the same cell cannot be naturally represented as two fully independent class predictions.

## 13.3 Difficulty with unusual aspect ratios and configurations

An aspect ratio describes the width-to-height relationship:

$$
\text{Aspect Ratio}=\frac{w}{h}.
$$

Because YOLOv1 learns bounding-box prediction patterns from training data, it may generalize poorly when test objects appear with unusual:

- aspect ratios;
- poses;
- spatial configurations;
- shapes not well represented in the training distribution.

For example, a standing person usually produces a tall, narrow box, while an unusual horizontal pose may produce a very different box geometry.

## 13.4 Coarse features for localization

The architecture repeatedly downsamples the input:

$$
448\rightarrow224\rightarrow112\rightarrow56\rightarrow28\rightarrow14\rightarrow7.
$$

Downsampling is useful for computational efficiency and semantic abstraction, but it removes fine spatial details.

The final localization is therefore based on relatively coarse features.

This makes it harder to determine precise object boundaries and particularly affects small objects.

## 13.5 Localization is the major source of error

The paper's error analysis shows that localization errors account for a substantial portion of YOLO's mistakes.

This reflects several earlier limitations:

- coarse spatial features;
- grid constraints;
- difficulty with small objects;
- imperfect localization-aware loss design.

## 13.6 Small-box errors remain difficult

Although $\sqrt{w}$ and $\sqrt{h}$ partially compensate for the different impact of width/height errors on small and large boxes, the problem is not eliminated.

A small absolute error in a small bounding box can cause a much larger IoU decrease than the same error in a large box.

## 13.7 Training loss is not perfectly aligned with AP/mAP

YOLOv1 optimizes a re-weighted SSE objective, while detection performance is evaluated using AP/mAP.

The authors explicitly recognize that the two objectives are not perfectly aligned.

This mismatch remains unresolved in YOLOv1.

---

# 14. Typical Applications

The paper motivates real-time object detection for applications requiring rapid visual understanding, including:

- autonomous or intelligent driving systems;
- assistive vision systems;
- general-purpose responsive robotic systems;
- real-time webcam or video detection;
- interactive computer-vision applications.

YOLO's high inference speed is especially attractive in scenarios where the detector must respond immediately rather than process images offline.

For later versions of the report, this application section can be extended to modern domains such as UAV perception, traffic monitoring, industrial inspection, agriculture, and surveillance, but those broader applications should be supported by additional sources rather than attributed directly to the YOLOv1 paper.

---

# 15. Simple Demo / Implementation Plan

> **Note:** This section describes a proposed course demo. It should not be presented as an experiment already completed unless the implementation is actually run and measured.

A simple YOLOv1 demonstration can illustrate the complete inference pipeline on one or several images.

## 15.1 Demo objective

Given an input image:

1. resize it to $448\times448$;
2. run the pretrained YOLOv1 model;
3. obtain the $7\times7\times30$ prediction tensor;
4. decode the 98 candidate bounding boxes;
5. calculate class-specific scores;
6. remove low-score predictions;
7. apply NMS;
8. draw the final boxes and class labels.

Conceptual pipeline:

```text
Input image
    |
    v
Resize to 448 x 448
    |
    v
YOLOv1 forward pass
    |
    v
7 x 7 x 30 output
    |
    v
Decode 98 candidate boxes
    |
    v
Class-specific scores
    |
    v
Score threshold
    |
    v
NMS
    |
    v
Final detections
```

## 15.2 What the demo should visualize

A useful educational demo can display both:

- **pre-NMS candidate predictions**, showing that several boxes may correspond to the same object;
- **post-NMS final detections**, showing how duplicate predictions are removed.

This makes the relationship between network output and final detection easy to explain.

## 15.3 Suggested observations

The demo can report:

- detected class;
- class-specific confidence score;
- bounding-box coordinates;
- inference time;
- examples where small or clustered objects are difficult.

These observations can then be connected directly to the advantages and limitations discussed above.

---

# 16. Key Terminology

## Fine-grained visual information

Detailed spatial information needed to accurately determine object location, size, and boundaries.

Classification mainly asks:

$$
\text{WHAT?}
$$

Detection additionally requires:

$$
\text{WHERE?}
$$

Therefore detection generally benefits from finer spatial information.

## Parameterization

The predefined representation used to express a quantity.

In YOLOv1:

- $x,y$ are parameterized as grid-cell offsets;
- $w,h$ are represented relative to the whole image.

## Offset

A displacement relative to a reference point.

For YOLOv1, $x,y$ specify how far the box center lies from the origin/boundaries of the responsible grid cell.

## Aspect ratio

$$
\text{Aspect Ratio}=\frac{w}{h}.
$$

It describes the shape of a box rather than its absolute size.

## Specialization

The tendency of different bounding-box predictors to become better at different types of boxes through the highest-IoU responsibility mechanism.

## Co-adaptation

Excessive dependence between learned features or neurons. Dropout reduces this by randomly disabling activations during training.

## Ground-truth bounding box

The manually annotated target bounding box in the supervised training data against which model predictions are compared.

---

# 17. YOLOv1 in One Complete Flow

The entire model can be summarized as:

```text
Pre-YOLO problem:
multi-stage classifier-based detection pipelines
        |
        v
YOLO reformulation:
object detection as unified regression
        |
        v
Input full image: 448 x 448
        |
        v
24 convolutional layers
        |
        v
global feature representation
        |
        v
2 fully connected layers
        |
        v
7 x 7 x 30 predictions
        |
        +-- each cell: 2 bounding boxes
        |              each box: x,y,w,h,confidence
        |
        +-- each cell: one 20-class probability vector
        |
        v
Training:
- first 20 conv pretrained on ImageNet
- add 4 conv + 2 FC
- detection input raised to 448 x 448
- Leaky ReLU hidden layers
- linear final layer
- re-weighted multi-part SSE
        |
        +-- lambda_coord = 5
        +-- lambda_noobj = 0.5
        +-- sqrt(w), sqrt(h)
        +-- highest-IoU responsible predictor
        |
        v
Inference:
98 candidate boxes
        |
        v
class-specific scores
        |
        v
thresholding
        |
        v
NMS
        |
        v
final detections
```

---

# 18. Main Takeaway

YOLOv1's most important contribution is not a novel convolution operator but a new **system-level formulation of object detection**.

It demonstrates that object detection can be treated as a unified regression problem in which a single CNN reasons globally about the full image and simultaneously predicts object locations and classes.

Its key strengths are:

- simplicity;
- real-time speed;
- global contextual reasoning;
- end-to-end learning;
- strong generalization relative to contemporary detection pipelines.

Its main weaknesses are:

- coarse localization;
- strong grid-based spatial constraints;
- difficulty with small clustered objects;
- limited representation of multiple classes within one cell;
- difficulty with unusual object geometries;
- an SSE-based loss that only partially reflects the final detection metric.

These weaknesses are especially important for a survey because many later developments in the YOLO family can be understood as responses to limitations already visible in YOLOv1.

---

# 19. Suggested Mapping to the Final LaTeX Report

A convenient final report structure can be:

```text
1. Introduction
   1.1 Object Detection Background
   1.2 Motivation for YOLO
   1.3 Scope of the Survey

2. YOLOv1: Unified Real-Time Object Detection
   2.1 Problem Motivation
   2.2 Main Idea: Unified Detection
   2.3 Output Representation
   2.4 Network Architecture
   2.5 Training Strategy
       2.5.1 ImageNet Pretraining
       2.5.2 Bounding-box Parameterization
       2.5.3 Activation Functions
       2.5.4 Loss Function
       2.5.5 Dropout and Data Augmentation
   2.6 Inference
   2.7 Key Innovations
   2.8 Advantages
   2.9 Limitations
   2.10 Typical Applications
   2.11 Simple Demonstration

3. Evolution Beyond YOLOv1
   [To be added after studying later YOLO versions]

4. Comparative Discussion
   [To be added later]

5. Conclusion
   [To be completed after the full survey]
```

This structure keeps YOLOv1 sufficiently detailed while leaving a clean path for adding later architectural milestones.

---

# References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, **“You Only Look Once: Unified, Real-Time Object Detection,”** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779–788, 2016.

---

# Notes for Future Expansion

The following sections are intentionally left for later work:

- YOLOv2 / YOLO9000;
- YOLOv3;
- later architectural milestones;
- cross-version comparison of backbone, neck, head, prediction strategy, loss, and training techniques;
- evolution of small-object detection;
- evolution from grid-based prediction toward later assignment and end-to-end strategies;
- final implementation comparison;
- overall survey conclusion.

When those versions are added, YOLOv1 should remain the **baseline conceptual milestone** against which later changes are explained.
