# YOLOv1 bằng PyTorch: train trên Kaggle và xuất ZIP

Implementation phục vụ yêu cầu **Simple demo / implementation example** trong `../Subject_Requirement.txt`. Model giữ kiến trúc Figure 3: 24 conv, 4 max-pool, 2 FC, input RGB `448×448`, output `[N,7,7,30]`, 20 lớp VOC. Không thêm BN, anchors hoặc sigmoid/softmax cuối. Có **271.703.550 parameters**. Đây là thực nghiệm quy mô học tập; chưa tuyên bố tái lập benchmark paper.

## Chạy trên Kaggle

Remote GitHub của bạn có thư mục gốc giống `paper/Personal_Task`, tức chứa `code/`, `YOLO_Paper/`, `LaTeX/`… Notebook clone sparse **`code/`** từ repo đó.

1. Push `code/` lên GitHub, gồm các file Python, configs, notebook và `code/pretrain/extraction.conv.weights` nếu muốn lấy weights từ GitHub. File này khoảng 47,7 MB. Hoặc gắn nó dưới dạng Kaggle Dataset riêng và đặt `PRETRAINED_PATH` thành đường dẫn tuyệt đối tới file đó. Không commit kết quả train/checkpoints.
2. Upload `notebooks/kaggle_train_yolov1.ipynb` lên Kaggle.
3. **Add Input**: [stpeteishii/pascal-voc-2007-dataset](https://www.kaggle.com/datasets/stpeteishii/pascal-voc-2007-dataset/data).
4. Chọn **GPU T4 ×2** và bật **Internet**. Notebook yêu cầu đúng 2 GPU, khởi chạy `torch.distributed.run --nproc_per_node=2` và DDP/NCCL, một process trên mỗi GPU. VRAM không được gộp thành một vùng 32 GB.
5. Sửa cell settings:

```python
REPO_URL = "https://github.com/Dacekey/YOLOv1_PersonalTask_DPL302m.git"
REPO_REF = "FULL_COMMIT_HASH"  # commit chứa các file implementation
RUN_MODE = "smoke"             # smoke | overfit | train
DATASET_DIR = ""                # tự tìm; đặt rõ nếu attach nhiều bộ VOC
PRETRAINED_PATH = "code/pretrain/extraction.conv.weights"  # tương đối từ root repo
RESUME_PATH = ""                # để trống ở lượt đầu
MAX_HOURS = 9.0                 # chọn thấp hơn giới hạn phiên thực tế
RUN_NAME = "yolov1_smoke_01"     # mỗi lượt mới dùng tên mới
```

6. Chạy `smoke` trước: 1 epoch, 2 ảnh train + 2 ảnh validation. Kiểm tra file ZIP và log. Tiếp theo chạy `overfit` (16 ảnh train, đánh giá lại trên chính 16 ảnh đó; dropout/augmentation tắt). Cuối cùng chạy `train` để dùng toàn bộ train/val. Dùng phiên mới hoặc Restart Session khi thay phiên bản code.
7. Với lượt chính thức, dùng **Save Version → Save & Run All**. Sau khi hoàn tất, tải `/kaggle/working/<RUN_NAME>.zip` từ Output. Notebook cũng tạo link tải.

Repo public clone trực tiếp được. Repo private cần cấu hình xác thực qua Kaggle Secrets hoặc quyền truy cập tương ứng; notebook hiện không nhúng token. Không đặt token trong `REPO_URL`.

## Dataset đã kiểm tra trực tiếp

Ngày 2026-10-15, API download của Kaggle trả về:

- Root: `VOCdevkit2007/VOC2007/`.
- `ImageSets/Main/train.txt`: **2.501** ID.
- `ImageSets/Main/val.txt`: **2.510** ID.
- `ImageSets/Main/trainval.txt`: **5.011** ID.
- Đọc được XML và JPEG mẫu `000005`.
- Request `ImageSets/Main/test.txt` trả HTTP 404. Vì vậy pipeline mặc định đánh giá **validation**, không gọi đó là test. Khi dataset thay đổi, `dataset.py` kiểm tra lại tại runtime.

Chi tiết trong `dataset_check.json`. Notebook đọc ảnh/XML trực tiếp từ `/kaggle/input`; không tải lại toàn dataset. Không random split và không tự lấy test làm validation. Mặc định train/val phải không giao nhau. Nếu cần test cuối, attach một bản VOC2007 test có ảnh/XML và split riêng, rồi chạy `evaluate.py --split test` với đường dẫn root đó.

## Pretrained weights: chỉ 16 conv

File đã kiểm tra:

```text
pretrain/extraction.conv.weights
bytes: 47657232
SHA-256: 048f832fbaddec03256f2c1cf380e5694d6b580f24639515587daa358ff32e9f
header: Darknet 0.1.0, 16 bytes
payload: 11.914.304 float32 values, bias rồi weight mỗi conv
```

`pretrained.py` nạp đúng `conv01`–`conv16`; không dư byte, không tự bỏ qua lỗi checksum. `conv17`–`conv24` và hai FC vẫn khởi tạo ngẫu nhiên. Giữ nguyên kiến trúc model. Binary không tự chứng minh dataset/lịch sử training; không ghi file này là “đã xác minh ImageNet pretraining đủ 20 conv”. Nguồn ứng viên đã tìm: [Oxford Straight To Shapes](https://robots.ox.ac.uk/~tvg/projects/StraightToShapes/index.php).

```bash
python pretrained.py pretrain/extraction.conv.weights
```

## Training và quy ước

- Ảnh RGB float32 `[0,1]`, resize trực tiếp `448×448`, không áp ImageNet mean/std. Flip ngang và thay đổi brightness/saturation nhẹ khi train; không dùng crop/mosaic/multiscale trong bản đầu.
- Cell output: `[20 class values, box0(x,y,sqrt_w,sqrt_h,confidence), box1(...)]`. Output tuyến tính. Tâm box dùng offset theo cell; width/height chuẩn hóa toàn ảnh và decode bằng bình phương raw size outputs.
- Mỗi cell dùng chung class vector. Target giữ object không-difficult lớn nhất trong cell, tie theo thứ tự XML. Collision được thống kê. **Evaluation giữ toàn bộ ground truth**, gồm object bị collision.
- Responsible predictor là box có IoU cao nhất; IoU confidence target được detach. Loss SSE chia theo số ảnh, `lambda_coord=5`, `lambda_noobj=0.5`. Các box không chịu trách nhiệm nhận confidence target 0, kể cả predictor còn lại trong cell có object.
- Cell chỉ có object `difficult` được bỏ qua no-object loss. Đánh giá VOC bỏ qua detection khớp GT difficult theo protocol; duplicate detection trên GT thường là false positive.
- Decode dùng class × confidence sau khi chặn giá trị âm riêng từng thành phần; không để hai giá trị âm nhân thành score dương. Không clip score về 1. NMS theo lớp. Validation dùng score threshold `0.001`, NMS `0.5`, tối đa `300` detection/ảnh; hình demo dùng `0.25`. Đây là các lựa chọn implementation được lưu cùng metrics.
- Metric: **VOC 2007 11-point AP, IoU > 0.5** theo điều kiện strict của devkit. Class không có GT thường được ghi AP `null` và loại khỏi trung bình; `classes_with_gt` giúp nhận ra giới hạn của smoke/overfit. Train/val đầy đủ có đủ 20 lớp.

Cấu hình một GPU `voc2007.json`: SGD momentum 0.9, decay 5e-4, LR phần mới 1e-3, LR conv pretrained bằng 0.1 lần, warmup 3 epoch, giảm 10 lần tại epoch index 30 và 45, 50 epoch, micro-batch 2, accumulation 8. AMP trên CUDA, clip gradient norm 10. Đây là **cấu hình khởi điểm**, cần pilot/điều chỉnh; không phải recipe paper gốc. Có thể tạo config pilot riêng 3–5 epoch trước lượt 50 epoch; phải chốt config trước khi resume.

Model lớn: ~1 GiB weights FP32, thêm gradients, momentum và activations. Đo bằng smoke trên GPU thực tế. Nếu OOM, giảm `batch_size` xuống 1; tăng `accumulation` để giữ effective batch. Không cài `requirements.lock.txt` CPU lên Kaggle. Giữ CUDA PyTorch có sẵn; cần torch >= 2.6. Dependency bổ sung ở `requirements-train.txt`.

## Chạy bằng CLI

Từ thư mục `code/`, trong môi trường có CUDA PyTorch phù hợp:

```bash
python -m pip install -r requirements-train.txt
python dataset.py --data /path/to/dataset --output dataset_inspection.json
python train.py --config configs/smoke.json --data /path/to/dataset \
  --pretrained pretrain/extraction.conv.weights --device cuda \
  --output /path/to/runs/smoke_01 --zip /path/to/smoke_01.zip
python train.py --config configs/voc2007.json --data /path/to/dataset \
  --pretrained pretrain/extraction.conv.weights --device cuda \
  --output /path/to/runs/train_01 --max-hours 9 --zip /path/to/train_01.zip
python evaluate.py --data /path/to/dataset --checkpoint /path/to/best.pt \
  --split val --output /path/to/evaluation --device cuda
python predict.py --checkpoint /path/to/best.pt --image /path/to/image.jpg \
  --output /path/to/prediction.jpg --device cuda
```

Không đổi config trong lúc resume. `--random-init` là lựa chọn tường minh nếu muốn thí nghiệm không pretrain; không phải mặc định.

## Resume và thời gian phiên

`last.pt` lưu model, momentum optimizer, scheduler, AMP scaler, epoch/global step, history, Python/NumPy/PyTorch/CUDA RNG và DataLoader generator. Khi DDP, lưu RNG/generator riêng từng rank; resume yêu cầu cùng `world_size`. Checkpoint một GPU trước đây vẫn dùng inference được, nhưng không resume trực tiếp thành hai GPU. Chỉ checkpoint **cuối epoch** là điểm resume; không có mid-epoch replay. Với `num_workers=0`, có kiểm tra resume khớp lượt chạy liên tục trên mạng nhỏ. CUDA có thể vẫn có sai khác số học giữa phần cứng/runtime.

Để tiếp tục trên Kaggle: giải nén ZIP trước, upload hoặc attach thư mục checkpoint thành input (gồm `last.pt` và `best.pt`), đặt `RESUME_PATH` tới `last.pt`, chọn `RUN_NAME` mới. Notebook đọc lại config từ checkpoint. Giữ đủ dataset và các split. `best.pt` dùng inference, không đủ optimizer để resume. Nếu chỉ có `last.pt`, chương trình vẫn resume; nếu không tạo best mới, evaluation ghi rõ dùng last thay vì best bị thiếu.

`MAX_HOURS` là ngân sách chủ động: trước mỗi epoch, chương trình dùng thời gian epoch trước và phần dự phòng export để quyết định dừng. Epoch đầu chưa có ước lượng; phần dự phòng mặc định 600 giây có thể cần tăng theo pilot. SIGINT/SIGTERM yêu cầu dừng cuối epoch. Checkpoint và ZIP không bảo đảm được tạo khi Kaggle/OOM killer kết thúc cứng tiến trình hoặc máy hết disk. Khi lỗi Python sau lúc train bắt đầu, chương trình cố xuất ZIP chẩn đoán và vẫn báo lỗi.

## Nội dung ZIP

```text
<RUN_NAME>/
  checkpoints/best.pt          # model + config + epoch, inference
  checkpoints/last.pt          # đủ state để resume cuối epoch
  config.json
  run_manifest.json           # commit, versions, GPU, IDs, weights checksum, status
  dataset_inspection.json     # notebook tạo trước train
  history.csv
  training.log
  metrics.json
  per_class_ap.csv
  predictions.json
  examples/*.jpg              # GT và prediction cạnh nhau
  curves.png
  source/*.py                 # snapshot code thực sự sử dụng
  artifact_checksums.json
```

Weights lớn nên ZIP không giảm nhiều dung lượng. Chỉ giữ best/last; không lưu một checkpoint mỗi epoch. ZIP không chứa dataset hoặc repo đầy đủ. Trạng thái trong `run_manifest.json` phân biệt completed, dừng theo thời gian/signal và failed; ZIP tồn tại không tự chứng minh train thành công. Overfit và smoke chỉ dùng kiểm tra, không phải kết quả tổng quát hóa.

## Kiểm tra local

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python inspect_model.py --meta
```

Tests gồm cả DDP hai process CPU/Gloo: accumulation với nhóm cuối không đủ batch, sampler padding, rank-zero validation/export và resume khớp tham số lượt chạy liên tục. Tests bao phủ target encoding/collision, tọa độ XML, loss/gradient, responsible assignment, difficult mask, IoU/NMS, AP/duplicate/difficult, weights 16 conv, code cells notebook và orchestration train→resume→evaluate→ZIP trên mạng nhỏ. Báo cáo kiểm tra thực tế ở `implementation_verification.json`; numerical smoke của model đầy đủ được ghi riêng với đúng dữ liệu và phần cứng sử dụng. Không có kết quả GPU Kaggle hoặc model hội tụ cho tới khi bạn chạy training thực tế.

## Cấu hình hai GPU T4 trong notebook

Notebook mặc định dùng `smoke_ddp.json`, `overfit_ddp.json`, `voc2007_ddp.json`. Config một GPU cũ vẫn còn cho CLI chạy riêng.

| Mode | Micro-batch / GPU | Accumulation | Global effective batch | Workers / rank |
| --- | ---: | ---: | ---: | ---: |
| smoke | 1 | 1 | 2 | 0 |
| overfit | 1 | 2 | 4 | 0 |
| train | 1 | 8 | 16 | 1 |

DDP đồng bộ gradient ở cuối mỗi nhóm accumulation bằng `no_sync()` cho các micro-batch trước đó. Log train loss được tổng hợp từ cả hai rank. `DistributedSampler.set_epoch()` thay shuffle mỗi epoch. Với 2.501 ảnh train, sampler padding một ảnh để mỗi rank có 1.251 mẫu; số duplicate được ghi trong manifest. Validation **chỉ rank 0**, dùng toàn bộ val không padding/duplicate. Chỉ rank 0 ghi checkpoint, ảnh, metrics, log và ZIP; rank 1 chờ khi validation/export. Dataloader có workers trong DDP dùng `spawn` để tránh fork sau khởi tạo NCCL.

Đường dẫn sau clone:

```text
/kaggle/temp/personal_yolov1_repo/           # root của repo GitHub = paper/Personal_Task
  code/train.py
  code/configs/voc2007_ddp.json
  code/pretrain/extraction.conv.weights     # PRETRAINED_PATH mặc định
```

`PRETRAINED_PATH` tương đối được nối với `REPO`, không nối với `CODE` hoặc current working directory. Nếu gắn weights bằng Kaggle Dataset riêng, đặt đường dẫn tuyệt đối `/kaggle/input/.../extraction.conv.weights`. Code vẫn kiểm tra đúng checksum trước train. Remote hiện có file weights tại đúng `code/pretrain/extraction.conv.weights`; cần push các thay đổi DDP trước khi chạy notebook mới.

CLI tương đương hai GPU:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --config configs/voc2007_ddp.json --data /path/to/VOC2007 \
  --pretrained pretrain/extraction.conv.weights --device cuda \
  --output /path/to/runs/ddp_train_01 --zip /path/to/ddp_train_01.zip
```

Không nhân learning rate thêm với số GPU: effective batch chính thức vẫn là 16 như cấu hình một GPU ban đầu. Chưa có benchmark GPU/T4 hoặc bảo đảm tốc độ tăng gấp đôi; kiểm tra thực tế bằng `smoke` trên Kaggle trước.

## AMP overflow và log gradient non-finite

Log `logs/yolov1-pascal-voc-2007.log` của lần chạy đầu cho thấy hai T4/NCCL đã khởi động, dataset và pretrained path đúng. Lỗi đầu tiên nằm ở gradient clipping trong bước update đầu, khi AMP bật. Phiên bản cũ dừng bằng `error_if_nonfinite=True` trước khi xử lý loss-scale overflow; `ChildFailedError` và lỗi notebook chỉ là hệ quả.

Bản sửa đặt `amp_init_scale=128` trong configs (thay vì mặc định GradScaler 65536). Sau unscale, `precision.py` kiểm tra norm trước khi sửa gradient. Nếu non-finite khi AMP bật, mọi rank cùng bỏ qua update và giảm loss scale; không cập nhật weights/momentum từ gradient lỗi. Gradient hữu hạn được clip rồi mới optimizer step. Log ghi scale trước/sau, `history.csv` ghi số optimizer steps thành công và AMP skips; `global_step` chỉ đếm update thành công. Nếu cả epoch không có update nào thành công, chương trình báo lỗi rõ ràng, không xuất một lượt train thành công giả. Gradient non-finite ở FP32 vẫn làm fail để chẩn đoán.

Để chạy lại: push cả `precision.py`, `train.py` và configs; chọn commit mới trong `REPO_REF` (hoặc `main`), đặt tên mới như `yolov1_voc2007_smoke_02`, để `RESUME_PATH=""`. Lần fail trước ở update đầu nên không có checkpoint hoàn tất epoch để resume. Nếu cần cô lập lỗi số học còn lại, tạo config `amp=false` để chạy FP32; chưa cần giảm learning rate chỉ vì overflow ở update đầu.

Tham khảo hành vi loss scaling/clipping: [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). Tests tái hiện finite YOLO loss nhưng backward FP16 overflow với scale 65536, kiểm tra backoff/recovery, scale 128, clipping hữu hạn và skip đồng bộ giữa hai rank kể cả chỉ một rank bị inject overflow. Đây là kiểm tra local CPU FP16/Gloo; không thay cho chạy lại trên T4.

## Notebook inference để chọn 3 ảnh minh họa

Notebook: `notebooks/kaggle_inference_yolov1.ipynb`. Push cả `inference_demo.py` cùng notebook lên GitHub trước khi chạy.

1. Upload notebook lên Kaggle, bật Internet và GPU (inference chỉ cần một GPU; nếu chọn T4 ×2 thì dùng GPU 0).
2. Add Input dataset `stpeteishii/pascal-voc-2007-dataset` và output của **lượt train đầy đủ 50 epoch**. Có thể gắn thư mục chứa `checkpoints/best.pt` hoặc file ZIP train. Không cần `extraction.conv.weights` cho inference.
3. `CHECKPOINT_PATH=""` tự tìm duy nhất một `best.pt` hoặc ZIP chứa nó. Nếu có nhiều checkpoint/ZIP, đặt đường dẫn tuyệt đối `/kaggle/input/.../best.pt` hoặc `/kaggle/input/.../training.zip`. Tên ZIP có chữ `smoke` không ảnh hưởng; saved config mới xác định loại run. Checkpoint overfit bị từ chối để tránh dùng nhầm.
4. Giữ `NUM_EXAMPLES=3`, `DEMO_SCORE_THRESHOLD=0.25`, đặt `RUN_NAME` mới. `REPO_REF` phải là main/commit có module inference mới. Chạy **Save Version → Save & Run All**, tải ZIP từ Output.

Notebook chạy inference mới trên **toàn bộ val**, không lấy lại prediction cũ làm kết quả inference. Metric chính giữ VOC2007 11-point mAP@0.5 với score threshold `0.001`; đồng thời xuất micro precision/recall/F1, TP/FP/FN, mean IoU của TP ở ngưỡng hiển thị `0.25` và bảng threshold 0.05/0.10/0.25/0.50. Không đổi threshold tự động để làm đẹp ảnh.

Chọn ảnh minh họa ưu tiên có TP đúng lớp với IoU > 0.5, precision/IoU tốt, và khác lớp khi có thể. Nếu thiếu ảnh có TP thì bổ sung ảnh có bbox bất kỳ, ghi nguyên trạng thái FP/ignored. GT chỉ dùng sau inference để đánh giá và chọn ví dụ, không tham gia dự đoán. Ba ảnh là ví dụ được chọn có chủ đích, không đại diện độ chính xác toàn bộ val. Metric toàn val được giữ riêng với metric từng ảnh được chọn.

ZIP inference **không chứa weights/dataset**, gồm:

- `metrics.json`, `per_class_ap.csv`, `per_class_ap.png/pdf`: kết quả AP/mAP toàn val.
- `operating_point_metrics.json`, `threshold_metrics.csv`, `per_image_metrics.csv`: precision/recall/F1 và các thống kê ngưỡng.
- `selected_examples.json/csv`, `examples/selected_detections.csv`: ID, class, score, IoU, TP/FP/ignored của ví dụ.
- Ba ảnh `*_comparison.jpg` gồm ảnh gốc / GT / prediction; thêm ảnh input, prediction riêng và `contact_sheet.jpg`.
- `predictions.json`, `summary.txt`, manifest có checksum checkpoint, split/IDs, epoch, runtime, selection policy, source snapshot và checksum artifacts.

Model là code tự định nghĩa và recipe 50 epoch cho demo; không tuyên bố tái lập benchmark paper. Score không phải xác suất đã được hiệu chuẩn. Thời gian evaluation có cả đọc ảnh/decoding/AP, không được báo thành model-only FPS.

CLI tương đương:

```bash
python inference_demo.py --data /path/to/VOC2007 \
  --checkpoint /path/to/training_output.zip --device cuda \
  --count 3 --demo-threshold 0.25 \
  --output /path/to/val_demo --zip /path/to/val_demo.zip
```

Cờ `--limit` chỉ dành cho kiểm tra local và sẽ ghi rõ `limited validation subset`; notebook mặc định không dùng cờ này.
