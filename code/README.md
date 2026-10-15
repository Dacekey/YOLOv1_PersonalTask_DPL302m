# YOLOv1 bằng PyTorch: train trên Kaggle và xuất ZIP

Implementation phục vụ yêu cầu **Simple demo / implementation example** trong `../Subject_Requirement.txt`. Model giữ kiến trúc Figure 3: 24 conv, 4 max-pool, 2 FC, input RGB `448×448`, output `[N,7,7,30]`, 20 lớp VOC. Không thêm BN, anchors hoặc sigmoid/softmax cuối. Có **271.703.550 parameters**. Đây là thực nghiệm quy mô học tập; chưa tuyên bố tái lập benchmark paper.

## Chạy trên Kaggle

Remote GitHub của bạn có thư mục gốc giống `paper/Personal_Task`, tức chứa `code/`, `YOLO_Paper/`, `LaTeX/`… Notebook clone sparse **`code/`** từ repo đó.

1. Push `code/` lên GitHub, gồm các file Python, configs, notebook và `code/pretrain/extraction.conv.weights` nếu muốn lấy weights từ GitHub. File này khoảng 47,7 MB. Hoặc gắn nó dưới dạng Kaggle Dataset riêng; notebook ưu tiên weights đã attach. Không commit kết quả train/checkpoints.
2. Upload `notebooks/kaggle_train_yolov1.ipynb` lên Kaggle.
3. **Add Input**: [stpeteishii/pascal-voc-2007-dataset](https://www.kaggle.com/datasets/stpeteishii/pascal-voc-2007-dataset/data).
4. Bật **GPU** và **Internet**. Notebook dùng một GPU; không tự gộp VRAM của hai GPU.
5. Sửa cell settings:

```python
REPO_URL = "https://github.com/USERNAME/REPOSITORY.git"
REPO_REF = "FULL_COMMIT_HASH"  # commit chứa các file implementation
RUN_MODE = "smoke"             # smoke | overfit | train
DATASET_DIR = ""                # tự tìm; đặt rõ nếu attach nhiều bộ VOC
PRETRAINED_PATH = ""            # tự tìm attached weights hoặc code/pretrain/
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

`voc2007.json`: SGD momentum 0.9, decay 5e-4, LR phần mới 1e-3, LR conv pretrained bằng 0.1 lần, warmup 3 epoch, giảm 10 lần tại epoch index 30 và 45, 50 epoch, micro-batch 2, accumulation 8. AMP trên CUDA, clip gradient norm 10. Đây là **cấu hình khởi điểm**, cần pilot/điều chỉnh; không phải recipe paper gốc. Có thể tạo config pilot riêng 3–5 epoch trước lượt 50 epoch; phải chốt config trước khi resume.

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

`last.pt` lưu model, momentum optimizer, scheduler, AMP scaler, epoch/global step, history, Python/NumPy/PyTorch/CUDA RNG và DataLoader generator. Chỉ checkpoint **cuối epoch** là điểm resume; không có mid-epoch replay. Với `num_workers=0`, có kiểm tra resume khớp lượt chạy liên tục trên mạng nhỏ. CUDA có thể vẫn có sai khác số học giữa phần cứng/runtime.

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

Tests bao phủ target encoding/collision, tọa độ XML, loss/gradient, responsible assignment, difficult mask, IoU/NMS, AP/duplicate/difficult, weights 16 conv, code cells notebook và orchestration train→resume→evaluate→ZIP trên mạng nhỏ. Báo cáo kiểm tra thực tế ở `implementation_verification.json`; numerical smoke của model đầy đủ được ghi riêng với đúng dữ liệu và phần cứng sử dụng. Không có kết quả GPU Kaggle hoặc model hội tụ cho tới khi bạn chạy training thực tế.
