# Phát hiện và theo vết đối tượng chuyển động

Chương trình tự động phát hiện các đối tượng **đang chuyển động** trong video/webcam,
nhận dạng tên đối tượng (tiếng Việt), gán ID và theo vết qua các khung hình.
Vật đứng yên (bàn ghế, xe đỗ, người đứng im...) không bị đóng khung.

## Quy trình xử lý (pipeline)

| Bước | Kỹ thuật | Vị trí trong code |
|---|---|---|
| 1. Phát hiện chuyển động | Trừ nền (running average) + trừ khung liên tiếp, ngưỡng T, lọc Gauss, phép mở/giãn | lớp `ChangeDetector` |
| 2. Nhận dạng | YOLOv8n (80 lớp COCO) | `model.track(...)` |
| 3. Theo vết | ByteTrack (Kalman + ghép IoU) | `tracker="bytetrack.yaml"` |
| 4. Lọc vật chuyển động | Tỉ lệ điểm chuyển động trong hộp ≥ 8%, ≥ 3/10 khung | lớp `MotionFilter`, hàm `drop_nested` |

## Cài đặt

Yêu cầu Python 3.8 trở lên.

```bash
pip install -r requirements.txt
```

Tệp `yolov8n.pt` đã có sẵn trong thư mục (nếu thiếu, Ultralytics sẽ tự tải về).

## Cách chạy

```bash
python motion_tracking.py                          # webcam
python motion_tracking.py --source demo.mp4        # tệp video
python motion_tracking.py --source demo.mp4 --output out.mp4   # lưu video kết quả
python motion_tracking.py --all                    # hiện cả vật đứng yên
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | `0` | đường dẫn video hoặc chỉ số webcam |
| `--model` | `yolov8n.pt` | mô hình YOLO (yolov8n/s/m.pt) |
| `--conf` | `0.4` | độ tin cậy tối thiểu |
| `--threshold` | `25` | ngưỡng T của bước phát hiện chuyển động |
| `--all` | tắt | hiện cả đối tượng đứng yên |
| `--output` | không | lưu video kết quả |
| `--no-display` | tắt | chạy không hiện cửa sổ |

Phím tắt: `q`/`ESC` thoát, `p`/`SPACE` tạm dừng, `r` học lại nền.
Lưu ý: giữ camera đứng yên khoảng 10 khung hình đầu để chương trình học nền.

## Tạo video thử nghiệm

```bash
python make_demo_video.py demo.mp4
```

Video 960×540, 25 fps, 200 khung: 2 người đứng yên (không được đóng khung),
1 người đi từ trái sang phải và 1 xe buýt chạy từ phải sang trái (phải được đóng khung).

## Đánh giá (FPS, Precision/Recall, IoU, kết quả trung gian)

```bash
python danh_gia.py --source demo.mp4 --gt-demo     # có precision/recall/IoU (so với vị trí thật)
python danh_gia.py --source video_khac.mp4         # dữ liệu mới: đo FPS + video trung gian
python danh_gia.py --source 0                      # webcam
python danh_gia.py --source demo.mp4 --gt-demo --imgsz 320   # thử ảnh nhỏ hơn cho máy yếu
```

`danh_gia.py` in ra FPS, thời gian từng bước và lưu `steps.mp4` gồm 4 ô:
khung gốc | mặt nạ chuyển động | mọi đối tượng YOLO+ByteTrack | kết quả sau lọc.

## Cấu trúc thư mục

```
motion_tracking.py   chương trình chính
make_demo_video.py   tạo video thử nghiệm
danh_gia.py          đo chỉ số đánh giá và xuất kết quả trung gian
yolov8n.pt           trọng số YOLOv8n (COCO)
demo.mp4             video thử nghiệm
requirements.txt     danh sách thư viện
```

## Giới hạn

- Camera phải đặt cố định (camera rung/quay sẽ báo chuyển động sai).
- Chỉ nhận dạng 80 lớp COCO.
- Có thể đổi ID khi đối tượng bị che khuất nhiều.
- Chạy chậm trên máy không có GPU nếu dùng ảnh đầu vào 640.
