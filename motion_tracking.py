"""
Phát hiện, nhận dạng và theo vết đối tượng chuyển động
Đề tài 14 - Tự động hoàn toàn: vật nào chuyển động thì hiện 1 khung kèm tên.

Quy trình:
  Bước 1 - PHÁT HIỆN CHUYỂN ĐỘNG (Change Detection, theo báo cáo)
     Hiệu pixel giữa khung hiện tại và nền > ngưỡng T => vùng chuyển động
  Bước 2 - NHẬN DẠNG đối tượng (YOLOv8)
     Tìm người, xe, động vật, đồ vật... trong khung hình và cho biết tên
  Bước 3 - THEO VẾT (ByteTrack)
     Mỗi đối tượng giữ 1 ID cố định qua các khung hình
  Bước 4 - LỌC: chỉ hiển thị đối tượng ĐANG CHUYỂN ĐỘNG
     (khung nằm trên vùng chuyển động của bước 1, hoặc tâm đã dịch đi đáng kể).
     Vật đứng yên từ đầu (bàn ghế, xe đỗ...) không bị đóng khung.

Cách chạy:
  python motion_tracking.py                       (webcam)
  python motion_tracking.py --source video.mp4    (file video)
  python motion_tracking.py --all                 (hiện cả vật đứng yên)

  python motion_tracking.py --fullscreen          (toàn màn hình)

Phím tắt:  q / ESC: thoát   p / SPACE: tạm dừng   r: học lại nền   f: bật/tắt toàn màn hình
Cửa sổ có thể kéo giãn bằng chuột hoặc bấm nút phóng to (□) của Windows.
"""

import argparse
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Tên tiếng Việt cho các lớp thường gặp của bộ dữ liệu COCO (lớp khác giữ tên gốc)
VI_NAMES = {
    "person": "Người", "bicycle": "Xe đạp", "car": "Ô tô", "motorcycle": "Xe máy",
    "bus": "Xe buýt", "truck": "Xe tải", "train": "Tàu hoả", "boat": "Thuyền",
    "airplane": "Máy bay", "bird": "Chim", "cat": "Mèo", "dog": "Chó", "horse": "Ngựa",
    "sheep": "Cừu", "cow": "Bò", "elephant": "Voi", "bear": "Gấu", "zebra": "Ngựa vằn",
    "giraffe": "Hươu cao cổ", "backpack": "Ba lô", "umbrella": "Ô (dù)", "handbag": "Túi xách",
    "suitcase": "Vali", "frisbee": "Đĩa ném", "sports ball": "Quả bóng", "kite": "Diều",
    "skateboard": "Ván trượt", "bottle": "Chai", "cup": "Cốc", "fork": "Dĩa", "knife": "Dao",
    "spoon": "Thìa", "bowl": "Bát", "banana": "Chuối", "apple": "Táo", "orange": "Cam",
    "chair": "Ghế", "couch": "Ghế sofa", "bed": "Giường", "dining table": "Bàn",
    "tv": "Tivi", "laptop": "Laptop", "mouse": "Chuột", "remote": "Điều khiển",
    "keyboard": "Bàn phím", "cell phone": "Điện thoại", "book": "Sách", "clock": "Đồng hồ",
    "scissors": "Kéo", "teddy bear": "Gấu bông", "toothbrush": "Bàn chải",
}

# Mỗi nhóm đối tượng 1 màu (BGR)
GROUP_COLORS = {"person": (0, 200, 0)}
VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck", "train", "boat", "airplane"}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def class_color(name):
    if name in GROUP_COLORS:
        return GROUP_COLORS[name]
    if name in VEHICLES:
        return (0, 140, 255)
    if name in ANIMALS:
        return (255, 0, 200)
    return (255, 160, 0)


# ---------------------------------------------------------------------------
# Bước 1: Change Detection - phát hiện thay đổi điểm ảnh
# ---------------------------------------------------------------------------
class ChangeDetector:
    """Trừ nền với ngưỡng T; nền được cập nhật chậm ở chỗ không có chuyển động."""

    def __init__(self, threshold=25, learning_rate=0.03, warmup=10):
        self.threshold = threshold          # ngưỡng T
        self.learning_rate = learning_rate  # tốc độ cập nhật nền
        self.warmup = warmup                # số khung đầu dùng để học nền
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self):
        self.background = None
        self.prev_gray = None
        self.count = 0

    @property
    def ready(self):
        return self.count > self.warmup

    def apply(self, frame):
        """Trả về mask vùng chuyển động (255 = chuyển động)."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        self.count += 1
        if self.background is None:
            self.background = gray.astype(np.float32)
        if not self.ready:
            cv2.accumulateWeighted(gray, self.background, 0.5)
            self.prev_gray = gray
            return np.zeros_like(gray)

        # Hiệu pixel với nền, so với ngưỡng T, rồi loại bỏ nhiễu
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # Chỉ giữ chỗ đang thay đổi giữa 2 khung liên tiếp (loại "bóng ma" của nền)
        motion = cv2.absdiff(gray, self.prev_gray) > self.threshold // 2
        motion = cv2.dilate(motion.astype(np.uint8) * 255, self.kernel, iterations=4)
        self.prev_gray = gray
        cv2.accumulateWeighted(gray, self.background, self.learning_rate, cv2.bitwise_not(motion))
        return cv2.dilate(mask & motion, self.kernel, iterations=2)


# ---------------------------------------------------------------------------
# Bước 4: xác định đối tượng nào đang chuyển động
# ---------------------------------------------------------------------------
class MotionFilter:
    def __init__(self, min_overlap=0.08, min_hits=3, hold_frames=25):
        self.min_overlap = min_overlap  # tỉ lệ pixel chuyển động tối thiểu trong vùng của vật
        self.min_hits = min_hits        # số khung có chuyển động (trong 10 khung gần nhất)
        self.hold_frames = hold_frames  # vật dừng lại vẫn hiện thêm ngần này khung
        self.hits = defaultdict(lambda: deque(maxlen=10))  # lịch sử có/không chuyển động
        self.last_moving = {}  # ID -> chỉ số khung gần nhất xác nhận chuyển động

    def moving_ids(self, boxes, ids, motion_mask, frame_idx):
        """Trả về tập ID đang chuyển động.

        Mỗi pixel chuyển động chỉ thuộc về khung NHỎ NHẤT chứa nó (thường là vật ở phía
        trước). Nhờ vậy người đi ngang qua trước 1 người đứng yên sẽ không làm người
        đứng yên bị tính là chuyển động."""
        owner = np.full(motion_mask.shape, -1, np.int32)
        order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        for i in order:  # khung lớn vẽ trước, khung nhỏ đè lên sau
            x1, y1, x2, y2 = boxes[i]
            owner[y1:y2, x1:x2] = i

        moving = set()
        for i, tid in enumerate(ids):
            own = owner == i
            area = int(own.sum())
            overlap = cv2.countNonZero(motion_mask & own.astype(np.uint8)) / area if area else 0.0
            self.hits[tid].append(overlap >= self.min_overlap)
            if sum(self.hits[tid]) >= self.min_hits:
                self.last_moving[tid] = frame_idx
            last = self.last_moving.get(tid)
            if last is not None and frame_idx - last <= self.hold_frames:
                moving.add(tid)
        return moving


def drop_nested(objects, ratio=0.85):
    """Bỏ khung nằm gần trọn trong khung lớn hơn (vd: người ngồi trong xe) -> 1 vật 1 khung."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    keep = []
    for o in objects:
        b = o[0]
        inside = False
        for other in objects:
            c = other[0]
            if other is o or area(c) <= area(b):
                continue
            ix = max(0, min(b[2], c[2]) - max(b[0], c[0]))
            iy = max(0, min(b[3], c[3]) - max(b[1], c[1]))
            if ix * iy >= ratio * area(b):
                inside = True
                break
        if not inside:
            keep.append(o)
    return keep


# ---------------------------------------------------------------------------
# Vẽ kết quả (dùng PIL để hiện được tiếng Việt có dấu)
# ---------------------------------------------------------------------------
def load_font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw(frame, objects, font, status):
    """objects: danh sách (box, tên, độ tin cậy, màu, ID)."""
    for (x1, y1, x2, y2), _, _, color, _ in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), name, conf, color, tid in objects:
        text = f"{name} #{tid} {conf:.0%}"
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
        ty = y1 - th - 6 if y1 - th - 6 >= 0 else y1
        d.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color[::-1])
        d.text((x1 + 4, ty + 2), text, font=font, fill=(255, 255, 255))
    d.text((10, 8), status, font=font, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phat hien, nhan dang & theo vet doi tuong chuyen dong")
    ap.add_argument("--source", default="0", help="duong dan video hoac chi so webcam (mac dinh 0)")
    ap.add_argument("--model", default="yolov8n.pt", help="mo hinh YOLO (yolov8n/s/m.pt)")
    ap.add_argument("--conf", type=float, default=0.4, help="do tin cay toi thieu de nhan dang")
    ap.add_argument("--threshold", type=int, default=25, help="nguong T cho change detection")
    ap.add_argument("--all", action="store_true", help="hien ca doi tuong dung yen")
    ap.add_argument("--output", default=None, help="luu video ket qua (vd: out.mp4)")
    ap.add_argument("--no-display", action="store_true", help="khong hien cua so (chay ngam)")
    ap.add_argument("--fullscreen", action="store_true", help="mo cua so toan man hinh")
    ap.add_argument("--window-width", type=int, default=1280, help="chieu rong cua so khi mo")
    args = ap.parse_args()

    webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if webcam else args.source)
    if not cap.isOpened():
        sys.exit(f"Khong mo duoc nguon video: {args.source}")
    if webcam:
        # Xin webcam độ phân giải HD (webcam không hỗ trợ sẽ tự dùng mức cao nhất có thể)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Nhan dang & theo vet doi tuong chuyen dong"
    fullscreen = args.fullscreen
    if not args.no_display:
        # WINDOW_NORMAL: cửa sổ kéo giãn được bằng chuột, ảnh tự phóng theo
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        cv2.resizeWindow(window, args.window_width, int(args.window_width * h / w))
        if fullscreen:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO(args.model)
    detector = ChangeDetector(args.threshold)
    motion_filter = MotionFilter()
    font = load_font(18)
    writer = None
    paused = False
    frame_idx = 0
    seen_ids = set()
    vis = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            motion_mask = detector.apply(frame)  # Bước 1
            # Bước 2 + 3: nhận dạng và theo vết (persist=True giữ ID giữa các khung)
            result = model.track(frame, persist=True, conf=args.conf, tracker="bytetrack.yaml",
                                 verbose=False)[0]

            objects = []
            if result.boxes.id is not None:
                h, w = frame.shape[:2]
                boxes = [[max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
                         for x1, y1, x2, y2 in result.boxes.xyxy.int().tolist()]
                ids = result.boxes.id.int().tolist()
                # Bước 4: chỉ giữ vật đang chuyển động
                moving = motion_filter.moving_ids(boxes, ids, motion_mask, frame_idx) \
                    if detector.ready else set()
                for box, tid, cls, conf in zip(boxes, ids, result.boxes.cls.int().tolist(),
                                               result.boxes.conf.tolist()):
                    if tid not in moving and not args.all:
                        continue
                    name = model.names[cls]
                    objects.append((box, VI_NAMES.get(name, name), conf, class_color(name), tid))
                objects = drop_nested(objects)
                seen_ids.update(o[4] for o in objects)

            if detector.ready:
                status = f"Đối tượng đang chuyển động: {len(objects)}"
            else:
                status = "Đang học nền - giữ camera đứng yên..."
            vis = draw(frame, objects, font, status)

            if args.output and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (vis.shape[1], vis.shape[0]))
            if writer:
                writer.write(vis)

        if args.no_display:
            continue
        cv2.imshow(window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break  # người dùng bấm nút X đóng cửa sổ
        if key in (ord("p"), ord(" ")):
            paused = not paused
        if key == ord("r"):
            detector.reset()

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Da xu ly {frame_idx} khung hinh, so doi tuong chuyen dong da theo vet: {len(seen_ids)}")


if __name__ == "__main__":
    main()
