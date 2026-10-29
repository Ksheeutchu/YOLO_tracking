"""
Counting engine: YOLOv8 + ByteTrack detection and line-crossing counting,
with no window or display code of its own.

It can be started, stopped and reset, and it hands back annotated frames,
so any front end (the OpenCV command-line window now, a GUI later) can use it.

Typical use:
    engine = CountingEngine(source=0, weights="yolov8m")
    engine.start()
    while True:
        frame = engine.read_frame()      # annotated BGR frame, or None at end
        if frame is None:
            break
        ...show the frame somewhere...
    engine.stop()
    engine.save()
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# Which crossing direction counts as IN (the on-screen label follows this).
# True : left -> right = IN,  right -> left = OUT
# False: right -> left = IN,  left -> right = OUT
LEFT_TO_RIGHT_IS_IN = True


# --------------------------------------------------------------------------
# Helpers (unchanged logic from the original counting_system.py)
# --------------------------------------------------------------------------
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"      # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"     # NVIDIA GPU (your PC desktop)
    return "cpu"


def side_of_line(point, line_p1, line_p2):
    """Return >0 / <0 / 0 depending which side of the line a point is on."""
    (x, y), (x1, y1), (x2, y2) = point, line_p1, line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def counter_path(folder="counters"):
    Path(folder).mkdir(exist_ok=True)
    return Path(folder) / f"counter for {datetime.now():%d.%m.%Y}.json"


def load_counters():
    path = counter_path()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"in": 0, "out": 0, "by_class": {}}


def save_counters(data):
    path = counter_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def resolve_weights(name_or_path: str) -> Path:
    """'yolov8s' -> models/yolov8s.pt next to this file; a .pt path is kept."""
    weights = Path(name_or_path)
    if weights.suffix != ".pt":
        weights = Path(__file__).parent / "models" / f"{name_or_path}.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Model file not found: {weights}")
    return weights


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class CountingEngine:
    def __init__(self, source=0, weights="yolov8m", conf=0.4,
                 classes=None, line=0.5, cam_size=None, mirror=False):
        self.source = source
        self.conf = conf
        self.classes = classes
        self.line = line                  # fraction of frame width, 0.0-1.0
        self.cam_size = cam_size          # (width, height) to request from a webcam
        self.mirror = mirror              # flip webcam image like a mirror (webcams only)

        self.device = pick_device()
        self.model = None
        self.cap = None

        self.data = load_counters()
        self._lock = threading.Lock()     # GUI thread (reset/save) vs worker thread (counting)
        self.last_side = {}               # track_id -> last side (+1 / -1)
        self._first_frame = True          # tells the tracker to start fresh

        self.frame_w = 640
        self.frame_h = 480
        self.line_p1 = (0, 0)
        self.line_p2 = (0, 0)

        self.load_model(weights)

    # ---- model ----------------------------------------------------------
    def load_model(self, weights):
        """Load (or swap) the YOLO model. Safe to call while stopped."""
        self.model = YOLO(str(resolve_weights(weights)))
        self._first_frame = True          # new model -> new tracker state

    # ---- start / stop ---------------------------------------------------
    @property
    def is_running(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        """Open the video source. Counters are kept (use reset_counters)."""
        if self.is_running:
            return
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError(f"Could not open source: {self.source}")

        if self.cam_size and isinstance(self.source, int):   # webcams only
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        line_x = int(self.frame_w * self.line)
        self.line_p1, self.line_p2 = (line_x, 0), (line_x, self.frame_h)

        self.last_side.clear()
        self._first_frame = True

    def stop(self):
        """Release the video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---- counters -------------------------------------------------------
    def reset_counters(self):
        """Zero the in-memory counters (does not touch the saved file)."""
        with self._lock:
            self.data = {"in": 0, "out": 0, "by_class": {}}
            self.last_side.clear()

    def save(self):
        """Write the counters to today's JSON file; returns the file path."""
        with self._lock:
            return save_counters(self.data)

    # ---- main step ------------------------------------------------------
    def read_frame(self):
        """Read one frame, detect, track, count, annotate.

        Returns the annotated BGR frame, or None if the source has ended
        or the engine is not running.
        """
        if not self.is_running:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.mirror and isinstance(self.source, int):
            frame = cv2.flip(frame, 1)    # before detection, so counting matches the screen

        results = self.model.track(
            frame,
            persist=not self._first_frame,   # first frame after start = fresh tracker
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        self._first_frame = False
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, tid, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                side = side_of_line((cx, cy), self.line_p1, self.line_p2)
                side_sign = 1 if side > 0 else -1

                prev = self.last_side.get(tid)
                if prev is not None and prev != side_sign:
                    # side_sign < 0: the object ended up on the RIGHT of the
                    # line, i.e. it moved left -> right.
                    moved_left_to_right = side_sign < 0
                    is_in = moved_left_to_right == LEFT_TO_RIGHT_IS_IN
                    direction = "in" if is_in else "out"
                    with self._lock:
                        self.data[direction] += 1
                        entry = self.data["by_class"].setdefault(
                            self.model.names[cls_id], {"in": 0, "out": 0})
                        entry[direction] += 1
                self.last_side[tid] = side_sign

                self._draw_box(frame, self.model.names[cls_id], conf,
                               x1, y1, x2, y2)

        self._draw_overlay(frame)
        return frame

    # ---- drawing (same look as before) ----------------------------------
    @staticmethod
    def _draw_box(frame, name, conf, x1, y1, x2, y2):
        label = f"{name} {conf:.2f}"
        box_color = (0, 200, 255)      # amber/orange, BGR
        text_color = (0, 0, 0)         # black text
        text_bg = (0, 200, 255)        # filled label background

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      box_color, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        label_y = int(y1) - 12 if int(y1) - 12 > th else int(y1) + th + 12
        cv2.rectangle(frame, (int(x1), label_y - th - 8),
                      (int(x1) + tw + 8, label_y + 4), text_bg, -1)
        cv2.putText(frame, label, (int(x1) + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

    def _draw_line_label(self, frame):
        """Small legend at the bottom of the counting line."""
        if LEFT_TO_RIGHT_IS_IN:
            rows = [("COUNTING LINE", (0, 0, 255)),
                    ("left -> right: IN+", (255, 255, 255)),
                    ("right -> left: OUT-", (255, 255, 255))]
        else:
            rows = [("COUNTING LINE", (0, 0, 255)),
                    ("right -> left: IN+", (255, 255, 255)),
                    ("left -> right: OUT-", (255, 255, 255))]

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        row_h = 30
        box_w = max(cv2.getTextSize(t, font, scale, thick)[0][0]
                    for t, _ in rows) + 20
        box_h = row_h * len(rows) + 10

        line_x = self.line_p1[0]
        left = min(max(line_x - box_w // 2, 10), self.frame_w - box_w - 10)
        bottom = self.frame_h - 10
        top = bottom - box_h

        cv2.rectangle(frame, (left, top), (left + box_w, bottom), (0, 0, 0), -1)
        for i, (text, color) in enumerate(rows):
            cv2.putText(frame, text, (left + 10, top + (i + 1) * row_h - 4),
                        font, scale, color, thick)

    def _draw_overlay(self, frame):
        data = self.data                  # snapshot: a reset may swap self.data
        cv2.line(frame, self.line_p1, self.line_p2, (0, 0, 255), 3)
        self._draw_line_label(frame)

        counter_text = f"IN+: {data['in']}  OUT-: {data['out']}"
        (ctw, cth), _ = cv2.getTextSize(counter_text,
                                         cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.rectangle(frame, (10, 10), (10 + ctw + 20, 10 + cth + 30),
                      (0, 0, 0), -1)
        cv2.putText(frame, counter_text, (20, 10 + cth + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        if data["by_class"]:
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            lines = [f"{name} - IN {c['in']}  OUT {c['out']}"
                     for name, c in sorted(data["by_class"].items())]
            line_h = 34
            panel_w = max(cv2.getTextSize(t, font, scale, thick)[0][0]
                           for t in lines) + 20
            panel_h = line_h * len(lines) + 16
            panel_top = self.frame_h - panel_h - 10

            cv2.rectangle(frame, (10, panel_top),
                          (10 + panel_w, self.frame_h - 10), (0, 0, 0), -1)
            for i, text in enumerate(lines):
                y = panel_top + (i + 1) * line_h - 6
                cv2.putText(frame, text, (20, y), font, scale,
                            (255, 255, 255), thick)
