"""
Real-time object detection + counting system.
Uses a pre-trained YOLOv8 model (COCO classes) with built-in ByteTrack
tracking, and counts objects that cross a virtual line.

This is the M1 milestone: verify the base detection/counting pipeline
works before swapping in your own trained weights and part classes.

Usage:
    python counting_system.py --source 0                # webcam
    python counting_system.py --source path/to/video.mp4
    python counting_system.py --source 0 --classes 39 41  # only count
                                                            # specific COCO
                                                            # class ids (e.g.
                                                            # bottle, cup)

Install:
    pip install ultralytics opencv-python
"""

import argparse
import cv2
import torch
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="0 for webcam, or path to a video file")
    parser.add_argument("--weights", default="yolov8n.pt",
                         help="model weights; swap to your custom .pt later")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="confidence threshold")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                         help="COCO class ids to detect/count; omit for all")
    parser.add_argument("--line", type=float, default=0.5,
                         help="counting line position as a fraction of "
                              "frame width (vertical line), 0.0-1.0")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = pick_device()
    print(f"Using device: {device}")

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    line_x = int(frame_w * args.line)
    line_p1, line_p2 = (line_x, 0), (line_x, frame_h)

    # track_id -> last known side of the line (+1 / -1)
    last_side = {}
    count_in = 0   # crossed left -> right
    count_out = 0  # crossed right -> left

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            classes=args.classes,
            device=device,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, tid, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                side = side_of_line((cx, cy), line_p1, line_p2)
                side_sign = 1 if side > 0 else -1

                prev = last_side.get(tid)
                if prev is not None and prev != side_sign:
                    if side_sign > 0:
                        count_in += 1
                    else:
                        count_out += 1
                last_side[tid] = side_sign

                label = f"#{tid} {model.names[cls_id]} {conf:.2f}"
                box_color = (0, 200, 255)      # amber/orange, BGR
                text_color = (0, 0, 0)         # black text
                text_bg = (0, 200, 255)        # filled label background

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              box_color, 3)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               1.0, 2)
                label_y = int(y1) - 12 if int(y1) - 12 > th else int(y1) + th + 12
                cv2.rectangle(frame, (int(x1), label_y - th - 8),
                              (int(x1) + tw + 8, label_y + 4), text_bg, -1)
                cv2.putText(frame, label, (int(x1) + 4, label_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        cv2.line(frame, line_p1, line_p2, (0, 0, 255), 3)

        counter_text = f"IN: {count_in}  OUT: {count_out}"
        (ctw, cth), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         1.4, 3)
        cv2.rectangle(frame, (10, 10), (10 + ctw + 20, 10 + cth + 30),
                      (0, 0, 0), -1)
        cv2.putText(frame, counter_text, (20, 10 + cth + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        cv2.imshow("Detection + Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final counts -> IN: {count_in}, OUT: {count_out}")


if __name__ == "__main__":
    main()