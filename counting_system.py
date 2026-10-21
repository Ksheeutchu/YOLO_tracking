"""
Command-line front end for the counting engine (OpenCV window).
Same arguments as before; the detection logic now lives in counting_engine.py.

Usage:
    python counting_system.py --source 0                # webcam
    python counting_system.py --source path/to/video.mp4
    python counting_system.py --source 0 --classes 39 41
"""

import argparse

import cv2

from counting_engine import CountingEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or path to a video file")
    parser.add_argument("--weights", default="yolov8m",
                        help="model name (yolov8n / yolov8s / yolov8m) "
                             "from the models folder, or a path to a .pt file")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="confidence threshold")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="COCO class ids to detect/count; omit for all")
    parser.add_argument("--line", type=float, default=0.5,
                        help="counting line position as a fraction of "
                             "frame width (vertical line), 0.0-1.0")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    engine = CountingEngine(source=source, weights=args.weights,
                            conf=args.conf, classes=args.classes,
                            line=args.line)
    print(f"Using device: {engine.device}")
    engine.start()

    while True:
        frame = engine.read_frame()
        if frame is None:
            break
        cv2.imshow("Detection + Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    engine.stop()
    cv2.destroyAllWindows()

    data = engine.data
    print(f"Final counts -> IN: {data['in']}, OUT: {data['out']}")
    for name, c in sorted(data["by_class"].items()):
        print(f"  {name} - IN {c['in']}, OUT {c['out']}")
    engine.save()


if __name__ == "__main__":
    main()
