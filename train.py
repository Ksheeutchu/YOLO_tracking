from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="conveyor-part-detection-2/data.yaml",  # match the paths shown in your data.yaml
    epochs=50,
    imgsz=640,
    batch=8,
    name="part_detector_v2"
)