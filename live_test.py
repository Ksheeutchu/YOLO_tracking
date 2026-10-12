from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # pretrained on general objects (COCO dataset)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()  # draws boxes + labels on the frame

    cv2.imshow("YOLOv8 Live Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
# from ultralytics import YOLO
# import cv2
#
# model = YOLO("runs/detect/part_detector_v2/weights/best.pt")
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     results = model(frame, conf=0.3, verbose=False)  # try 0.1–0.2, tune to taste
#     annotated_frame = results[0].plot()
#
#     cv2.imshow("Your Trained Model v2", annotated_frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()