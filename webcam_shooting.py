import cv2
from pathlib import Path

out = Path("webcam_capture")
out.mkdir(exist_ok=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
n = len(list(out.glob("*.jpg")))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)          # the app mirrors the feed too
    cv2.imshow("SPACE = save, Q = quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(str(out / f"cam_{n:04d}.jpg"), frame)
        n += 1
        print("saved", n)
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()