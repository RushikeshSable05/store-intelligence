import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cameras = {
    "CAM 1": "../footage/CCTV Footage/CAM 1.mp4",
    "CAM 2": "../footage/CCTV Footage/CAM 2.mp4",
    "CAM 3": "../footage/CCTV Footage/CAM 3.mp4",
    "CAM 5": "../footage/CCTV Footage/CAM 5.mp4",
}

for name, path in cameras.items():
    cap = cv2.VideoCapture(path)
    total_people = 0
    frames_checked = 0
    frame_n = 0

    while frames_checked < 20:   # sample 20 frames per camera
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % 60 != 0:    # every 60th frame = every 2 seconds
            continue
        results = model(frame, classes=[0], verbose=False, conf=0.4)
        count = len(results[0].boxes)
        total_people += count
        frames_checked += 1

    cap.release()
    avg = total_people / max(frames_checked, 1)
    print(f"{name}: avg {avg:.1f} people per frame across {frames_checked} samples")

