import cv2
import json
import uuid
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "../footage/CAM 1.mp4"   # ← change this filename
EVENTS_FILE = "../events/events.jsonl"
ENTRY_LINE_Y = 400       # pixel row where the door line is (adjust after viewing)
STAFF_ZONE_X = (0, 200)  # pixel columns that are the counter area (adjust)
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")   # downloads automatically on first run (~6MB)
tracker = DeepSort(max_age=50, n_init=3)

Path(EVENTS_FILE).parent.mkdir(exist_ok=True)

track_history = {}   # track_id → {"side": "in"/"out", "frames_in_staff_zone": int}
session_map = {}     # track_id → session_id

def emit_event(event_type, track_id, zone="main_entrance", is_staff=False, conf=0.0):
    session_id = session_map.get(track_id, str(uuid.uuid4()))
    session_map[track_id] = session_id
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "track_id": str(track_id),
        "zone": zone,
        "is_staff": is_staff,
        "confidence": round(conf, 3),
        "session_id": session_id
    }
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"EVENT: {event_type} | track={track_id} | staff={is_staff}")

def is_staff(track_id):
    h = track_history.get(track_id, {})
    total = h.get("total_frames", 1)
    staff_frames = h.get("frames_in_staff_zone", 0)
    return (staff_frames / total) > 0.8

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video at {VIDEO_PATH}")
        return

    frame_count = 0
    print("Processing video... (press Ctrl+C to stop)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:   # process every 3rd frame for speed
            continue

        height, width = frame.shape[:2]

        # Detect people (class 0)
        results = model(frame, classes=[0], verbose=False, conf=0.4)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            ltrb = track.to_ltrb()
            cx = int((ltrb[0] + ltrb[2]) / 2)
            cy = int((ltrb[1] + ltrb[3]) / 2)

            # Update history
            if tid not in track_history:
                track_history[tid] = {
                    "side": "out" if cy > ENTRY_LINE_Y else "in",
                    "total_frames": 0,
                    "frames_in_staff_zone": 0,
                    "entry_emitted": False,
                    "exit_emitted": False
                }

            h = track_history[tid]
            h["total_frames"] += 1

            # Check staff zone
            if STAFF_ZONE_X[0] <= cx <= STAFF_ZONE_X[1]:
                h["frames_in_staff_zone"] += 1

            staff = is_staff(tid)

            # Entry detection: was outside (cy > line), now inside (cy < line)
            if h["side"] == "out" and cy < ENTRY_LINE_Y:
                h["side"] = "in"
                if not h["entry_emitted"]:
                    emit_event("ENTRY", tid, "main_entrance", staff, 0.9)
                    h["entry_emitted"] = True
                    h["exit_emitted"] = False

            # Exit detection: was inside, now outside
            elif h["side"] == "in" and cy > ENTRY_LINE_Y:
                h["side"] = "out"
                if not h["exit_emitted"]:
                    emit_event("EXIT", tid, "main_entrance", staff, 0.9)
                    h["exit_emitted"] = True
                    h["entry_emitted"] = False  # allow re-entry

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Done. Total frames: {frame_count}")
    print(f"Events saved to {EVENTS_FILE}")

if __name__ == "__main__":
    process_video()