import cv2, json, uuid
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "../footage/CCTV Footage/CAM 3.mp4"
EVENTS_FILE = "../events/events_cam3.jsonl"

# CAM3: side angle view — use VERTICAL line instead
# Glass door panel is roughly at x=530 in the frame
# People enter by moving from RIGHT (outside) to LEFT (inside)
ENTRY_LINE_X = 650       # vertical line at glass door
STORE_REGION_Y = (300, 900)   # ignore detections outside this y band

STAFF_ZONE_Y = (0, 200)  # top area near entrance sign = staff unlikely
CAMERA_ID    = "cam3_entrance"

model   = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=60, n_init=3)
Path(EVENTS_FILE).parent.mkdir(exist_ok=True)

# Clear old events from v1
open(EVENTS_FILE, 'w').close()
print("Cleared old events_cam3.jsonl")

track_history = {}
session_map   = {}

def emit_event(etype, tid, zone="entrance", is_staff=False, conf=0.9):
    sid = session_map.setdefault(tid, str(uuid.uuid4()))
    event = {
        "event_id":   str(uuid.uuid4()),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "type":       etype,
        "track_id":   f"{CAMERA_ID}_{tid}",
        "zone":       zone,
        "camera":     CAMERA_ID,
        "is_staff":   is_staff,
        "confidence": round(conf, 3),
        "session_id": sid
    }
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[CAM3] {etype} | track={tid} | staff={is_staff}")

def is_staff(tid):
    h = track_history.get(tid, {})
    total = max(h.get("total_frames", 1), 1)
    return (h.get("frames_in_staff_zone", 0) / total) > 0.75

def process():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return
    frame_n = 0
    print("Processing CAM 3 (entrance) v2 — vertical line mode...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % 3 != 0:
            continue

        results = model(frame, classes=[0], verbose=False, conf=0.45)
        dets = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cy = (y1 + y2) / 2
            # Only track people in the valid store region vertically
            if STORE_REGION_Y[0] <= cy <= STORE_REGION_Y[1]:
                dets.append(([x1,y1,x2-x1,y2-y1], float(box.conf[0]), "person"))

        tracks = tracker.update_tracks(dets, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid  = t.track_id
            ltrb = t.to_ltrb()
            cx   = int((ltrb[0]+ltrb[2])/2)
            cy   = int((ltrb[1]+ltrb[3])/2)

            if tid not in track_history:
                # Right of line = outside/near door, Left = inside store
                track_history[tid] = {
                    "side": "outside" if cx > ENTRY_LINE_X else "inside",
                    "total_frames": 0,
                    "frames_in_staff_zone": 0,
                    "entry_emitted": False,
                    "exit_emitted": False,
                    "positions": []
                }

            h = track_history[tid]
            h["total_frames"] += 1
            h["positions"].append(cx)

            if cy < STAFF_ZONE_Y[1]:
                h["frames_in_staff_zone"] += 1

            staff = is_staff(tid)

            # ENTRY: was outside (cx > line), now inside (cx < line)
            if h["side"] == "outside" and cx < ENTRY_LINE_X:
                h["side"] = "inside"
                if not h["entry_emitted"]:
                    emit_event("ENTRY", tid, "main_entrance", staff)
                    h["entry_emitted"] = True
                    h["exit_emitted"]  = False

            # EXIT: was inside (cx < line), now outside (cx > line)
            elif h["side"] == "inside" and cx > ENTRY_LINE_X:
                h["side"] = "outside"
                if not h["exit_emitted"]:
                    emit_event("EXIT", tid, "main_entrance", staff)
                    h["exit_emitted"]  = True
                    h["entry_emitted"] = False

        if frame_n % 200 == 0:
            print(f"  {frame_n} frames processed...")

    cap.release()
    print(f"CAM3 done. Total frames: {frame_n}")

if __name__ == "__main__":
    process()
