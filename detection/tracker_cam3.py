import cv2, json, uuid
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "../footage/CCTV Footage/CAM 3.mp4"
EVENTS_FILE = "../events/events_cam3.jsonl"
ENTRY_LINE_Y = 490
STAFF_ZONE_X = (0, 200)
CAMERA_ID    = "cam3_entrance"

model   = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50, n_init=3)
Path(EVENTS_FILE).parent.mkdir(exist_ok=True)

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
    print("Processing CAM 3 (entrance)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % 3 != 0:
            continue
        results = model(frame, classes=[0], verbose=False, conf=0.4)
        dets = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
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
                track_history[tid] = {
                    "side": "outside" if cy > ENTRY_LINE_Y else "inside",
                    "total_frames": 0, "frames_in_staff_zone": 0,
                    "entry_emitted": False, "exit_emitted": False
                }
            h = track_history[tid]
            h["total_frames"] += 1
            if STAFF_ZONE_X[0] <= cx <= STAFF_ZONE_X[1]:
                h["frames_in_staff_zone"] += 1
            staff = is_staff(tid)
            # ENTRY: crossing from outside(cy>line) to inside(cy<line)
            if h["side"] == "outside" and cy < ENTRY_LINE_Y:
                h["side"] = "inside"
                if not h["entry_emitted"]:
                    emit_event("ENTRY", tid, "main_entrance", staff)
                    h["entry_emitted"] = True
                    h["exit_emitted"]  = False
            # EXIT: crossing from inside to outside
            elif h["side"] == "inside" and cy > ENTRY_LINE_Y:
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
