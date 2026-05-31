import cv2, json, uuid
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERAS = [
    {
        "name":       "cam2_makeup",
        "path":       "../footage/CCTV Footage/CAM 2.mp4",
        "zone":       "makeup_section",
        "events_out": "../events/events_cam2.jsonl",
        # Staff exclusion: person in black uniform stays near left wall (x < 300)
        # and top area near accessories counter (y < 250)
        "staff_x_max": 300,
        "staff_y_max": 250,
    },
    {
        "name":       "cam1_skincare",
        "path":       "../footage/CCTV Footage/CAM 1.mp4",
        "zone":       "skincare_section",
        "events_out": "../events/events_cam1.jsonl",
        # Staff near right side counter (x > 1150)
        "staff_x_min": 1150,
        "staff_y_max": 400,
    },
]

DWELL_EMIT_SECONDS = 10   # emit DWELL event after person is seen for 10s
model = YOLO("yolov8n.pt")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def emit(f, etype, cam, tid, zone, is_staff, extra={}):
    sid = str(uuid.uuid4())
    event = {
        "event_id":  str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type":      etype,
        "track_id":  f"{cam}_{tid}",
        "zone":      zone,
        "camera":    cam,
        "is_staff":  is_staff,
        "confidence": 0.9,
        "session_id": sid,
        **extra
    }
    f.write(json.dumps(event) + "\n")
    print(f"[{cam}] {etype} | track={tid} | staff={is_staff} | {extra}")

def staff_check(cx, cy, cfg):
    # CAM2: staff stays on left side
    if "staff_x_max" in cfg and cx < cfg["staff_x_max"]:
        return True
    # CAM1: staff stays on right side
    if "staff_x_min" in cfg and cx > cfg["staff_x_min"]:
        return True
    return False

# ── PROCESS EACH CAMERA ───────────────────────────────────────────────────────
def process_camera(cfg):
    print(f"\n{'='*50}")
    print(f"Processing {cfg['name']} → {cfg['zone']}")
    print(f"{'='*50}")

    Path(cfg["events_out"]).parent.mkdir(exist_ok=True)
    tracker = DeepSort(max_age=60, n_init=3)
    cap = cv2.VideoCapture(cfg["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    track_history = {}   # tid → {frames_seen, is_staff, first_ts, last_ts, entry_emitted, exit_emitted}
    frame_n = 0

    with open(cfg["events_out"], "w") as f:
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
                dets.append(([x1,y1,x2-x1,y2-y1], float(box.conf[0]), "person"))

            tracks = tracker.update_tracks(dets, frame=frame)
            current_tids = set()

            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid  = t.track_id
                ltrb = t.to_ltrb()
                cx   = int((ltrb[0]+ltrb[2])/2)
                cy   = int((ltrb[1]+ltrb[3])/2)
                current_tids.add(tid)

                if tid not in track_history:
                    track_history[tid] = {
                        "frames_seen": 0,
                        "is_staff": staff_check(cx, cy, cfg),
                        "first_frame": frame_n,
                        "last_frame":  frame_n,
                        "entry_emitted": False,
                        "exit_emitted":  False,
                        "dwell_emitted": False,
                    }

                h = track_history[tid]
                h["frames_seen"] += 1
                h["last_frame"]   = frame_n

                # Re-check staff every 30 frames (in case initial detection was wrong)
                if frame_n % 30 == 0:
                    if staff_check(cx, cy, cfg):
                        h["is_staff"] = True

                is_staff = h["is_staff"]

                # ENTRY: first time we see this person (after 3 confirmed frames)
                if not h["entry_emitted"] and h["frames_seen"] >= 3:
                    emit(f, "ENTRY", cfg["name"], tid, cfg["zone"], is_staff)
                    h["entry_emitted"] = True

                # DWELL: person has been visible for DWELL_EMIT_SECONDS
                frames_needed = int(fps * DWELL_EMIT_SECONDS)
                if (not h["dwell_emitted"] and
                    not is_staff and
                    h["frames_seen"] >= frames_needed):
                    dwell_so_far = round(h["frames_seen"] / fps, 1)
                    emit(f, "DWELL", cfg["name"], tid, cfg["zone"], is_staff,
                         {"dwell_seconds": dwell_so_far})
                    h["dwell_emitted"] = True

            # EXIT: track was confirmed but disappeared this frame
            confirmed_ids = {t.track_id for t in tracks if t.is_confirmed()}
            for tid, h in track_history.items():
                if (tid not in confirmed_ids and
                    h["entry_emitted"] and
                    not h["exit_emitted"] and
                    (frame_n - h["last_frame"]) > 45):  # gone for 45 frames = ~1.5s

                    dwell = round((h["last_frame"] - h["first_frame"]) / fps, 1)
                    emit(f, "EXIT", cfg["name"], tid, cfg["zone"], h["is_staff"],
                         {"dwell_seconds": dwell})
                    h["exit_emitted"] = True

            if frame_n % 300 == 0:
                print(f"  {frame_n} frames...")

    cap.release()
    print(f"Done {cfg['name']} — {frame_n} frames")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for cfg in CAMERAS:
        process_camera(cfg)

    # Summary
    print("\n\n=== SUMMARY ===")
    for cfg in CAMERAS:
        p = Path(cfg["events_out"])
        if p.exists():
            lines = p.read_text().strip().split("\n")
            events = [json.loads(l) for l in lines if l]
            entries  = [e for e in events if e["type"]=="ENTRY"  and not e["is_staff"]]
            exits    = [e for e in events if e["type"]=="EXIT"   and not e["is_staff"]]
            dwells   = [e for e in events if e["type"]=="DWELL"  and not e["is_staff"]]
            staff_e  = [e for e in events if e["is_staff"]]
            print(f"{cfg['name']}: {len(entries)} customer entries | "
                  f"{len(exits)} exits | {len(dwells)} dwells | "
                  f"{len(staff_e)} staff events")
