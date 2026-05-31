import cv2, json, uuid
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CAMERAS = [
    {
        "name":       "cam2_makeup",
        "path":       "../footage/CCTV Footage/CAM 2.mp4",
        "zone":       "makeup_section",
        "events_out": "../events/events_cam2.jsonl",
        "staff_zone": {"x_max": 280, "y_max": 300},  # left wall area
    },
    {
        "name":       "cam1_skincare",
        "path":       "../footage/CCTV Footage/CAM 1.mp4",
        "zone":       "skincare_section",
        "events_out": "../events/events_cam1.jsonl",
        "staff_zone": {"x_min": 1200, "y_max": 500}, # right counter area
    },
]

STAFF_RATIO = 0.70   # if 70%+ of frames in staff zone → staff
model = YOLO("yolov8n.pt")

def now_ts():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def in_staff_zone(cx, cy, cfg):
    sz = cfg.get("staff_zone", {})
    x_ok = (cx < sz["x_max"]) if "x_max" in sz else (cx > sz.get("x_min", 9999))
    y_ok = cy < sz.get("y_max", 9999)
    return x_ok and y_ok

def emit(f, etype, cam, tid, zone, is_staff, extra={}):
    event = {
        "event_id":   str(uuid.uuid4()),
        "timestamp":  now_ts(),
        "type":       etype,
        "track_id":   f"{cam}_{tid}",
        "zone":       zone,
        "camera":     cam,
        "is_staff":   is_staff,
        "confidence": 0.9,
        "session_id": str(uuid.uuid4()),
        **extra
    }
    f.write(json.dumps(event) + "\n")
    flag = "👔 STAFF" if is_staff else "🛍 CUSTOMER"
    print(f"  [{cam}] {etype} track={tid} {flag} {extra if extra else ''}")

def process_camera(cfg):
    print(f"\n{'='*55}")
    print(f"  {cfg['name']}  →  {cfg['zone']}")
    print(f"{'='*55}")

    Path(cfg["events_out"]).parent.mkdir(exist_ok=True)
    tracker = DeepSort(max_age=60, n_init=3)
    cap = cv2.VideoCapture(cfg["path"])
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    frame_n = 0
    # tid → {frames_total, frames_in_staff_zone, first_frame, last_frame,
    #         entry_emitted, exit_emitted, is_staff_final}
    history = {}

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

            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid  = t.track_id
                ltrb = t.to_ltrb()
                cx   = int((ltrb[0]+ltrb[2])/2)
                cy   = int((ltrb[1]+ltrb[3])/2)

                if tid not in history:
                    history[tid] = {
                        "frames_total": 0,
                        "frames_staff_zone": 0,
                        "first_frame": frame_n,
                        "last_frame": frame_n,
                        "entry_emitted": False,
                        "exit_emitted": False,
                    }

                h = history[tid]
                h["frames_total"] += 1
                h["last_frame"] = frame_n
                if in_staff_zone(cx, cy, cfg):
                    h["frames_staff_zone"] += 1

                # Staff decision: only finalize after 15 frames of data
                ratio = h["frames_staff_zone"] / max(h["frames_total"], 1)
                is_staff = ratio >= STAFF_RATIO and h["frames_total"] >= 15

                # ENTRY: emit once after 3 confirmed frames
                if not h["entry_emitted"] and h["frames_total"] >= 3:
                    emit(f, "ENTRY", cfg["name"], tid, cfg["zone"], is_staff)
                    h["entry_emitted"] = True
                    h["is_staff_final"] = is_staff

            # EXIT: track disappeared for 45+ frames
            active = {t.track_id for t in tracks if t.is_confirmed()}
            for tid, h in history.items():
                if (tid not in active
                        and h["entry_emitted"]
                        and not h["exit_emitted"]
                        and (frame_n - h["last_frame"]) > 45):
                    dwell = round((h["last_frame"] - h["first_frame"]) / fps, 1)
                    is_staff = h.get("is_staff_final", False)
                    emit(f, "EXIT", cfg["name"], tid, cfg["zone"], is_staff,
                         {"dwell_seconds": dwell})
                    h["exit_emitted"] = True

            if frame_n % 300 == 0:
                print(f"  {frame_n} frames...")

    # Emit EXIT for anyone still in frame at video end
    with open(cfg["events_out"], "a") as f:
        for tid, h in history.items():
            if h["entry_emitted"] and not h["exit_emitted"]:
                dwell = round((h["last_frame"] - h["first_frame"]) / fps, 1)
                is_staff = h.get("is_staff_final", False)
                emit(f, "EXIT", cfg["name"], tid, cfg["zone"], is_staff,
                     {"dwell_seconds": dwell, "note": "still_in_frame_at_end"})

    cap.release()
    print(f"  Done — {frame_n} total frames")

if __name__ == "__main__":
    for cfg in CAMERAS:
        process_camera(cfg)

    print("\n\n========== FINAL SUMMARY ==========")
    for cfg in CAMERAS:
        p = Path(cfg["events_out"])
        if not p.exists():
            continue
        events   = [json.loads(l) for l in p.read_text().strip().splitlines() if l]
        cust_in  = [e for e in events if e["type"]=="ENTRY" and not e["is_staff"]]
        cust_out = [e for e in events if e["type"]=="EXIT"  and not e["is_staff"]]
        staff_e  = [e for e in events if e["is_staff"]]
        dwells   = [e["dwell_seconds"] for e in cust_out if "dwell_seconds" in e]
        avg_d    = round(sum(dwells)/len(dwells), 1) if dwells else 0
        print(f"{cfg['name']}:")
        print(f"  customers  → {len(cust_in)} entries, {len(cust_out)} exits")
        print(f"  avg dwell  → {avg_d}s per customer")
        print(f"  staff evts → {len(staff_e)}")
