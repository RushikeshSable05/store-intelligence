import json, logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from datetime import datetime, timezone
import os, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("store-intelligence")

app = FastAPI(title="Purplle Store Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

EVENTS_SALES = Path(os.getenv("EVENTS_SALES_PATH", "/events/events_sales.jsonl"))
EVENTS_CAM2  = Path(os.getenv("EVENTS_CAM2_PATH",  "/events/events_cam2.jsonl"))
EVENTS_CAM1  = Path(os.getenv("EVENTS_CAM1_PATH",  "/events/events_cam1.jsonl"))

def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def load_all():
    return load(EVENTS_SALES) + load(EVENTS_CAM2) + load(EVENTS_CAM1)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} {response.status_code} {ms}ms")
    return response

@app.get("/health")
def health():
    return {"status": "ok", "store": "Brigade_Bangalore", "date": "2026-04-10"}

@app.get("/metrics")
def metrics():
    sales = load(EVENTS_SALES)
    cam1  = load(EVENTS_CAM1)
    cam2  = load(EVENTS_CAM2)
    visits    = [e for e in sales if e["type"] == "STORE_VISIT"]
    total_gmv = sum(e["gmv"] for e in visits)
    avg_basket = round(total_gmv / len(visits), 2) if visits else 0
    hour_counts = defaultdict(int)
    for e in visits:
        hour_counts[e["timestamp"][11:13]] += 1
    peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else "N/A"
    peak_str = f"{peak_hour}:00-{int(peak_hour)+1:02d}:00" if peak_hour != "N/A" else "N/A"
    dep_events = [e for e in sales if e["type"] == "DEPARTMENT_VISIT"]
    dep_counts = defaultdict(int)
    dep_gmv    = defaultdict(float)
    for e in dep_events:
        dep_counts[e["zone"]] += 1
        dep_gmv[e["zone"]]    += e.get("gmv", 0)
    cam_entries = [e for e in cam1+cam2 if e["type"]=="ENTRY" and not e["is_staff"]]
    cam_exits   = [e for e in cam1+cam2 if e["type"]=="EXIT"  and not e["is_staff"]]
    return {
        "store": "Brigade_Bangalore", "date": "2026-04-10",
        "total_transactions": len(visits),
        "total_line_items": len(sales) - len(visits),
        "total_gmv": round(total_gmv, 2),
        "avg_basket_value": avg_basket,
        "peak_hour": peak_str,
        "hourly_traffic": dict(sorted(hour_counts.items())),
        "department_visits": dict(sorted(dep_counts.items(), key=lambda x: -x[1])),
        "department_gmv": {k: round(v,2) for k,v in sorted(dep_gmv.items(), key=lambda x: -x[1])},
        "live_occupancy": max(0, len(cam_entries) - len(cam_exits)),
        "data_sources": ["sales_csv", "cam1_skincare", "cam2_makeup"]
    }

@app.get("/funnel")
def funnel():
    sales = load(EVENTS_SALES)
    visits = [e for e in sales if e["type"] == "STORE_VISIT"]
    purchased = len(visits)
    estimated_walkins = round(purchased / 0.35) if purchased > 0 else 1
    browsed = round(estimated_walkins * 0.72)
    reached_checkout = round(estimated_walkins * 0.42)
    dep_counts = defaultdict(int)
    for e in sales:
        if e["type"] == "DEPARTMENT_VISIT":
            dep_counts[e["zone"]] += 1
    return {
        "funnel": [
            {"stage": "Estimated Walk-ins",  "count": estimated_walkins, "pct": 100},
            {"stage": "Browsed Products",    "count": browsed,           "pct": round(browsed/estimated_walkins*100,1)},
            {"stage": "Reached Checkout",    "count": reached_checkout,  "pct": round(reached_checkout/estimated_walkins*100,1)},
            {"stage": "Completed Purchase",  "count": purchased,         "pct": round(purchased/estimated_walkins*100,1)},
        ],
        "conversion_rate": f"{round(purchased/estimated_walkins*100,1)}%",
        "department_funnel": dict(sorted(dep_counts.items(), key=lambda x: -x[1])),
        "methodology": "Walk-ins estimated using 35% industry conversion benchmark. Purchases from actual sales CSV."
    }

@app.get("/anomalies")
def anomalies():
    sales  = load(EVENTS_SALES)
    visits = [e for e in sales if e["type"] == "STORE_VISIT"]
    result = []
    hour_counts = defaultdict(int)
    for e in visits:
        hour_counts[e["timestamp"][11:13]] += 1
    if hour_counts:
        avg = len(visits) / len(hour_counts)
        for h, cnt in hour_counts.items():
            if cnt >= avg * 2:
                result.append({"type": "traffic_surge", "severity": "high",
                    "hour": f"{h}:00", "count": cnt,
                    "details": f"{cnt} transactions vs avg {avg:.1f}/hr"})
        all_hours = range(int(min(hour_counts)), int(max(hour_counts))+1)
        for h in all_hours:
            if str(h).zfill(2) not in hour_counts:
                result.append({"type": "dead_hour", "severity": "medium",
                    "hour": f"{h:02d}:00", "details": "Zero transactions"})
    high_disc = [e for e in visits if e["gmv"] > 0 and (e["gmv"]-e["total_amount"])/e["gmv"] > 0.4]
    if len(high_disc) > 3:
        result.append({"type": "high_discount_rate", "severity": "medium",
            "count": len(high_disc), "details": f"{len(high_disc)} transactions had >40% discount"})
    sp_counts = defaultdict(int)
    for e in visits:
        sp_counts[e.get("salesperson","unknown")] += 1
    if sp_counts:
        top_sp  = max(sp_counts, key=sp_counts.get)
        top_pct = round(sp_counts[top_sp]/len(visits)*100, 1) if visits else 0
        if top_pct > 40:
            result.append({"type": "salesperson_concentration", "severity": "low",
                "salesperson": top_sp, "pct": f"{top_pct}%",
                "details": f"{top_sp} handled {top_pct}% of transactions"})
    return {"anomalies": result, "total": len(result)}

@app.get("/events")
def events(limit: int = 50, type: str = None):
    all_ev = sorted(load_all(), key=lambda x: x["timestamp"], reverse=True)
    if type:
        all_ev = [e for e in all_ev if e["type"] == type]
    return {"events": all_ev[:limit], "total": len(all_ev)}

@app.get("/departments")
def departments():
    sales = load(EVENTS_SALES)
    data  = defaultdict(lambda: {"visits": 0, "gmv": 0.0})
    for e in sales:
        if e["type"] == "DEPARTMENT_VISIT":
            data[e["zone"]]["visits"] += 1
            data[e["zone"]]["gmv"]    += e.get("gmv", 0)
    return {"departments": {k: {"visits": v["visits"], "gmv": round(v["gmv"],2)}
            for k,v in sorted(data.items(), key=lambda x: -x[1]["gmv"])}}
