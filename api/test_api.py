import os
os.environ["EVENTS_SALES_PATH"] = "/root/store-intelligence/events/events_sales.jsonl"
os.environ["EVENTS_CAM1_PATH"]  = "/root/store-intelligence/events/events_cam1.jsonl"
os.environ["EVENTS_CAM2_PATH"]  = "/root/store-intelligence/events/events_cam2.jsonl"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_metrics_shape():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_gmv" in data
    assert "total_transactions" in data
    assert "peak_hour" in data
    assert "total_gmv" in data  # structure check only

def test_funnel_dropoff():
    r = client.get("/funnel")
    assert r.status_code == 200
    funnel = r.json()["funnel"]
    for i in range(1, len(funnel)):
        assert funnel[i]["count"] <= funnel[i-1]["count"]

def test_anomalies():
    r = client.get("/anomalies")
    assert r.status_code == 200
    assert "anomalies" in r.json()
    assert "total" in r.json()
