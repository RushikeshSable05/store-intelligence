# System Design — Purplle Store Intelligence

## Architecture Overview
CCTV Footage (5 cameras)
|
v
YOLOv8n (Person Detection) + DeepSORT (Tracking)
|
v
events/*.jsonl  <----  Sales CSV (Brigade_Bangalore 10-Apr-2026)
|
v
FastAPI REST API (/metrics /funnel /anomalies /events /departments)
|
v
Live Dashboard (Chart.js) + Docker Compose

## Components

### 1. Detection Pipeline
- Model: YOLOv8n (ultralytics) — detects persons in each frame
- Tracker: DeepSORT — assigns persistent IDs across frames
- Cameras used: CAM2 (makeup zone), CAM1 (skincare zone)
- Staff exclusion: tracks spending >70% of frames in known staff zones are excluded
- Output: append-only JSONL event log per camera

### 2. Sales Data Pipeline
- Source: Brigade_Bangalore sales CSV (10-Apr-2026)
- 101 line items across 24 unique invoices (customer transactions)
- Generates STORE_VISIT and DEPARTMENT_VISIT events with real timestamps
- Provides ground truth for GMV, conversion rate, and department performance

### 3. Event Schema
Each event is a JSON object:
```json
{
  "event_id": "uuid",
  "timestamp": "2026-04-10T19:00:00Z",
  "type": "STORE_VISIT | ENTRY | EXIT | DEPARTMENT_VISIT",
  "session_id": "uuid",
  "zone": "makeup_section | skincare_section | store_floor",
  "camera": "cam2_makeup | sales_data",
  "is_staff": false,
  "source": "sales_csv | cctv"
}
```

### 4. REST API (FastAPI)
| Endpoint | Description |
|---|---|
| GET /health | System health check |
| GET /metrics | KPIs: GMV, transactions, peak hour, occupancy |
| GET /funnel | Conversion funnel with drop-off at each stage |
| GET /anomalies | Detected anomalies: surges, dead hours, discounts |
| GET /events | Raw event log with optional type filter |
| GET /departments | Per-department visits and GMV |

### 5. Dashboard
- Single-page HTML with Chart.js
- Polls API every 30 seconds
- Shows: hourly traffic bar chart, department GMV donut, conversion funnel, anomaly alerts

### 6. Infrastructure
- Docker Compose: api + dashboard services
- Events stored as mounted volume (persist across restarts)
- API reads event files on each request (no DB needed at this scale)

## Camera Mapping to Store Layout
Based on Brigade Road store layout XLSX:
- CAM1: Skincare/Face section (left wall shelves)
- CAM2: Makeup section (center floor + right wall)
- CAM3: Glass storefront — closing-time footage only, outside corridor visible
- CAM4: Stock room — excluded
- CAM5: Billing counter area — staff only during footage window

## Data Flow
1. Detection scripts process each camera video independently
2. Events appended to per-camera JSONL files
3. Sales CSV processed into structured events
4. API merges all event sources at query time
5. Dashboard renders merged data as charts
