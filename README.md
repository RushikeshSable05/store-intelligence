# Purplle Store Intelligence System
Brigade Road, Bangalore - Purplle Tech Challenge 2026 Round 2

## Quick Start
1. Place CCTV footage in footage/CCTV Footage/
2. Place sales CSV in project root
3. python generate_events_from_csv.py
4. docker compose up --build
5. API: http://localhost:8000/docs
6. Dashboard: http://localhost:3000

## API Endpoints
GET /health, /metrics, /funnel, /anomalies, /events, /departments

## Tech Stack
YOLOv8n + DeepSORT, FastAPI, Chart.js, Docker Compose
