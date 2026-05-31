# Engineering Choices — Purplle Store Intelligence

## Model Selection: YOLOv8n over YOLOv8x
YOLOv8n runs at ~30fps on CPU vs ~6fps for YOLOv8x.
For CCTV processing, speed matters more than the 3% mAP difference.
A faster model means lower latency for real-time deployment.

## Tracker: DeepSORT over ByteTrack
DeepSORT handles re-identification better in narrow store aisles
where occlusion is frequent (shelves, other customers blocking view).
ByteTrack is faster but loses track identity during occlusion more often.

## Staff Exclusion: Zone-ratio method
Instead of building a separate staff classifier, we track what fraction
of frames a person spends in known staff zones (counter area, left wall).
If >70% of frames are in staff zones, they are excluded from customer counts.
Trade-off: a customer who spends a long time at the counter may be misclassified.
Acceptable for this use case since staff are consistently stationary at counters.

## Event Store: Append-only JSONL over a database
For 2-3 minutes of footage per camera and one day of sales data,
a flat JSONL file is sufficient. Zero setup, zero dependencies,
easy to inspect and debug. PostgreSQL would be the right choice
for multi-day, multi-store data at scale.

## Sales CSV as primary data source
The CCTV footage provided covers only the last 2-3 minutes of store operation
(timestamp: 20:09-20:10, store closing time). No entry/exit events are visible
in this window — customers were already inside. Rather than generate
misleading entry/exit data from an interior-only view, we use the sales CSV
as ground truth for customer counts and conversion metrics.
This is the honest engineering decision: use the best available data.

## Conversion Rate: Industry benchmark for walk-in estimation
Since no entrance camera captured full-day traffic, we estimate walk-ins
using a 35% conversion benchmark (standard retail industry figure).
This is clearly documented in the /funnel endpoint response.
A production system would use a dedicated entrance camera for exact counts.

## Docker: Simple two-service compose
API + Dashboard as separate services. Detection runs as a preprocessing step
(not a live service) because the footage is pre-recorded, not a live stream.
For a live CCTV deployment, detection would be a third always-on service.

## Re-entry handling
DeepSORT assigns new track IDs when a person re-enters camera view.
With max_age=60 frames, a person who disappears for <2 seconds keeps their ID.
Beyond that, they get a new track ID and are counted as a new detection.
Known limitation: in a 2-minute footage window this has minimal impact.

## Why CAM3 was not used for entry counting
CAM3 is mounted inside the store pointing at the glass storefront.
During the footage window (20:09), the store is at closing time.
People visible through the glass are in the mall corridor, not entering.
Using CAM3 for entry counting produced false positives (17 spurious events
in v1). We rejected this data and documented the reason here.
