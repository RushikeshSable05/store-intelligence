import csv, json, uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

CSV_PATH    = "/root/store-intelligence/Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
EVENTS_FILE = "/root/store-intelligence/events/events_sales.jsonl"

Path(EVENTS_FILE).parent.mkdir(exist_ok=True)

with open(CSV_PATH, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

invoices = defaultdict(list)
for r in rows:
    invoices[r['invoice_number']].append(r)

print(f"Total line items : {len(rows)}")
print(f"Unique invoices  : {len(invoices)}")

events = []

for inv_no, items in invoices.items():
    first      = items[0]
    order_time = first['order_time']
    dt_str     = f"2026-04-10T{order_time}"
    try:
        dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
    except:
        continue

    session_id  = str(uuid.uuid4())
    customer    = first['customer_name'].strip() or "Guest"
    salesperson = first['salesperson_name'].strip()
    departments = list(set(i['dep_name'] for i in items))
    total_gmv   = sum(float(i['GMV'] or 0) for i in items)
    total_amt   = sum(float(i['total_amount'] or 0) for i in items)
    total_qty   = sum(int(i['qty'] or 0) for i in items)

    events.append({
        "event_id":        str(uuid.uuid4()),
        "timestamp":       dt.isoformat().replace("+00:00","Z"),
        "type":            "STORE_VISIT",
        "session_id":      session_id,
        "invoice_number":  inv_no,
        "customer":        customer,
        "salesperson":     salesperson,
        "departments":     departments,
        "items_purchased": len(items),
        "qty_total":       total_qty,
        "gmv":             total_gmv,
        "total_amount":    total_amt,
        "zone":            "store_floor",
        "camera":          "sales_data",
        "is_staff":        False,
        "source":          "sales_csv"
    })

    for dep in departments:
        dep_items = [i for i in items if i['dep_name'] == dep]
        dep_gmv   = sum(float(i['GMV'] or 0) for i in dep_items)
        events.append({
            "event_id":   str(uuid.uuid4()),
            "timestamp":  dt.isoformat().replace("+00:00","Z"),
            "type":       "DEPARTMENT_VISIT",
            "session_id": session_id,
            "zone":       dep,
            "camera":     "sales_data",
            "is_staff":   False,
            "gmv":        dep_gmv,
            "source":     "sales_csv"
        })

events.sort(key=lambda e: e["timestamp"])

with open(EVENTS_FILE, "w") as f:
    for e in events:
        f.write(json.dumps(e) + "\n")

print(f"Events written   : {len(events)}")

visits    = [e for e in events if e["type"]=="STORE_VISIT"]
total_gmv = sum(e["gmv"] for e in visits)
hours     = defaultdict(int)
for e in visits:
    hours[e["timestamp"][11:13]] += 1

print(f"\n=== DAY SUMMARY ===")
print(f"Total customers  : {len(visits)}")
print(f"Total GMV        : Rs.{total_gmv:,.2f}")
print(f"Avg basket       : Rs.{total_gmv/len(visits):,.2f}")
print(f"\nHourly traffic:")
for h in sorted(hours):
    bar = "x" * hours[h]
    print(f"  {h}:00  {bar}  ({hours[h]})")
SCRIPT
