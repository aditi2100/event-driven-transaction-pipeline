# E-commerce Event Data Pipeline (AWS-Style ETL)

A production-aware data engineering project that simulates a serverless ETL workflow for transaction/event data.

## What it demonstrates

- Raw event ingestion (S3-style landing zone)
- Validation + schema enforcement (Lambda-style transform)
- Data quality controls (null checks, domain checks, logic checks)
- Deduplication by event ID
- Clean warehouse load (SQLite local stand-in for PostgreSQL/Redshift)
- Rejected row logging with reason codes
- Pipeline run observability (JSONL logs + run summaries)

---

## Architecture (local demo)

1. Generate incoming raw CSV batch into `raw_data/`
2. Validate and transform:
   - Required field checks
   - Type coercion
   - Allowed value checks (`source_system`)
   - Relationship checks:
     - `items_added_to_cart <= items_viewed`
     - `orders <= items_added_to_cart`
3. Split accepted vs rejected records
4. Write:
   - accepted snapshot → `processed_data/`
   - rejected snapshot → `rejected_data/`
5. Load accepted rows into `ecommerce_events` table
6. Store rejected rows + reason in `rejected_events`
7. Track all batch events in `pipeline_log.jsonl`

---

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

---

## Project files

- `app.py` — full pipeline + UI
- `requirements.txt`
- `README.md`
- `.gitignore`

Generated runtime artifacts:
- `raw_data/`
- `processed_data/`
- `rejected_data/`
- `pipeline.db`
- `pipeline_log.jsonl`

---

## AWS upgrade path (interview discussion)

To deploy this as cloud-native:
- Replace local raw folder with **Amazon S3**
- Trigger transform logic with **AWS Lambda** on new object upload
- Load curated data into **PostgreSQL (RDS)** or **Redshift**
- Add orchestration with **EventBridge + Step Functions** (or Airflow)
- Push metrics/logs to **CloudWatch**

---

## Resume bullets (neutral domain)

- Designed and implemented an event-driven ETL pipeline to ingest, validate, deduplicate, and load high-volume e-commerce records with robust data quality gates.
- Built reject-path logging and operational monitoring to ensure reliable downstream analytics and ML consumption.