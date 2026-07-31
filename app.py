import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Paths / Config
# ----------------------------
BASE = Path(".")
RAW_DIR = BASE / "raw_data"
PROCESSED_DIR = BASE / "processed_data"
REJECTED_DIR = BASE / "rejected_data"
DB_PATH = BASE / "pipeline.db"
LOG_PATH = BASE / "pipeline_log.jsonl"

REQUIRED_COLUMNS = [
    "event_id",
    "timestamp",
    "order_id",
    "source_system",
    "items_viewed",
    "items_added_to_cart",
    "orders",
    "order_value",
]
SOURCE_SYSTEMS = ["Web", "Mobile", "POS", "Marketplace"]


# ----------------------------
# Setup helpers
# ----------------------------
def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)


def log_event(level, message, details=None):
    rec = {
        "ts": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        "details": details or {},
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ecommerce_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT,
            order_id TEXT,
            source_system TEXT,
            items_viewed INTEGER,
            items_added_to_cart INTEGER,
            orders INTEGER,
            order_value REAL,
            add_to_cart_rate REAL,
            order_rate REAL,
            avg_order_value_per_order REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rejected_events (
            reject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            reason TEXT,
            raw_record TEXT,
            rejected_at TEXT
        )
        """
    )
    return conn


# ----------------------------
# Step 1: Simulate incoming raw data
# ----------------------------
def simulate_raw_data(n_rows=20000, seed=42):
    np.random.seed(seed)
    now = datetime.utcnow().isoformat()

    source_system = np.random.choice(SOURCE_SYSTEMS, size=n_rows, p=[0.45, 0.30, 0.15, 0.10])
    order_id = [f"ORD_{i:06d}" for i in np.random.randint(1, 400000, size=n_rows)]
    items_viewed = np.random.poisson(lam=20, size=n_rows).clip(min=0)

    base_cart_rate = np.where(source_system == "Web", 0.18,
                     np.where(source_system == "Mobile", 0.15,
                     np.where(source_system == "POS", 0.22, 0.13)))
    cart_noise = np.random.normal(0, 0.02, size=n_rows)
    add_to_cart_rate = np.clip(base_cart_rate + cart_noise, 0.01, 0.60)
    items_added_to_cart = (items_viewed * add_to_cart_rate).astype(int)
    items_added_to_cart = np.minimum(items_added_to_cart, items_viewed)

    order_rate = np.clip(np.random.normal(0.35, 0.08, size=n_rows), 0.01, 0.95)
    orders = (items_added_to_cart * order_rate).astype(int)
    orders = np.minimum(orders, items_added_to_cart)

    avg_item_value = np.where(source_system == "POS", np.random.uniform(15, 45, n_rows),
                     np.where(source_system == "Web", np.random.uniform(12, 40, n_rows),
                     np.where(source_system == "Mobile", np.random.uniform(10, 35, n_rows), np.random.uniform(8, 30, n_rows))))
    order_value = np.round(np.maximum(orders, 1) * avg_item_value, 2)

    event_id = [f"EVT_{i}_{int(time.time())}" for i in range(n_rows)]
    ts = [now] * n_rows

    df = pd.DataFrame({
        "event_id": event_id,
        "timestamp": ts,
        "order_id": order_id,
        "source_system": source_system,
        "items_viewed": items_viewed,
        "items_added_to_cart": items_added_to_cart,
        "orders": orders,
        "order_value": order_value,
    })

    # Inject production-like data issues
    for col in ["order_id", "source_system", "timestamp"]:
        idx = np.random.choice(df.index, size=max(1, n_rows // 200), replace=False)
        df.loc[idx, col] = None

    # Invalid logic
    idx_bad_cart = np.random.choice(df.index, size=max(1, n_rows // 300), replace=False)
    df.loc[idx_bad_cart, "items_added_to_cart"] = df.loc[idx_bad_cart, "items_viewed"] + np.random.randint(1, 5, len(idx_bad_cart))

    idx_bad_orders = np.random.choice(df.index, size=max(1, n_rows // 300), replace=False)
    df.loc[idx_bad_orders, "orders"] = df.loc[idx_bad_orders, "items_added_to_cart"] + np.random.randint(1, 3, len(idx_bad_orders))

    # Duplicates
    dup = df.sample(n=max(1, n_rows // 150), random_state=42)
    df = pd.concat([df, dup], ignore_index=True)

    file_name = f"raw_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    out = RAW_DIR / file_name
    df.to_csv(out, index=False)
    log_event("INFO", "Generated raw file", {"file": str(out), "rows": int(len(df))})
    return out, len(df)


# ----------------------------
# Step 2: Validate + transform
# ----------------------------
def validate_transform(df):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = df[REQUIRED_COLUMNS].copy()

    for c in ["items_viewed", "items_added_to_cart", "orders", "order_value"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    reason = pd.Series("", index=data.index, dtype="object")

    for c in REQUIRED_COLUMNS:
        reason = reason.mask(data[c].isna() & (reason == ""), f"missing_{c}")

    reason = reason.mask(~data["source_system"].isin(SOURCE_SYSTEMS) & (reason == ""), "invalid_source_system")

    reason = reason.mask((data["items_viewed"] < 0) & (reason == ""), "negative_items_viewed")
    reason = reason.mask((data["items_added_to_cart"] < 0) & (reason == ""), "negative_items_added_to_cart")
    reason = reason.mask((data["orders"] < 0) & (reason == ""), "negative_orders")
    reason = reason.mask((data["order_value"] < 0) & (reason == ""), "negative_order_value")
    reason = reason.mask((data["items_added_to_cart"] > data["items_viewed"]) & (reason == ""), "added_to_cart_gt_viewed")
    reason = reason.mask((data["orders"] > data["items_added_to_cart"]) & (reason == ""), "orders_gt_added_to_cart")

    parsed_ts = pd.to_datetime(data["timestamp"], errors="coerce")
    reason = reason.mask(parsed_ts.isna() & (reason == ""), "invalid_timestamp")

    rejected = data[reason != ""].copy()
    rejected["reason"] = reason[reason != ""].values

    accepted = data[reason == ""].copy()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")

    before = len(accepted)
    accepted = accepted.drop_duplicates(subset=["event_id"], keep="first")
    deduped = before - len(accepted)

    accepted["add_to_cart_rate"] = np.where(accepted["items_viewed"] > 0, accepted["items_added_to_cart"] / accepted["items_viewed"], 0.0)
    accepted["order_rate"] = np.where(accepted["items_added_to_cart"] > 0, accepted["orders"] / accepted["items_added_to_cart"], 0.0)
    accepted["avg_order_value_per_order"] = np.where(accepted["orders"] > 0, accepted["order_value"] / accepted["orders"], 0.0)

    stats = {
        "input_rows": int(len(df)),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "deduped_rows": int(deduped),
    }
    return accepted, rejected, stats


# ----------------------------
# Step 3: Load to DB + rejected log
# ----------------------------
def load_to_db(accepted_df, rejected_df):
    conn = get_conn()

    inserted = 0
    for _, r in accepted_df.iterrows():
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO ecommerce_events
            (event_id, timestamp, order_id, source_system, items_viewed, items_added_to_cart, orders, order_value, add_to_cart_rate, order_rate, avg_order_value_per_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(r["event_id"]),
                str(r["timestamp"]),
                str(r["order_id"]),
                str(r["source_system"]),
                int(r["items_viewed"]),
                int(r["items_added_to_cart"]),
                int(r["orders"]),
                float(r["order_value"]),
                float(r["add_to_cart_rate"]),
                float(r["order_rate"]),
                float(r["avg_order_value_per_order"]),
            ),
        )
        if cur.rowcount == 1:
            inserted += 1

    rejected_inserted = 0
    for _, r in rejected_df.iterrows():
        raw_record = json.dumps({
            "event_id": None if pd.isna(r["event_id"]) else str(r["event_id"]),
            "timestamp": None if pd.isna(r["timestamp"]) else str(r["timestamp"]),
            "order_id": None if pd.isna(r["order_id"]) else str(r["order_id"]),
            "source_system": None if pd.isna(r["source_system"]) else str(r["source_system"]),
            "items_viewed": None if pd.isna(r["items_viewed"]) else float(r["items_viewed"]),
            "items_added_to_cart": None if pd.isna(r["items_added_to_cart"]) else float(r["items_added_to_cart"]),
            "orders": None if pd.isna(r["orders"]) else float(r["orders"]),
            "order_value": None if pd.isna(r["order_value"]) else float(r["order_value"]),
        })
        conn.execute(
            """
            INSERT INTO rejected_events (event_id, reason, raw_record, rejected_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                None if pd.isna(r["event_id"]) else str(r["event_id"]),
                str(r["reason"]),
                raw_record,
                datetime.utcnow().isoformat(),
            ),
        )
        rejected_inserted += 1

    conn.commit()
    conn.close()
    return inserted, rejected_inserted


# ----------------------------
# Orchestration
# ----------------------------
def run_pipeline(n_rows=20000):
    ensure_dirs()

    raw_path, generated_rows = simulate_raw_data(n_rows=n_rows, seed=int(time.time()) % 100000)
    raw_df = pd.read_csv(raw_path)
    accepted, rejected, stats = validate_transform(raw_df)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    processed_path = PROCESSED_DIR / f"processed_{stamp}.csv"
    rejected_path = REJECTED_DIR / f"rejected_{stamp}.csv"

    accepted.to_csv(processed_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    inserted, rejected_inserted = load_to_db(accepted, rejected)

    run_summary = {
        "raw_file": str(raw_path),
        "processed_file": str(processed_path),
        "rejected_file": str(rejected_path),
        "generated_rows": generated_rows,
        **stats,
        "inserted_to_ecommerce_events": int(inserted),
        "inserted_to_rejected_events": int(rejected_inserted),
    }
    log_event("INFO", "Pipeline run complete", run_summary)
    return run_summary


# ----------------------------
# Analytics read
# ----------------------------
def fetch_kpis():
    conn = get_conn()
    total = pd.read_sql_query("SELECT COUNT(*) AS n FROM ecommerce_events", conn)["n"].iloc[0]
    rej_total = pd.read_sql_query("SELECT COUNT(*) AS n FROM rejected_events", conn)["n"].iloc[0]

    by_source = pd.read_sql_query(
        """
        SELECT
            source_system,
            SUM(items_viewed) AS items_viewed,
            SUM(items_added_to_cart) AS items_added_to_cart,
            SUM(orders) AS orders,
            SUM(order_value) AS order_value
        FROM ecommerce_events
        GROUP BY source_system
        ORDER BY order_value DESC
        """,
        conn,
    )

    recent_rejects = pd.read_sql_query(
        """
        SELECT event_id, reason, rejected_at
        FROM rejected_events
        ORDER BY reject_id DESC
        LIMIT 50
        """,
        conn,
    )
    conn.close()

    if len(by_source) > 0:
        by_source["add_to_cart_rate"] = np.where(by_source["items_viewed"] > 0, by_source["items_added_to_cart"] / by_source["items_viewed"], 0.0)
        by_source["order_rate"] = np.where(by_source["items_added_to_cart"] > 0, by_source["orders"] / by_source["items_added_to_cart"], 0.0)
        by_source["avg_order_value_per_order"] = np.where(by_source["orders"] > 0, by_source["order_value"] / by_source["orders"], 0.0)
    return int(total), int(rej_total), by_source, recent_rejects


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="E-commerce Event Data Pipeline", layout="wide")
st.title("⚙️ E-commerce Event Data Pipeline (AWS-Style ETL Demo)")
st.caption("Simulates S3-style raw ingestion → validation/transformation (Lambda-style) → warehouse load with reject-path logging.")

ensure_dirs()

with st.sidebar:
    st.header("Run Control")
    n_rows = st.number_input("Rows per simulated batch", min_value=1000, max_value=200000, value=20000, step=1000)
    run_btn = st.button("Run Pipeline Batch")
    st.markdown("---")
    st.write("Local artifacts:")
    st.code("raw_data/\nprocessed_data/\nrejected_data/\npipeline.db\npipeline_log.jsonl")

if run_btn:
    with st.spinner("Running pipeline..."):
        summary = run_pipeline(n_rows=int(n_rows))
    st.success("Pipeline run complete.")
    st.json(summary)

total_rows, total_rejected, by_source, recent_rejects = fetch_kpis()

c1, c2, c3 = st.columns(3)
c1.metric("Rows in ecommerce_events", f"{total_rows:,}")
c2.metric("Rows in rejected_events", f"{total_rejected:,}")

if total_rows > 0 and len(by_source) > 0:
    total_value = by_source["order_value"].sum()
    total_orders = by_source["orders"].sum()
    c3.metric("Average Order Value", f"${(total_value / total_orders):.2f}" if total_orders > 0 else "N/A")
else:
    c3.metric("Average Order Value", "N/A")

st.subheader("Source System Aggregates")
if len(by_source) == 0:
    st.info("No processed rows yet. Click 'Run Pipeline Batch'.")
else:
    st.dataframe(by_source, use_container_width=True)
    st.bar_chart(by_source.set_index("source_system")[["items_viewed", "items_added_to_cart", "orders"]], use_container_width=True)
    st.bar_chart(by_source.set_index("source_system")[["order_value"]], use_container_width=True)

st.subheader("Recent Rejected Records")
if len(recent_rejects) == 0:
    st.info("No rejected rows yet.")
else:
    st.dataframe(recent_rejects, use_container_width=True)

st.subheader("Pipeline Log (latest 20)")
if LOG_PATH.exists():
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    latest = lines[-20:] if lines else []
    st.code("\n".join(latest) if latest else "No logs yet.", language="json")
else:
    st.code("No logs yet.", language="json")