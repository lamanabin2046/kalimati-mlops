"""
news_event_ingestion - AWS Lambda Version
------------------------------------------
- Fetches only the CURRENT MONTH's news window (incremental)
- Reads existing CSVs from S3
- Appends new records and writes back to S3

S3 paths:
  s3://kalimati-price-prediction/raw/events/news_events_raw.csv
  s3://kalimati-price-prediction/raw/events/district_events.csv
"""

from __future__ import annotations

import io
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET        = "kalimati-price-prediction"
S3_RAW_KEY       = "raw/events/news_events_raw.csv"
S3_DISTRICT_KEY  = "raw/events/district_events.csv"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DISTRICTS = ["Kathmandu", "Sarlahi", "Dhading", "Kavre", "Kavrepalanchok"]

EVENT_KEYWORDS = {
    "Landslide": ["landslide", "landslides", "slope failure", "hill collapse"],
    "Flood": ["flood", "floods", "flooding", "inundation", "river overflow"],
    "Road_Block": [
        "road blocked", "road obstruction", "highway blocked",
        "highway obstructed", "traffic disrupted", "road disruption",
        "road closed", "highway closed", "one-way traffic", "transport disrupted",
    ],
    "Heavy_Rain_Disruption": [
        "heavy rain", "continuous rainfall", "rain-induced",
        "monsoon damage", "rain disruption",
    ],
}

SOURCE_SITES = [
    "english.onlinekhabar.com",
    "kathmandupost.com",
    "myrepublica.nagariknetwork.com",
    "thehimalayantimes.com",
    "risingnepaldaily.com",
]

GOOGLE_NEWS_RSS      = "https://news.google.com/rss/search?q={query}&hl=en-NP&gl=NP&ceid=NP:en"
REQUEST_TIMEOUT      = 25
REQUEST_SLEEP_SECONDS = 1.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"[INFO] Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df
    except Exception as e:
        print(f"[INFO] No existing file at {key}: {e}. Starting fresh.")
        return pd.DataFrame()


def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Query / RSS Helpers
# ---------------------------------------------------------
def build_rss_url(window_start: date, window_end: date) -> str:
    district_part = " OR ".join(f'"{d}"' for d in DISTRICTS)
    keyword_list  = sorted({kw for kws in EVENT_KEYWORDS.values() for kw in kws})
    keyword_part  = " OR ".join(f'"{k}"' for k in keyword_list)
    site_part     = " OR ".join(f"site:{s}" for s in SOURCE_SITES)
    query = (
        f"({district_part}) ({keyword_part}) ({site_part}) Nepal "
        f"after:{window_start.isoformat()} before:{window_end.isoformat()}"
    )
    return GOOGLE_NEWS_RSS.format(query=quote_plus(query))


def fetch_rss_items(rss_url: str) -> List[Dict]:
    resp = requests.get(rss_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    root  = ET.fromstring(resp.content)
    items = []
    for item in root.findall(".//item"):
        title     = item.findtext("title", default="").strip()
        link      = item.findtext("link", default="").strip()
        pub_date  = item.findtext("pubDate", default="").strip()
        source_el = item.find("source")
        source    = source_el.text.strip() if source_el is not None and source_el.text else ""
        items.append({"title": title, "link": link, "pub_date_raw": pub_date, "source": source})
    return items


# ---------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------
def parse_pub_date(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if getattr(dt, "tzinfo", None):
            dt = dt.astimezone().replace(tzinfo=None)
        return pd.Timestamp(dt)
    except Exception:
        return None


def normalize_district(name: str) -> str:
    return "Kavre" if name.lower() == "kavrepalanchok" else name


def detect_district(text: str) -> Optional[str]:
    if not text:
        return None
    txt = text.lower()
    for district in DISTRICTS:
        if district.lower() in txt:
            return normalize_district(district)
    return None


def detect_event_type(text: str) -> Optional[str]:
    if not text:
        return None
    txt = text.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in txt:
                return event_type
    return None


def infer_severity(text: str) -> int:
    if not text:
        return 1
    txt   = text.lower()
    score = 1
    moderate_terms = ["road blocked", "one-way traffic", "flooding", "landslide", "flood", "heavy rain"]
    severe_terms   = ["major", "severe", "widespread", "highway blocked", "transport disrupted"]
    for t in moderate_terms:
        if t in txt:
            score = max(score, 2)
    for t in severe_terms:
        if t in txt:
            score = max(score, 3)
    if "killed" in txt or "fatal" in txt:
        score = max(score, 4)
    if "nationwide" in txt or "extreme" in txt:
        score = max(score, 5)
    return score


# ---------------------------------------------------------
# Fetch News for a Date Window
# ---------------------------------------------------------
def fetch_window(window_start: date, window_end: date) -> List[Dict]:
    rss_url = build_rss_url(window_start, window_end)
    print(f"[INFO] Fetching window: {window_start} → {window_end}")

    try:
        items = fetch_rss_items(rss_url)
        print(f"[INFO] RSS items fetched: {len(items)}")
    except Exception as e:
        print(f"[WARN] Failed to fetch RSS: {e}")
        return []

    records = []
    for item in items:
        title        = item.get("title", "")
        link         = item.get("link", "")
        source       = item.get("source", "")
        pub_date_raw = item.get("pub_date_raw", "")
        pub_dt       = parse_pub_date(pub_date_raw)
        combined     = title

        district   = detect_district(combined)
        event_type = detect_event_type(combined)
        severity   = infer_severity(combined) if event_type else None

        records.append({
            "date":         pub_dt.date().isoformat() if pub_dt is not None else None,
            "published_at": pub_dt.isoformat() if pub_dt is not None else None,
            "source":       source,
            "title":        title,
            "url":          link,
            "district":     district,
            "event_type":   event_type,
            "severity":     severity,
            "window_start": window_start.isoformat(),
            "window_end":   (window_end - timedelta(days=1)).isoformat(),
        })

    return records


def build_district_event_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Date", "District", "Event_Type", "Severity"])

    df = raw_df.copy()
    df["Date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["Severity"]   = pd.to_numeric(df["severity"], errors="coerce")
    df["District"]   = df["district"]
    df["Event_Type"] = df["event_type"]
    df = df.dropna(subset=["Date", "District", "Event_Type", "Severity"])
    df = df[["Date", "District", "Event_Type", "Severity"]].copy()
    df["Severity"] = df["Severity"].astype(int)
    df = df[df["District"].isin(["Kathmandu", "Sarlahi", "Dhading", "Kavre"])]
    return df.sort_values(["Date", "District"]).reset_index(drop=True)


# ---------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------
def lambda_handler(event, context):
    print("[INFO] News event ingestion Lambda started")

    today = date.today()

    # Fetch current month window only (incremental daily run)
    window_start = date(today.year, today.month, 1)
    window_end   = today + timedelta(days=1)

    # Step 1: Fetch new records
    new_records = fetch_window(window_start, window_end)
    time.sleep(REQUEST_SLEEP_SECONDS)

    if not new_records:
        print("[INFO] No new records found.")
        return {"statusCode": 200, "body": "No new records"}

    new_raw_df = pd.DataFrame(new_records)
    new_raw_df = new_raw_df.drop_duplicates(subset=["title", "url"]).reset_index(drop=True)

    # Step 2: Read existing raw CSV from S3 and merge
    existing_raw = read_csv_from_s3(S3_BUCKET, S3_RAW_KEY)
    if not existing_raw.empty:
        merged_raw = (pd.concat([existing_raw, new_raw_df], ignore_index=True)
                        .drop_duplicates(subset=["title", "url"])
                        .reset_index(drop=True))
    else:
        merged_raw = new_raw_df

    # Step 3: Build district events from full merged raw
    district_df = build_district_event_df(merged_raw)

    # Step 4: Read existing district CSV and merge
    existing_district = read_csv_from_s3(S3_BUCKET, S3_DISTRICT_KEY)
    if not existing_district.empty:
        merged_district = (pd.concat([existing_district, district_df], ignore_index=True)
                             .drop_duplicates()
                             .sort_values(["Date", "District"])
                             .reset_index(drop=True))
    else:
        merged_district = district_df

    # Step 5: Write both back to S3
    write_csv_to_s3(merged_raw, S3_BUCKET, S3_RAW_KEY)
    write_csv_to_s3(merged_district, S3_BUCKET, S3_DISTRICT_KEY)

    return {
        "statusCode": 200,
        "body": f"New records: {len(new_raw_df)}. Total raw: {len(merged_raw)}. Total district events: {len(merged_district)}"
    }
