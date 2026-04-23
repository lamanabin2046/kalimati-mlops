"""
news_event_ingestion.py
-----------------------
Backfill disruption news from 2022-01-01 to today using Google News RSS
with monthly date windows, then build a district-level event dataset.

Outputs:
1. data/raw/events/news_events_raw.csv
2. data/raw/events/district_events.csv
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_EVENTS_DIR = PROJECT_ROOT / "data" / "raw" / "events"
RAW_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT_PATH = RAW_EVENTS_DIR / "news_events_raw.csv"
DISTRICT_OUTPUT_PATH = RAW_EVENTS_DIR / "district_events.csv"


# =========================================================
# Config
# =========================================================
START_DATE = date(2022, 1, 1)
END_DATE = date.today()

DISTRICTS = ["Kathmandu", "Sarlahi", "Dhading", "Kavre", "Kavrepalanchok"]

EVENT_KEYWORDS = {
    "Landslide": [
        "landslide",
        "landslides",
        "slope failure",
        "hill collapse",
    ],
    "Flood": [
        "flood",
        "floods",
        "flooding",
        "inundation",
        "river overflow",
    ],
    "Road_Block": [
        "road blocked",
        "road obstruction",
        "highway blocked",
        "highway obstructed",
        "traffic disrupted",
        "road disruption",
        "road closed",
        "highway closed",
        "one-way traffic",
        "transport disrupted",
    ],
    "Heavy_Rain_Disruption": [
        "heavy rain",
        "continuous rainfall",
        "rain-induced",
        "monsoon damage",
        "rain disruption",
    ],
}

SOURCE_SITES = [
    "english.onlinekhabar.com",
    "kathmandupost.com",
    "myrepublica.nagariknetwork.com",
    "thehimalayantimes.com",
    "risingnepaldaily.com",
]

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-NP&gl=NP&ceid=NP:en"

REQUEST_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

REQUEST_SLEEP_SECONDS = 1.0


# =========================================================
# Date Window Helpers
# =========================================================
def month_windows(start: date, end: date) -> List[tuple[date, date]]:
    windows: List[tuple[date, date]] = []
    current = date(start.year, start.month, 1)

    while current <= end:
        if current.month == 12:
            nxt = date(current.year + 1, 1, 1)
        else:
            nxt = date(current.year, current.month + 1, 1)

        window_end = min(nxt, end + timedelta(days=1))
        windows.append((current, window_end))
        current = nxt

    return windows


# =========================================================
# Query Building
# =========================================================
def build_query(window_start: date, window_end: date) -> str:
    district_part = " OR ".join(f'"{d}"' for d in DISTRICTS)
    keyword_list = sorted({kw for kws in EVENT_KEYWORDS.values() for kw in kws})
    keyword_part = " OR ".join(f'"{k}"' for k in keyword_list)
    site_part = " OR ".join(f"site:{s}" for s in SOURCE_SITES)

    # Google News RSS supports after:/before: in the query string
    # We use monthly windows to reduce truncation risk.
    query = (
        f"({district_part}) ({keyword_part}) ({site_part}) Nepal "
        f"after:{window_start.isoformat()} before:{window_end.isoformat()}"
    )
    return query


def build_rss_url(window_start: date, window_end: date) -> str:
    query = build_query(window_start, window_end)
    return GOOGLE_NEWS_RSS.format(query=quote_plus(query))


# =========================================================
# RSS Fetch / Parse
# =========================================================
def fetch_rss_items(rss_url: str) -> List[Dict]:
    resp = requests.get(rss_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = []

    for item in root.findall(".//item"):
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""

        items.append(
            {
                "title": title,
                "link": link,
                "pub_date_raw": pub_date,
                "source": source,
            }
        )

    return items


# =========================================================
# Optional Article Text Scraping
# =========================================================
def extract_article_text(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")

        texts = []
        for p in paragraphs:
            t = p.get_text(" ", strip=True)
            if t and len(t) > 30:
                texts.append(t)

        article_text = " ".join(texts)
        article_text = re.sub(r"\s+", " ", article_text).strip()
        return article_text[:20000]
    except Exception:
        return ""


# =========================================================
# Parsing Helpers
# =========================================================
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
    if name.lower() == "kavrepalanchok":
        return "Kavre"
    return name


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
    """
    Simple rule-based severity:
    1 minor
    2 moderate
    3 severe
    4 major
    5 extreme
    """
    if not text:
        return 1

    txt = text.lower()
    score = 1

    moderate_terms = [
        "road blocked",
        "road obstruction",
        "one-way traffic",
        "flooding",
        "landslide",
        "flood",
        "heavy rain",
        "continuous rainfall",
        "damaged road",
    ]
    severe_terms = [
        "major",
        "severe",
        "widespread",
        "highway blocked",
        "highway obstructed",
        "power supply disrupted",
        "transport disrupted",
        "multiple places",
        "several places",
        "district affected",
    ]

    for term in moderate_terms:
        if term in txt:
            score = max(score, 2)

    for term in severe_terms:
        if term in txt:
            score = max(score, 3)

    if "death toll" in txt or "killed" in txt or "fatal" in txt:
        score = max(score, 4)

    if "nationwide" in txt or "extreme" in txt:
        score = max(score, 5)

    return score


# =========================================================
# Transform
# =========================================================
def collect_historical_news(fetch_article_body: bool = False) -> pd.DataFrame:
    records: List[Dict] = []

    windows = month_windows(START_DATE, END_DATE)
    print(f"[INFO] Total monthly windows: {len(windows)}")

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        rss_url = build_rss_url(window_start, window_end)
        print(f"[INFO] Window {idx}/{len(windows)}: {window_start} -> {window_end - timedelta(days=1)}")

        try:
            items = fetch_rss_items(rss_url)
            print(f"[INFO] RSS items fetched: {len(items)}")
        except Exception as e:
            print(f"[WARN] Failed window {window_start} -> {window_end}: {e}")
            continue

        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            source = item.get("source", "")
            pub_date_raw = item.get("pub_date_raw", "")
            pub_dt = parse_pub_date(pub_date_raw)

            article_text = extract_article_text(link) if fetch_article_body and link else ""
            combined_text = f"{title} {article_text}"

            district = detect_district(combined_text)
            event_type = detect_event_type(combined_text)
            severity = infer_severity(combined_text) if event_type else None

            records.append(
                {
                    "date": pub_dt.date().isoformat() if pub_dt is not None else None,
                    "published_at": pub_dt.isoformat() if pub_dt is not None else None,
                    "source": source,
                    "title": title,
                    "url": link,
                    "district": district,
                    "event_type": event_type,
                    "severity": severity,
                    "raw_text": article_text,
                    "window_start": window_start.isoformat(),
                    "window_end": (window_end - timedelta(days=1)).isoformat(),
                }
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "url"]).reset_index(drop=True)

    return df


def build_district_event_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Date", "District", "Event_Type", "Severity"])

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["severity"], errors="coerce")
    df["District"] = df["district"]
    df["Event_Type"] = df["event_type"]

    # make sure these columns exist even if raw_df is malformed
    for col in ["Date", "District", "Event_Type", "Severity"]:
        if col not in df.columns:
            df[col] = None

    df = df.dropna(subset=["Date", "District", "Event_Type", "Severity"])
    df = df[["Date", "District", "Event_Type", "Severity"]].copy()
    df["Severity"] = df["Severity"].astype(int)

    # only keep your four districts
    df = df[df["District"].isin(["Kathmandu", "Sarlahi", "Dhading", "Kavre"])]

    return df.sort_values(["Date", "District"]).reset_index(drop=True)


# =========================================================
# Main
# =========================================================
def main():
    print("[INFO] Starting historical news event ingestion...")

    raw_df = collect_historical_news(fetch_article_body=False)
    raw_df.to_csv(RAW_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Raw news events saved -> {RAW_OUTPUT_PATH}")

    district_df = build_district_event_df(raw_df)
    district_df.to_csv(DISTRICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] District event dataset saved -> {DISTRICT_OUTPUT_PATH}")

    print("\n[INFO] Raw preview:")
    print(raw_df.head())

    print("\n[INFO] District event preview:")
    print(district_df.head())


if __name__ == "__main__":
    main()