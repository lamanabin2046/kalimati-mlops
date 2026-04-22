"""
noc_diesel_scraper - AWS Lambda Version
-----------------------------------------
- Scrapes NOC diesel prices (requests + BeautifulSoup)
- Reads existing diesel.csv from S3
- Appends only new data (incremental)
- Writes updated diesel.csv back to S3

S3 path: s3://kalimati-price-prediction/raw/macro/diesel.csv
"""

import io
import re
import time
from datetime import datetime, timedelta

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET = "kalimati-price-prediction"
S3_KEY    = "raw/macro/diesel.csv"

# ---------------------------------------------------------
# NOC Configuration
# ---------------------------------------------------------
BASE_URL  = "https://noc.org.np/retailprice"
PAGE_SIZE = 10
MAX_PAGES = 15
DELAY_SEC = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
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
        print(f"[INFO] Loaded existing data from s3://{bucket}/{key} — {len(df)} rows")
        return df
    except Exception as e:
        print(f"[INFO] No existing file or error: {e}. Starting fresh.")
        return pd.DataFrame()


def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Date Parsing
# ---------------------------------------------------------
def extract_english_date(raw_text: str):
    if not raw_text:
        return None

    cleaned = re.sub(r'^[^\d]+', '', raw_text.strip())
    pattern = r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})'
    matches = re.findall(pattern, cleaned)

    if not matches:
        return None

    for year_str, month_str, day_str in matches:
        year  = int(year_str)
        month = int(month_str)
        day   = int(day_str)

        if year < 2050:
            try:
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


# ---------------------------------------------------------
# Page Scraper
# ---------------------------------------------------------
def scrape_page(offset: int, session: requests.Session) -> list:
    url = f"{BASE_URL}?offset={offset}&max={PAGE_SIZE}"

    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Failed to fetch offset={offset}: {e}")
        return []

    soup  = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    headers_row = table.find("tr")
    if not headers_row:
        return []

    headers = [th.get_text(strip=True).lower() for th in headers_row.find_all(["th", "td"])]

    date_idx   = None
    diesel_idx = None

    for i, h in enumerate(headers):
        if ("date" in h or "effective" in h) and date_idx is None:
            date_idx = i
        if "diesel" in h or "hsd" in h:
            diesel_idx = i

    if date_idx is None or diesel_idx is None:
        print(f"[WARN] Could not identify columns at offset={offset}. Headers: {headers}")
        return []

    records = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) <= max(date_idx, diesel_idx):
            continue

        raw_date  = cells[date_idx].get_text(strip=True)
        raw_price = cells[diesel_idx].get_text(strip=True)

        eng_date = extract_english_date(raw_date)
        if not eng_date:
            continue

        try:
            diesel_price = float(raw_price.replace(",", "").strip())
        except (ValueError, AttributeError):
            continue

        records.append({"date": eng_date, "diesel": diesel_price})

    return records


# ---------------------------------------------------------
# Scrape All Pages
# ---------------------------------------------------------
def scrape_all_pages() -> pd.DataFrame:
    session  = requests.Session()
    all_rows = []

    for page_num in range(MAX_PAGES):
        offset = page_num * PAGE_SIZE
        print(f"[INFO] Scraping page {page_num + 1}/{MAX_PAGES} (offset={offset})")
        rows = scrape_page(offset, session)
        all_rows.extend(rows)
        if page_num < MAX_PAGES - 1:
            time.sleep(DELAY_SEC)

    print(f"[INFO] Total raw rows scraped: {len(all_rows)}")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------
# Preprocess → Daily Forward-Fill
# ---------------------------------------------------------
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No data scraped from NOC.")

    df = raw_df.copy()
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["diesel"] = pd.to_numeric(df["diesel"], errors="coerce")
    df = df.dropna(subset=["date", "diesel"])
    df = (df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True))

    df = df[df["date"] >= "2022-01-01"].reset_index(drop=True)

    yesterday    = pd.Timestamp(datetime.now().date()) - timedelta(days=1)
    daily_index  = pd.date_range(start=df["date"].min(), end=yesterday, freq="D")

    df_daily = (df.set_index("date")
                  .reindex(daily_index)
                  .rename_axis("date")
                  .ffill()
                  .reset_index())

    df_daily["date"] = df_daily["date"].dt.strftime("%Y-%m-%d")

    print(f"[INFO] Daily rows after forward-fill: {len(df_daily)}")
    return df_daily


# ---------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------
def lambda_handler(event, context):
    print("[INFO] NOC Diesel scraper Lambda started")

    # Step 1: Scrape all pages
    raw_df = scrape_all_pages()

    # Step 2: Preprocess to daily series
    new_df = preprocess(raw_df)

    # Step 3: Read existing data from S3
    existing = read_csv_from_s3(S3_BUCKET, S3_KEY)

    # Step 4: Merge old + new
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        merged = (pd.concat([existing, new_df], ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True))
    else:
        merged = new_df

    # Step 5: Write back to S3
    write_csv_to_s3(merged, S3_BUCKET, S3_KEY)

    return {
        "statusCode": 200,
        "body": f"Total rows saved: {len(merged)}"
    }
