# =========================================================
# NOC Diesel Price Scraper
# Source : https://noc.org.np/retailprice
# Method : requests + BeautifulSoup  (NO Selenium needed —
#           the page is plain server-rendered HTML)
#
# What it does:
#   1. Scrapes all 15 pages of the NOC retail price table
#   2. Parses the English date from each row
#   3. Extracts the diesel price column
#   4. Filters to 2022-01-01 onwards
#   5. Forward-fills to DAILY frequency
#      (price stays valid until NOC publishes the next change)
#   6. Saves  data/raw/macro/diesel.csv
#
# Output columns:
#   date         YYYY-MM-DD  (daily, no gaps)
#   diesel       float       NPR per litre
# =========================================================

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_MACRO_DIR / "diesel.csv"

# =========================================================
# Configuration
# =========================================================
BASE_URL    = "https://noc.org.np/retailprice"
PAGE_SIZE   = 10
MAX_PAGES   = 15          # 15 pages × 10 rows = 150 price-change records
START_DATE  = "2022-01-01"
DELAY_SEC   = 1.5         # polite delay between page requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# =========================================================
# Date Parsing
# =========================================================
def extract_english_date(raw_text: str) -> str | None:
    """
    NOC uses two date formats (they swapped order at some point):

    Recent rows :  '2082.12.11(2026.03.26)'
                   Nepali date first, English date in parentheses.

    Older rows  :  '2018.04.02 (2074.12.19)'
                   English date first, Nepali date in parentheses.

    Special case:  'प्रेस 2082.05.15 (2025.08.31)'
                   Has a Nepali text prefix — strip it first.

    Rule to identify English date:
        Year < 2050  →  English (AD) year
        Year > 2050  →  Nepali (BS) year

    Returns 'YYYY-MM-DD' string or None if parsing fails.
    """
    if not raw_text:
        return None

    # Remove any leading Nepali/non-ASCII characters before digits
    cleaned = re.sub(r'^[^\d]+', '', raw_text.strip())

    # Find all date-like patterns: NNNN.NN.NN or NNNN-NN-NN
    pattern = r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})'
    matches = re.findall(pattern, cleaned)

    if not matches:
        return None

    for year_str, month_str, day_str in matches:
        year  = int(year_str)
        month = int(month_str)
        day   = int(day_str)

        # English (AD) year is < 2050; Nepali (BS) year is > 2050
        if year < 2050:
            try:
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue   # invalid date — try next match

    return None


# =========================================================
# Page Scraper
# =========================================================
def scrape_page(offset: int, session: requests.Session) -> list[dict]:
    """
    Fetch one page of the NOC retail price table and return
    a list of dicts with keys: date, diesel.
    """
    url    = f"{BASE_URL}?offset={offset}&max={PAGE_SIZE}"
    params = {}

    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch offset={offset}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Find the main data table
    table = soup.find("table")
    if not table:
        print(f"  [WARN] No table found at offset={offset}")
        return []

    # Parse header to find column positions
    headers_row = table.find("tr")
    if not headers_row:
        return []

    headers = [th.get_text(strip=True).lower() for th in headers_row.find_all(["th", "td"])]

    # Identify column indices
    date_idx   = None
    diesel_idx = None

    for i, h in enumerate(headers):
        if "date" in h or "effective" in h:
            if date_idx is None:
                date_idx = i
        if "diesel" in h or "hsd" in h:
            diesel_idx = i

    if date_idx is None or diesel_idx is None:
        print(f"  [WARN] Could not identify columns at offset={offset}. "
              f"Headers found: {headers}")
        return []

    # Parse data rows
    records = []
    rows = table.find_all("tr")[1:]   # skip header row

    for row in rows:
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
            continue   # empty cell or non-numeric — skip

        records.append({"date": eng_date, "diesel": diesel_price})

    return records


# =========================================================
# Scrape All Pages
# =========================================================
def scrape_all_pages() -> pd.DataFrame:
    """Scrape all 15 pages and return a combined DataFrame."""
    session  = requests.Session()
    all_rows = []

    print(f"[INFO] Scraping {MAX_PAGES} pages from NOC...")
    print(f"[INFO] URL pattern: {BASE_URL}?offset=N&max={PAGE_SIZE}")
    print()

    for page_num in range(MAX_PAGES):
        offset = page_num * PAGE_SIZE
        print(f"  Page {page_num + 1:2d}/{MAX_PAGES}  (offset={offset})", end="  →  ")

        rows = scrape_page(offset, session)
        print(f"{len(rows)} rows")
        all_rows.extend(rows)

        if page_num < MAX_PAGES - 1:
            time.sleep(DELAY_SEC)

    print(f"\n[INFO] Total raw rows scraped: {len(all_rows)}")
    return pd.DataFrame(all_rows)


# =========================================================
# Preprocessing
# =========================================================
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the scraped price-change records
    into a DAILY time series ready for joining with the
    main tomato dataset.

    Steps:
      1. Parse dates, remove duplicates, sort ascending
      2. Filter to 2022-01-01 onwards
      3. Forward-fill diesel price to daily frequency
         (each price stays valid until the next NOC update)
      4. Trim to yesterday (avoid incomplete today)
    """
    if raw_df.empty:
        raise ValueError("No data scraped. Check your internet connection or the NOC website.")

    df = raw_df.copy()
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["diesel"] = pd.to_numeric(df["diesel"], errors="coerce")

    # Drop rows with missing date or price
    df = df.dropna(subset=["date", "diesel"])

    # Remove duplicates — keep the last published price per date
    df = (df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True))

    print(f"\n[INFO] After deduplication: {len(df)} unique price-change records")
    print(f"[INFO] Date range in raw data: "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    # Filter to project start date
    df = df[df["date"] >= START_DATE].reset_index(drop=True)
    print(f"[INFO] After filtering to {START_DATE}: {len(df)} records")

    if df.empty:
        raise ValueError(
            f"No records found after {START_DATE}. "
            "NOC data may not go back that far on the current pages."
        )

    # ── Forward-fill to daily frequency ──────────────────────────
    # NOC publishes price changes every 2 weeks on average.
    # Between changes, the previous price applies.
    # We expand the price-change records into a daily series.

    today     = pd.Timestamp(datetime.now().date())
    yesterday = today - timedelta(days=1)

    daily_index = pd.date_range(
        start=df["date"].min(),
        end=yesterday,
        freq="D"
    )

    # Reindex to daily, then forward-fill
    df_daily = (df.set_index("date")
                  .reindex(daily_index)
                  .rename_axis("date")
                  .ffill()                 # forward-fill price gaps
                  .reset_index())

    df_daily["date"] = df_daily["date"].dt.strftime("%Y-%m-%d")

    print(f"[INFO] After daily forward-fill: {len(df_daily)} rows")
    print(f"[INFO] Final date range: "
          f"{df_daily['date'].iloc[0]} → {df_daily['date'].iloc[-1]}")

    return df_daily


# =========================================================
# Merge with Existing File (Incremental Update)
# =========================================================
def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    If diesel.csv already exists, merge old + new data.
    This allows incremental daily updates without re-scraping everything.
    """
    if not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0:
        return new_df

    existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
    existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")

    merged = (pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True))

    print(f"[INFO] Merged: {len(existing)} existing + "
          f"{len(new_df)} new → {len(merged)} total rows")
    return merged


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 60)
    print("[INFO] NOC Diesel Price Scraper")
    print(f"[INFO] Source: {BASE_URL}")
    print(f"[INFO] Output: {OUTPUT_FILE}")
    print("=" * 60)

    # ── Step 1: Scrape all pages ──────────────────────────────────
    raw_df = scrape_all_pages()

    # ── Step 2: Preprocess → daily time series ────────────────────
    print("\n[INFO] Preprocessing...")
    daily_df = preprocess(raw_df)

    # ── Step 3: Merge with existing saved data ────────────────────
    final_df = merge_with_existing(daily_df)

    # ── Step 4: Save ──────────────────────────────────────────────
    final_df.to_csv(OUTPUT_FILE, index=False)

    # ── Step 5: Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Saved → {OUTPUT_FILE}")
    print(f"  Total daily rows : {len(final_df)}")
    print(f"  Date range       : {final_df['date'].iloc[0]}  →  {final_df['date'].iloc[-1]}")
    print(f"  Min diesel price : {final_df['diesel'].min()} NPR/L")
    print(f"  Max diesel price : {final_df['diesel'].max()} NPR/L")
    print(f"  Current price    : {final_df['diesel'].iloc[-1]} NPR/L  "
          f"(as of {final_df['date'].iloc[-1]})")

    # Check for unexpected gaps
    dates     = pd.to_datetime(final_df["date"])
    all_days  = pd.date_range(dates.min(), dates.max(), freq="D")
    missing   = sorted(set(all_days) - set(dates))
    if missing:
        print(f"\n  ⚠ Unexpected gaps ({len(missing)} days):")
        for d in missing[:5]:
            print(f"    {d.date()}")
    else:
        print("  ✅ No gaps — complete daily series")

    print("\n[INFO] Sample — first 5 rows:")
    print(final_df.head(5).to_string(index=False))

    print("\n[INFO] Sample — latest 5 rows:")
    print(final_df.tail(5).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
