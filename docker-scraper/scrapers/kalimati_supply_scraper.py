"""
kalimati_supply_scraper.py — EC2 + S3 Version
-----------------------------------------------
S3 path: s3://kalimati-price-prediction/raw/kalimati/supply_volume.csv
"""

import io
import time
from datetime import datetime, timedelta

import boto3
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET = "kalimati-price-prediction"
S3_KEY    = "raw/kalimati/supply_volume.csv"

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
URL                = "https://kalimatimarket.gov.np/daily-arrivals"
START_DATE_STR     = "01/01/2022"
SLEEP_BETWEEN_DAYS = 1
PAGE_LOAD_SLEEP    = 3
TABLE_LOAD_SLEEP   = 2
WAIT_TIMEOUT       = 25

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"[INFO] Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df
    except Exception as e:
        print(f"[INFO] No existing file: {e}. Starting fresh.")
        return pd.DataFrame()


def write_csv_to_s3(rows, headers, bucket, key):
    s3       = boto3.client("s3")
    existing = read_csv_from_s3(bucket, key)
    new_df   = pd.DataFrame(rows, columns=headers)

    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    buffer = io.StringIO()
    merged.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(merged)} total rows to s3://{bucket}/{key}")


def get_last_date_from_s3(bucket, key):
    df = read_csv_from_s3(bucket, key)
    if df.empty:
        return None
    try:
        df["Date"] = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y", errors="coerce")
        return df["Date"].max()
    except Exception:
        return None


# ---------------------------------------------------------
# Date Utilities
# ---------------------------------------------------------
def today_nepal_date():
    now_utc    = datetime.utcnow()
    nepal_time = now_utc + timedelta(hours=5, minutes=45)
    return nepal_time.replace(hour=0, minute=0, second=0, microsecond=0)


def format_date_mmddyyyy(dt):
    return dt.strftime("%m/%d/%Y")


def format_date_iso(dt):
    return dt.strftime("%Y-%m-%d")


def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%m/%d/%Y")
    except Exception:
        return None


# ---------------------------------------------------------
# Selenium Setup
# ---------------------------------------------------------
def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(options=options)


# ---------------------------------------------------------
# Date Input
# ---------------------------------------------------------
def set_date(driver, target_date):
    target_mmddyyyy = format_date_mmddyyyy(target_date)
    target_iso      = format_date_iso(target_date)

    try:
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='date'], input[type='text']")
        for el in inputs:
            if not el.is_displayed() or not el.is_enabled():
                continue
            input_type = (el.get_attribute("type") or "").lower()
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", el)
                time.sleep(0.5)
                if input_type == "date":
                    driver.execute_script(
                        "arguments[0].value = arguments[1];"
                        "arguments[0].dispatchEvent(new Event('input',  { bubbles: true }));"
                        "arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
                        el, target_iso,
                    )
                else:
                    el.clear()
                    el.send_keys(target_mmddyyyy)
                    driver.execute_script(
                        "arguments[0].dispatchEvent(new Event('input',  { bubbles: true }));"
                        "arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
                        el,
                    )
                return True
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] Error finding date input: {e}")
    return False


# ---------------------------------------------------------
# Scraping Logic
# ---------------------------------------------------------
def scrape_arrival_for_date(driver, wait, target_date, all_rows, headers):
    target_date_str = format_date_mmddyyyy(target_date)
    print(f"[INFO] Scraping arrival for {target_date_str}...")

    driver.get(URL)
    time.sleep(PAGE_LOAD_SLEEP)

    if not set_date(driver, target_date):
        print(f"[WARN] Could not set date {target_date_str}")
        return 0

    try:
        btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(),'आगमन डाटा जाँच्नुहोस्')]")
        ))
        driver.execute_script("arguments[0].click();", btn)
    except Exception as e:
        print(f"[WARN] Could not click button: {e}")
        return 0

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
        time.sleep(TABLE_LOAD_SLEEP)
    except Exception:
        print(f"[WARN] Table not found for {target_date_str}")
        return 0

    if "टेबलमा डाटा उपलब्ध भएन" in driver.page_source:
        print(f"[INFO] No data for {target_date_str}")
        return 0

    rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
    if not rows:
        return 0

    if not headers:
        header_cells = rows[0].find_elements(By.TAG_NAME, "th")
        headers.extend(["Date"] + [th.text.strip() for th in header_cells])

    added = 0
    for row in rows[1:]:
        cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
        if cols:
            all_rows.append([target_date_str] + cols)
            added += 1

    print(f"[SUCCESS] Added {added} rows for {target_date_str}")
    return added


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    start_date = parse_date(START_DATE_STR)
    end_date   = today_nepal_date()
    last_date  = get_last_date_from_s3(S3_BUCKET, S3_KEY)

    if last_date:
        if last_date >= end_date:
            print("[INFO] Data already up to date.")
            return
        start_date = last_date + timedelta(days=1)
        print(f"[INFO] Resuming from {format_date_mmddyyyy(start_date)}")
    else:
        print("[INFO] No previous data. Starting fresh.")

    driver     = setup_driver()
    wait       = WebDriverWait(driver, WAIT_TIMEOUT)
    all_rows   = []
    headers    = []
    total_rows = 0

    try:
        current_date = start_date
        while current_date <= end_date:
            total_rows  += scrape_arrival_for_date(driver, wait, current_date, all_rows, headers)
            current_date += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_DAYS)
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        driver.quit()

    if all_rows and headers:
        write_csv_to_s3(all_rows, headers, S3_BUCKET, S3_KEY)

    print(f"[DONE] Total new rows: {total_rows}")


if __name__ == "__main__":
    main()

