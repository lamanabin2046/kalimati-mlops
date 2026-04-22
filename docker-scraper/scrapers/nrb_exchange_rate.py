"""
nrb_exchange_rate.py — EC2 + S3 Version
-----------------------------------------
- Downloads forex CSV from NRB using Selenium
- Reads existing exchange_rate_usd_sell.csv from S3
- Merges and writes back to S3

S3 path: s3://kalimati-price-prediction/raw/macro/exchange_rate_usd_sell.csv
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET   = "kalimati-price-prediction"
S3_KEY      = "raw/macro/exchange_rate_usd_sell.csv"

# ---------------------------------------------------------
# Local tmp paths (EC2 writable)
# ---------------------------------------------------------
DOWNLOAD_DIR = Path("/tmp/forex_download")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

URL        = "https://www.nrb.org.np/forex/"
START_DATE = "2022-01-01"

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["date"])
        print(f"[INFO] Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df
    except Exception as e:
        print(f"[INFO] No existing file: {e}. Starting fresh.")
        return pd.DataFrame()


def write_csv_to_s3(df, bucket, key):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Determine Fetch Range
# ---------------------------------------------------------
def get_fetch_range():
    today     = datetime.now().strftime("%Y-%m-%d")
    existing  = read_csv_from_s3(S3_BUCKET, S3_KEY)

    if not existing.empty:
        last_date = existing["date"].max().date()
        next_date = last_date + timedelta(days=1)
        if str(next_date) >= today:
            print("[INFO] Exchange rate data already up to date.")
            return None, None
        from_date = str(next_date)
        print(f"[INFO] Fetching: {from_date} → {today}")
    else:
        from_date = START_DATE
        print(f"[INFO] No existing data. Fetching: {from_date} → {today}")

    return from_date, today


# ---------------------------------------------------------
# Chrome Driver
# ---------------------------------------------------------
def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("prefs", {
        "download.default_directory":   str(DOWNLOAD_DIR.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade":   True,
        "safebrowsing.enabled":         True,
    })
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
        "behavior":     "allow",
        "downloadPath": str(DOWNLOAD_DIR.resolve()),
    })
    return driver


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def clean_folder(folder):
    for f in folder.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass


def wait_for_download(timeout=120):
    end_time = time.time() + timeout
    while time.time() < end_time:
        files   = list(DOWNLOAD_DIR.glob("*"))
        done    = [f for f in files if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
        partial = [f for f in files if f.suffix.lower() in {".crdownload", ".tmp", ".part"}]
        if done and not partial:
            return max(done, key=lambda f: f.stat().st_mtime)
        time.sleep(1)
    raise TimeoutError("Download did not complete.")


def set_date_input(driver, element, date_str):
    driver.execute_script(
        "arguments[0].value = arguments[1];"
        "arguments[0].dispatchEvent(new Event('input',  { bubbles: true }));"
        "arguments[0].dispatchEvent(new Event('change', { bubbles: true }));",
        element, date_str,
    )


# ---------------------------------------------------------
# Download Forex CSV
# ---------------------------------------------------------
def download_forex_csv(from_date, to_date):
    clean_folder(DOWNLOAD_DIR)
    driver = setup_driver()
    wait   = WebDriverWait(driver, 30)

    try:
        print(f"[INFO] Opening: {URL}")
        driver.get(URL)
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(5)

        from_input = wait.until(EC.presence_of_element_located((By.NAME, "dateFrom")))
        set_date_input(driver, from_input, from_date)
        print(f"[INFO] From date set: {from_input.get_attribute('value')}")

        to_input = wait.until(EC.presence_of_element_located((By.NAME, "dateTo")))
        set_date_input(driver, to_input, to_date)
        print(f"[INFO] To date set: {to_input.get_attribute('value')}")

        time.sleep(2)

        # Select CSV export type
        try:
            export_select_el = wait.until(EC.presence_of_element_located((By.NAME, "export_type")))
            export_select    = Select(export_select_el)
            for opt in export_select.options:
                if "csv" in opt.text.lower():
                    export_select.select_by_visible_text(opt.text)
                    print(f"[INFO] Selected export type: {opt.text}")
                    break
        except Exception as e:
            print(f"[WARN] Could not select export type: {e}")

        time.sleep(2)

        export_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[normalize-space()='Export']")
        ))
        driver.execute_script("arguments[0].click();", export_btn)
        print("[INFO] Export clicked. Waiting for download...")

        downloaded = wait_for_download()
        print(f"[OK] Downloaded: {downloaded.name}")
        return downloaded

    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e
    finally:
        driver.quit()


# ---------------------------------------------------------
# Extract USD Sell Rate
# ---------------------------------------------------------
def extract_usd_sell(file_path):
    suffix = file_path.suffix.lower()
    df     = pd.read_csv(file_path) if suffix == ".csv" else pd.read_excel(file_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Find date column
    date_col = None
    for candidate in ["date", "published_date", "published_date_(a.d.)", "day", "period"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        for col in df.columns:
            if "date" in col:
                date_col = col
                break
    if date_col is None:
        raise ValueError(f"Date column not found. Columns: {df.columns.tolist()}")

    # Find USD sell column
    usd_col = None
    for candidate in ["usd_sell", "usd_sell_rate", "u.s._dollar_sell", "us_dollar_sell"]:
        if candidate in df.columns:
            usd_col = candidate
            break
    if usd_col is None:
        for col in df.columns:
            if "usd" in col and "sell" in col:
                usd_col = col
                break
    if usd_col is None:
        sell_cols = [c for c in df.columns if "sell" in c]
        if sell_cols:
            usd_col = sell_cols[0]
    if usd_col is None:
        raise ValueError(f"USD sell column not found. Columns: {df.columns.tolist()}")

    print(f"[INFO] Using date={date_col}, usd_sell={usd_col}")

    out = df[[date_col, usd_col]].copy()
    out.columns = ["date", "usd_sell"]
    out["date"]     = pd.to_datetime(out["date"], errors="coerce")
    out["usd_sell"] = pd.to_numeric(out["usd_sell"], errors="coerce")
    out = (out.dropna(subset=["date", "usd_sell"])
              .sort_values("date")
              .drop_duplicates(subset=["date"])
              .reset_index(drop=True))
    return out


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("[INFO] NRB Exchange Rate Scraper — EC2 + S3 Version")
    print("=" * 60)

    from_date, to_date = get_fetch_range()
    if from_date is None:
        return

    # Download
    downloaded_file = download_forex_csv(from_date, to_date)

    # Extract
    new_df = extract_usd_sell(downloaded_file)
    print(f"[INFO] Extracted {len(new_df)} rows")

    # Merge with existing S3 data
    existing = read_csv_from_s3(S3_BUCKET, S3_KEY)
    if not existing.empty:
        final = (pd.concat([existing, new_df], ignore_index=True)
                   .drop_duplicates(subset=["date"])
                   .sort_values("date")
                   .reset_index(drop=True))
    else:
        final = new_df

    # Save to S3
    write_csv_to_s3(final, S3_BUCKET, S3_KEY)

    # Cleanup
    try:
        downloaded_file.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[DONE] Total rows: {len(final)}")


if __name__ == "__main__":
    main()

