"""
nrb_inflation_scraper.py — EC2 + S3 Version
---------------------------------------------
- Downloads inflation Excel from NRB using Selenium
- Reads existing inflation.csv from S3
- Merges and writes back to S3

S3 path: s3://kalimati-price-prediction/raw/macro/inflation.csv
"""

import io
import re
import time
from pathlib import Path

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
S3_KEY    = "raw/macro/inflation.csv"

# ---------------------------------------------------------
# Local tmp paths
# ---------------------------------------------------------
DOWNLOAD_DIR = Path("/tmp/nrb_inflation")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.nrb.org.np/database-on-nepalese-economy/real-sector/"

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


def write_csv_to_s3(df, bucket, key):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Selenium Driver
# ---------------------------------------------------------
def make_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
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
def wait_page_ready(driver, timeout=30):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def wait_for_download(timeout=120):
    end_time = time.time() + timeout
    while time.time() < end_time:
        files   = list(DOWNLOAD_DIR.glob("*"))
        done    = [f for f in files if f.suffix.lower() in [".csv", ".xlsx", ".xls"]]
        partial = [f for f in files if f.suffix.lower() in [".crdownload", ".tmp", ".part"]]
        if done and not partial:
            return max(done, key=lambda f: f.stat().st_mtime)
        time.sleep(1)
    raise TimeoutError("Download did not complete.")


def handle_cookie_popup(driver):
    for xp in ["//button[contains(., 'I Understand')]", "//button[contains(., 'Accept')]"]:
        try:
            btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1)
            return
        except Exception:
            pass


def click_monthly(driver):
    for xp in ["//a[contains(., 'Monthly')]", "//button[contains(., 'Monthly')]"]:
        try:
            el = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click();", el)
            time.sleep(3)
            return
        except Exception:
            pass


def click_price(driver):
    for xp in ["//a[contains(., 'Price')]", "//button[contains(., 'Price')]"]:
        try:
            el = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click();", el)
            time.sleep(3)
            return
        except Exception:
            pass


def click_download(driver):
    xpaths = [
        "//a[contains(@href,'.xlsx')]", "//a[contains(@href,'.xls')]",
        "//button[contains(., 'Download')]", "//a[contains(., 'Download')]",
        "//button[contains(., 'Export')]",  "//a[contains(., 'Export')]",
        "//button[contains(., 'Excel')]",   "//a[contains(., 'Excel')]",
    ]
    for xp in xpaths:
        try:
            el = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click();", el)
            print("[INFO] Download started.")
            return
        except Exception:
            pass
    raise RuntimeError("Download button not found.")


# ---------------------------------------------------------
# Data Parsing
# ---------------------------------------------------------
MONTH_START_MAP = {
    "jul/aug": 7, "aug/sep": 8,  "sep/oct": 9,  "oct/nov": 10,
    "nov/dec": 11, "dec/jan": 12, "jan/feb": 1,  "feb/mar": 2,
    "mar/apr": 3,  "apr/may": 4,  "may/jun": 5,  "jun/jul": 6,
}


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_downloaded_file(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return normalize_columns(pd.read_csv(file_path))
    for skip in range(10):
        try:
            df   = pd.read_excel(file_path, skiprows=skip)
            df   = normalize_columns(df)
            cols = [str(c).lower() for c in df.columns]
            if any("fiscal year" in c for c in cols) or any("mid-month" in c for c in cols):
                return df
        except Exception:
            continue
    return normalize_columns(pd.read_excel(file_path))


def find_date_column(columns):
    for c in columns:
        lc = str(c).lower()
        if any(k in lc for k in ["fiscal year", "mid-month", "english date", "date", "month", "period"]):
            return c
    raise ValueError(f"Date column not found. Columns: {list(columns)}")


def find_inflation_column(columns):
    cols = list(columns)
    for i, c in enumerate(cols):
        if "overall index" in str(c).lower() and i + 1 < len(cols):
            return cols[i + 1]
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["overall inflation", "inflation", "% change"]):
            return c
    raise ValueError(f"Inflation column not found. Columns: {list(columns)}")


def extract_start_year(text):
    if pd.isna(text):
        return None
    m = re.match(r"^(\d{4})/\d{2}", str(text).strip())
    return int(m.group(1)) if m else None


def build_inflation_df(raw_df, date_col, inflation_col):
    temp = raw_df[[date_col, inflation_col]].copy()
    temp.columns = ["raw_date", "raw_inflation"]

    records = []
    current_fy = None

    for _, row in temp.iterrows():
        raw_date = row["raw_date"]
        if pd.isna(raw_date):
            continue
        raw_str = str(raw_date).strip().lower()
        if raw_str in ["", "nan"]:
            continue

        fy = extract_start_year(str(raw_date))
        if fy is not None:
            current_fy = fy
            continue

        if raw_str in MONTH_START_MAP and current_fy is not None:
            month_num = MONTH_START_MAP[raw_str]
            year      = current_fy if month_num >= 7 else current_fy + 1
            val       = pd.to_numeric(row["raw_inflation"], errors="coerce")
            if pd.notna(val):
                records.append({"Date": pd.Timestamp(year=year, month=month_num, day=1), "Inflation": float(val)})

    out = pd.DataFrame(records)
    return out.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)


def clean_inflation(file_path):
    df           = read_downloaded_file(file_path)
    date_col     = find_date_column(df.columns)
    inflation_col = find_inflation_column(df.columns)
    return build_inflation_df(df, date_col, inflation_col)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("[INFO] NRB Inflation Scraper — EC2 + S3 Version")

    # Clean download dir
    for f in DOWNLOAD_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

    driver = make_driver()

    try:
        driver.get(BASE_URL)
        wait_page_ready(driver)
        time.sleep(5)

        handle_cookie_popup(driver)
        click_monthly(driver)
        click_price(driver)
        click_download(driver)

        downloaded = wait_for_download()
        print(f"[INFO] Downloaded: {downloaded}")

        cleaned = clean_inflation(downloaded)
        cleaned = cleaned[cleaned["Date"] >= "2022-01-01"].reset_index(drop=True)
        print(f"[INFO] Cleaned rows: {len(cleaned)}")

        # Merge with existing S3 data
        existing = read_csv_from_s3(S3_BUCKET, S3_KEY)
        if not existing.empty:
            existing["Date"] = pd.to_datetime(existing["Date"])
            final = (pd.concat([existing, cleaned], ignore_index=True)
                       .drop_duplicates(subset=["Date"])
                       .sort_values("Date")
                       .reset_index(drop=True))
        else:
            final = cleaned

        write_csv_to_s3(final, S3_BUCKET, S3_KEY)

        # Cleanup
        try:
            downloaded.unlink(missing_ok=True)
        except Exception:
            pass

        print(f"[DONE] Total rows: {len(final)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()

