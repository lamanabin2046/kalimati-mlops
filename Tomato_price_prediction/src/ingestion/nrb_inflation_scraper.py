"""
NRB Inflation Scraper
---------------------
Downloads inflation data from NRB Real Sector page,
extracts Date + Inflation, and saves cleaned CSV.

Output:
data/raw/macro/inflation.csv
"""

import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOWNLOAD_DIR = PROJECT_ROOT / "downloads"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.nrb.org.np/database-on-nepalese-economy/real-sector/"


# =========================================================
# Selenium Driver
# =========================================================
def make_driver(download_dir: Path) -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {
            "behavior": "allow",
            "downloadPath": str(download_dir.resolve()),
        },
    )
    return driver


# =========================================================
# Helpers
# =========================================================
def wait_page_ready(driver, timeout=30):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def wait_for_download(download_dir: Path, timeout=120) -> Path:
    end_time = time.time() + timeout

    while time.time() < end_time:
        files = list(download_dir.glob("*"))

        completed = [
            f for f in files
            if f.suffix.lower() in [".csv", ".xlsx", ".xls"]
        ]

        partial = [
            f for f in files
            if f.suffix.lower() in [".crdownload", ".tmp", ".part"]
        ]

        if completed and not partial:
            return max(completed, key=lambda f: f.stat().st_mtime)

        time.sleep(1)

    raise TimeoutError("Download did not complete in time.")


# =========================================================
# Page Clicks
# =========================================================
def handle_cookie_popup(driver):
    xpaths = [
        "//button[contains(., 'I Understand')]",
        "//button[contains(., 'Accept')]",
        "//a[contains(., 'Accept')]",
    ]

    for xp in xpaths:
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", btn)
            print("[INFO] Cookie popup handled.")
            time.sleep(1)
            return
        except Exception:
            pass


def click_monthly(driver):
    print("[INFO] Opening Monthly section...")

    xpaths = [
        "//a[contains(., 'Monthly')]",
        "//button[contains(., 'Monthly')]",
    ]

    for xp in xpaths:
        try:
            el = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", el)
            time.sleep(3)
            return
        except Exception:
            pass


def click_price(driver):
    print("[INFO] Opening Price section...")

    xpaths = [
        "//a[contains(., 'Price')]",
        "//button[contains(., 'Price')]",
    ]

    for xp in xpaths:
        try:
            el = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", el)
            time.sleep(3)
            return
        except Exception:
            pass


def click_download(driver):
    print("[INFO] Looking for download/export button...")

    xpaths = [
        "//a[contains(@href,'.xlsx')]",
        "//a[contains(@href,'.xls')]",
        "//button[contains(., 'Download')]",
        "//a[contains(., 'Download')]",
        "//button[contains(., 'Export')]",
        "//a[contains(., 'Export')]",
        "//button[contains(., 'Excel')]",
        "//a[contains(., 'Excel')]",
    ]

    for xp in xpaths:
        try:
            el = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            driver.execute_script("arguments[0].click();", el)
            print("[SUCCESS] Download started.")
            return
        except Exception:
            pass

    raise RuntimeError("Download button not found.")


# =========================================================
# Data Reading
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_downloaded_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
        return normalize_columns(df)

    # Try multiple skiprows values for Excel
    for skip in range(10):
        try:
            df = pd.read_excel(file_path, skiprows=skip)
            df = normalize_columns(df)
            cols = [str(c).lower() for c in df.columns]

            if any("fiscal year" in c for c in cols) or any("mid-month" in c for c in cols):
                print(f"[INFO] Correct header detected after skipping {skip} rows")
                return df
        except Exception:
            continue

    df = pd.read_excel(file_path)
    return normalize_columns(df)


def find_date_column(columns):
    for c in columns:
        lc = str(c).lower()
        if "fiscal year" in lc:
            return c
        if "mid-month" in lc:
            return c
        if "english date" in lc:
            return c
        if "date" in lc:
            return c
        if "month" in lc:
            return c
        if "period" in lc:
            return c

    raise ValueError(f"Date column not found. Columns detected: {list(columns)}")


def find_inflation_column(columns):
    cols = list(columns)

    # Prefer the column right after Overall Index
    for i, c in enumerate(cols):
        lc = str(c).lower()
        if "overall index" in lc:
            if i + 1 < len(cols):
                return cols[i + 1]
            return c

    for c in cols:
        lc = str(c).lower()
        if "overall inflation" in lc:
            return c
        if "inflation" in lc:
            return c
        if "% change" in lc:
            return c

    raise ValueError(f"Inflation column not found. Columns detected: {list(columns)}")


# =========================================================
# Inflation Date Logic
# =========================================================
MONTH_START_MAP = {
    "jul/aug": 7,
    "aug/sep": 8,
    "sep/oct": 9,
    "oct/nov": 10,
    "nov/dec": 11,
    "dec/jan": 12,
    "jan/feb": 1,
    "feb/mar": 2,
    "mar/apr": 3,
    "apr/may": 4,
    "may/jun": 5,
    "jun/jul": 6,
}


def extract_start_year_from_fiscal_label(text: str):
    """
    Example:
    '1974/75 (2031/32)' -> 1974
    '2021/22' -> 2021
    """
    if pd.isna(text):
        return None

    s = str(text).strip()
    m = re.match(r"^(\d{4})/\d{2}", s)
    if m:
        return int(m.group(1))
    return None


def build_inflation_dataframe(raw_df: pd.DataFrame, date_col: str, inflation_col: str) -> pd.DataFrame:
    """
    Convert NRB CPI monthly layout into:
    Date, Inflation

    Layout pattern:
    - fiscal year row
    - month rows below it
    """
    temp = raw_df[[date_col, inflation_col]].copy()
    temp.columns = ["raw_date", "raw_inflation"]

    print("\n[DEBUG] Raw extracted preview:")
    print(temp.head(25))

    records = []
    current_fiscal_start_year = None

    for _, row in temp.iterrows():
        raw_date = row["raw_date"]
        raw_inflation = row["raw_inflation"]

        if pd.isna(raw_date):
            continue

        raw_date_str = str(raw_date).strip().lower()

        # skip header-like row
        if raw_date_str in ["", "nan"]:
            continue

        # detect fiscal year row
        fy_year = extract_start_year_from_fiscal_label(str(raw_date))
        if fy_year is not None:
            current_fiscal_start_year = fy_year

            # if fiscal-year summary row itself has inflation, ignore it
            continue

        # monthly row like Jul/Aug
        if raw_date_str in MONTH_START_MAP and current_fiscal_start_year is not None:
            month_num = MONTH_START_MAP[raw_date_str]

            # Nepal fiscal year starts in July
            # Jul-Dec belong to fiscal start year
            # Jan-Jun belong to next calendar year
            if month_num >= 7:
                year = current_fiscal_start_year
            else:
                year = current_fiscal_start_year + 1

            inflation_value = pd.to_numeric(raw_inflation, errors="coerce")

            if pd.notna(inflation_value):
                records.append(
                    {
                        "Date": pd.Timestamp(year=year, month=month_num, day=1),
                        "Inflation": float(inflation_value),
                    }
                )

    out = pd.DataFrame(records)
    out = out.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    return out


def clean_date_and_inflation(file_path: Path) -> pd.DataFrame:
    df = read_downloaded_file(file_path)

    print("[INFO] Detected columns:")
    print(df.columns.tolist())

    date_col = find_date_column(df.columns)
    inflation_col = find_inflation_column(df.columns)

    print(f"[INFO] Using date column: {date_col}")
    print(f"[INFO] Using inflation column: {inflation_col}")

    out = build_inflation_dataframe(df, date_col, inflation_col)

    print("\n[DEBUG] Final cleaned preview:")
    print(out.head(20))
    print("\n[DEBUG] Final non-null counts:")
    print(out.notna().sum())

    return out


# =========================================================
# Main Pipeline
# =========================================================
def main():
    print("====================================================")
    print("[INFO] NRB Inflation Scraper Started")
    print("====================================================")

    driver = make_driver(DOWNLOAD_DIR)

    try:
        driver.get(BASE_URL)
        wait_page_ready(driver)
        time.sleep(5)

        handle_cookie_popup(driver)
        click_monthly(driver)
        click_price(driver)
        click_download(driver)

        downloaded_file = wait_for_download(DOWNLOAD_DIR)
        print(f"[SUCCESS] Downloaded file: {downloaded_file}")

        cleaned = clean_date_and_inflation(downloaded_file)

        # =====================================================
        # FILTER DATA FROM 2022-01-01 TO NOW
        # =====================================================
        cleaned = cleaned[cleaned["Date"] >= "2022-01-01"]

        # Optional: reset index
        cleaned = cleaned.reset_index(drop=True)

        print("\n[INFO] Filtered dataset (2022 onward):")
        print(cleaned.head())

        # =====================================================
        # SAVE FILE
        # =====================================================
        output_file = RAW_MACRO_DIR / "inflation.csv"
        cleaned.to_csv(output_file, index=False)

        print("\n[SUCCESS] Cleaned inflation file saved to:")
        print(output_file)

        print("\nPreview:")
        print(cleaned.head(10))

        try:
            downloaded_file.unlink(missing_ok=True)
            print(f"\n[INFO] Deleted downloaded source file: {downloaded_file.name}")
        except Exception:
            pass

    finally:
        driver.quit()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()