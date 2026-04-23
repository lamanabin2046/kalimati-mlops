# =========================================================
# Kalimati Daily Supply Volume Scraper (Project Version)
# Folder-compatible version for Tomato_price_prediction
# =========================================================

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "kalimati"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "supply_volume.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
URL = "https://kalimatimarket.gov.np/daily-arrivals"
START_DATE_STR = "01/01/2022"
SLEEP_BETWEEN_DAYS = 1
PAGE_LOAD_SLEEP = 3
TABLE_LOAD_SLEEP = 2
WAIT_TIMEOUT = 25


# ---------------------------------------------------------
# Date Utilities
# ---------------------------------------------------------
def today_nepal_date():
    """Return today's date in Nepal time as naive datetime."""
    now_utc = datetime.utcnow()
    nepal_time = now_utc + timedelta(hours=5, minutes=45)
    return nepal_time.replace(hour=0, minute=0, second=0, microsecond=0)


def format_date_mmddyyyy(dt: datetime) -> str:
    return dt.strftime("%m/%d/%Y")


def format_date_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def parse_date(date_string: str):
    try:
        return datetime.strptime(date_string, "%m/%d/%Y")
    except Exception:
        return None


# ---------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------
def latest_date_in_csv(csv_path: Path):
    """Return the latest scraped date found in CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    latest = None
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if not row:
                continue
            row_date = parse_date(row[0].strip())
            if row_date and (latest is None or row_date > latest):
                latest = row_date

    return latest


def ensure_csv_header(csv_path: Path, headers: list):
    """Create CSV file with header if it does not exist or is empty."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


# ---------------------------------------------------------
# Selenium Setup
# ---------------------------------------------------------
def setup_driver():
    """Initialize headless Chrome driver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(options=options)


# ---------------------------------------------------------
# Date Input Handling
# ---------------------------------------------------------
def set_date(driver, target_date: datetime) -> bool:
    """
    Try to set target date in the date input field.
    Supports both input[type='date'] and input[type='text'].
    """
    target_mmddyyyy = format_date_mmddyyyy(target_date)
    target_iso = format_date_iso(target_date)

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
                        """
                        const el = arguments[0];
                        const value = arguments[1];
                        el.value = value;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        """,
                        el,
                        target_iso,
                    )
                else:
                    driver.execute_script(
                        """
                        const el = arguments[0];
                        el.value = '';
                        """,
                        el,
                    )
                    el.clear()
                    el.send_keys(target_mmddyyyy)
                    driver.execute_script(
                        """
                        const el = arguments[0];
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        """,
                        el,
                    )

                actual_value = el.get_attribute("value")
                print(f"[INFO] Date input set attempt: expected={target_mmddyyyy} / actual={actual_value}")
                return True

            except Exception as input_error:
                print(f"[WARN] Failed on one input field: {input_error}")
                continue

    except Exception as e:
        print(f"[WARN] Error finding date input: {e}")

    return False


# ---------------------------------------------------------
# Scraping Logic
# ---------------------------------------------------------
def scrape_arrival_for_date(driver, wait, target_date: datetime) -> int:
    """
    Scrape Kalimati arrival/supply data for a single date.
    Returns number of rows added.
    """
    target_date_str = format_date_mmddyyyy(target_date)
    print(f"\n[INFO] Scraping arrival data for {target_date_str} ...")

    driver.get(URL)
    time.sleep(PAGE_LOAD_SLEEP)

    if not set_date(driver, target_date):
        print(f"[WARN] Could not set date {target_date_str}")
        return 0

    try:
        btn = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(),'आगमन डाटा जाँच्नुहोस्')]")
            )
        )
        driver.execute_script("arguments[0].click();", btn)
    except Exception as e:
        print(f"[WARN] Could not click button for {target_date_str}: {e}")
        return 0

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
        time.sleep(TABLE_LOAD_SLEEP)
    except Exception as e:
        print(f"[WARN] Table not found for {target_date_str}: {e}")
        return 0

    if "टेबलमा डाटा उपलब्ध भएन" in driver.page_source:
        print(f"[INFO] No arrival data for {target_date_str}")
        return 0

    rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
    if not rows:
        print(f"[WARN] No table rows found for {target_date_str}")
        return 0

    header_cells = rows[0].find_elements(By.TAG_NAME, "th")
    headers = ["Date"] + [th.text.strip() for th in header_cells]
    ensure_csv_header(OUT_FILE, headers)

    added = 0
    with open(OUT_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        for row in rows[1:]:
            cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
            if cols:
                writer.writerow([target_date_str] + cols)
                added += 1

    print(f"[SUCCESS] Added {added} arrival rows for {target_date_str}")
    return added


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    start_date = parse_date(START_DATE_STR)
    end_date = today_nepal_date()
    last_date = latest_date_in_csv(OUT_FILE)

    if start_date is None:
        raise ValueError(f"Invalid START_DATE_STR format: {START_DATE_STR}")

    if last_date:
        if last_date >= end_date:
            print(f"[INFO] Data already up-to-date. Last entry: {format_date_mmddyyyy(last_date)}")
            return

        print(f"[INFO] Resuming from {format_date_mmddyyyy(last_date + timedelta(days=1))}")
        start_date = last_date + timedelta(days=1)
    else:
        print("[INFO] No previous data found. Starting fresh.")

    driver = setup_driver()
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    total_rows = 0

    try:
        current_date = start_date
        while current_date <= end_date:
            total_rows += scrape_arrival_for_date(driver, wait, current_date)
            current_date += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_DAYS)

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    finally:
        driver.quit()

    print(f"\n[DONE] Scraping completed. Total new rows added: {total_rows}")
    print(f"[DONE] Output saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()