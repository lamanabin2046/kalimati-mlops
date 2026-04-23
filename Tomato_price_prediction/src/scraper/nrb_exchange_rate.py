# =========================================================
# NRB Exchange Rate Downloader — Fixed Version
# Fixes:
#   1. Targets dateFrom / dateTo inputs by NAME (not placeholder)
#   2. Targets export_type select by NAME (not position)
#   3. Uses dedicated clean download folder (never picks old files)
#   4. Uses YYYY-MM-DD format for type='date' inputs
#   5. Incremental update — only fetches missing dates
#   6. Broad USD column detection (handles any NRB column naming)
# =========================================================

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC


# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT       = Path(__file__).resolve().parents[2]
RAW_MACRO_DIR      = PROJECT_ROOT / "data" / "raw" / "macro"
PROCESSED_DIR      = PROJECT_ROOT / "data" / "processed"
FOREX_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "tmp_forex"
DEBUG_DIR          = PROJECT_ROOT / "data" / "debug"

RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FOREX_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "exchange_rate_usd_sell.csv"
URL         = "https://www.nrb.org.np/forex/"
START_DATE  = "2022-01-01"


# =========================================================
# Chrome Driver
# =========================================================
def setup_driver(download_dir: Path) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("prefs", {
        "download.default_directory":   str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade":   True,
        "safebrowsing.enabled":         True,
    })
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
        "behavior":     "allow",
        "downloadPath": str(download_dir.resolve()),
    })
    return driver


# =========================================================
# Helpers
# =========================================================
def clean_folder(folder: Path) -> None:
    """Remove all files so we never accidentally pick up old downloads."""
    for f in folder.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass


def wait_for_download(download_dir: Path, timeout: int = 120) -> Path:
    """Wait until a complete CSV/Excel file appears."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        files   = list(download_dir.glob("*"))
        done    = [f for f in files if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
        partial = [f for f in files if f.suffix.lower() in {".crdownload", ".tmp", ".part"}]
        if done and not partial:
            return max(done, key=lambda f: f.stat().st_mtime)
        time.sleep(1)
    raise TimeoutError(
        "Download did not complete.\n"
        f"Check: {DEBUG_DIR / 'debug_before_export.html'}"
    )


def set_date_input(driver, element, date_str: str) -> None:
    """Set a type='date' input via JavaScript and fire events. date_str = YYYY-MM-DD."""
    driver.execute_script(
        """
        arguments[0].value = arguments[1];
        arguments[0].dispatchEvent(new Event('input',  { bubbles: true }));
        arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
        """,
        element,
        date_str,
    )


def save_debug(driver, label: str) -> None:
    """Save screenshot + page source for debugging."""
    try:
        driver.save_screenshot(str(DEBUG_DIR / f"{label}.png"))
        (DEBUG_DIR / f"{label}.html").write_text(driver.page_source, encoding="utf-8")
        print(f"[DEBUG] Saved → {DEBUG_DIR / label}.png / .html")
    except Exception as e:
        print(f"[WARN] Could not save debug: {e}")


# =========================================================
# Determine Fetch Range (Incremental)
# =========================================================
def get_fetch_range() -> tuple:
    today = datetime.now().strftime("%Y-%m-%d")

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        existing  = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        last_date = existing["date"].max().date()
        next_date = last_date + timedelta(days=1)

        if str(next_date) >= today:
            print("[INFO] Exchange rate data is already up to date.")
            return None, None

        from_date = str(next_date)
        print(f"[INFO] Existing data found. Fetching: {from_date} → {today}")
    else:
        from_date = START_DATE
        print(f"[INFO] No existing data. Fetching: {from_date} → {today}")

    return from_date, today


# =========================================================
# Download
# =========================================================
def download_forex_csv(from_date: str, to_date: str) -> Path:
    # Always start with a clean folder
    clean_folder(FOREX_DOWNLOAD_DIR)

    driver = setup_driver(FOREX_DOWNLOAD_DIR)
    wait   = WebDriverWait(driver, 30)

    try:
        print(f"[INFO] Opening: {URL}")
        driver.get(URL)

        # Wait for full page load
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(5)
        print(f"[INFO] Page title: {driver.title}")

        # ── Debug: print all page elements ────────────────────────
        print("[DEBUG] ── Inputs ──────────────────────────────────────")
        for i, el in enumerate(driver.find_elements(By.TAG_NAME, "input")):
            print(f"  [{i}] type='{el.get_attribute('type')}'  "
                  f"name='{el.get_attribute('name')}'  "
                  f"id='{el.get_attribute('id')}'  "
                  f"placeholder='{el.get_attribute('placeholder')}'  "
                  f"visible={el.is_displayed()}")

        print("[DEBUG] ── Selects ─────────────────────────────────────")
        for i, el in enumerate(driver.find_elements(By.TAG_NAME, "select")):
            opts = [o.text.strip() for o in Select(el).options]
            print(f"  [{i}] name='{el.get_attribute('name')}'  "
                  f"id='{el.get_attribute('id')}'  options={opts}")

        print("[DEBUG] ── Buttons ─────────────────────────────────────")
        for i, el in enumerate(driver.find_elements(By.TAG_NAME, "button")):
            print(f"  [{i}] text='{el.text.strip()}'  id='{el.get_attribute('id')}'")

        # ── FIX 1: Set From date by NAME ─────────────────────────
        from_input = wait.until(
            EC.presence_of_element_located((By.NAME, "dateFrom"))
        )
        set_date_input(driver, from_input, from_date)
        actual_from = from_input.get_attribute("value")
        print(f"[INFO] From date set: {actual_from}")

        if not actual_from:
            raise RuntimeError(
                "From date was not accepted by the page. "
                "The input may need a different format or interaction."
            )

        # ── FIX 2: Set To date by NAME ───────────────────────────
        to_input = wait.until(
            EC.presence_of_element_located((By.NAME, "dateTo"))
        )
        set_date_input(driver, to_input, to_date)
        actual_to = to_input.get_attribute("value")
        print(f"[INFO] To date set  : {actual_to}")

        time.sleep(2)

        # ── FIX 3: Select export_type by NAME ────────────────────
        export_select_el = wait.until(
            EC.presence_of_element_located((By.NAME, "export_type"))
        )
        export_select = Select(export_select_el)
        options = [o.text.strip() for o in export_select.options]
        print(f"[INFO] Export type options: {options}")

        # Try common CSV label variations
        csv_selected = False
        for label in ["CSV", "csv", ".csv", "Export CSV", "Download CSV",
                       "Comma Separated", "Comma-Separated"]:
            try:
                export_select.select_by_visible_text(label)
                csv_selected = True
                print(f"[OK] Selected export format: {label}")
                break
            except Exception:
                continue

        if not csv_selected:
            # Fall back: pick first non-empty, non-placeholder option
            for opt in export_select.options:
                text = opt.text.strip()
                if text and text.lower() not in {"select", "select format",
                                                  "select date range", "--", ""}:
                    export_select.select_by_visible_text(text)
                    print(f"[OK] Selected format (fallback): {text}")
                    csv_selected = True
                    break

        if not csv_selected:
            print("[WARN] Could not select export format — attempting download anyway.")

        time.sleep(2)

        # Save debug snapshot before clicking export
        save_debug(driver, "debug_before_export")

        # ── Click Export button ───────────────────────────────────
        export_btn = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[normalize-space()='Export']")
            )
        )
        driver.execute_script("arguments[0].click();", export_btn)
        print("[OK] Export button clicked. Waiting for download...")

        # ── Wait for file ─────────────────────────────────────────
        downloaded = wait_for_download(FOREX_DOWNLOAD_DIR, timeout=120)
        print(f"[OK] Downloaded: {downloaded.name}  "
              f"({downloaded.stat().st_size:,} bytes)")
        return downloaded

    except Exception as e:
        save_debug(driver, "debug_error")
        raise RuntimeError(
            f"Download failed: {e}\n"
            f"Open {DEBUG_DIR / 'debug_error.html'} in your browser to inspect."
        ) from e

    finally:
        driver.quit()


# =========================================================
# Extract USD Sell Rate
# =========================================================
def extract_usd_sell(file_path: Path) -> pd.DataFrame:
    # Read file
    suffix = file_path.suffix.lower()
    df = pd.read_csv(file_path) if suffix == ".csv" else pd.read_excel(file_path)

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[INFO] Raw columns : {df.columns.tolist()}")
    print(f"[INFO] Raw shape   : {df.shape}")

    # ── Find date column ──────────────────────────────────────────
    date_col = None
    for candidate in ["date", "published_date", "published_date_(a.d.)",
                       "day", "period"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        # Any column with 'date' in the name
        for col in df.columns:
            if "date" in col:
                date_col = col
                break
    if date_col is None:
        raise ValueError(
            f"Date column not found.\nColumns: {df.columns.tolist()}"
        )
    print(f"[INFO] Using date column : {date_col}")

    # ── Find USD sell column ──────────────────────────────────────
    usd_col = None

    # Exact match first
    for candidate in ["usd_sell", "usd_sell_rate", "u.s._dollar_sell",
                       "us_dollar_sell", "dollar_sell", "usd"]:
        if candidate in df.columns:
            usd_col = candidate
            break

    # Contains 'usd' AND 'sell'
    if usd_col is None:
        for col in df.columns:
            if "usd" in col and "sell" in col:
                usd_col = col
                break

    # Contains 'sell' alone
    if usd_col is None:
        sell_cols = [c for c in df.columns if "sell" in c]
        print(f"[DEBUG] Columns containing 'sell': {sell_cols}")
        if sell_cols:
            usd_col = sell_cols[0]
            print(f"[WARN] Using first sell column: {usd_col}")

    # Contains 'usd' alone
    if usd_col is None:
        usd_cols = [c for c in df.columns if "usd" in c]
        print(f"[DEBUG] Columns containing 'usd': {usd_cols}")
        if usd_cols:
            usd_col = usd_cols[0]
            print(f"[WARN] Using first usd column: {usd_col}")

    if usd_col is None:
        raise ValueError(
            f"No USD sell column found.\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Open {file_path} to check the downloaded file's column names.\n"
            f"Then update the candidates list in extract_usd_sell()."
        )

    print(f"[INFO] Using USD sell column: {usd_col}")

    # ── Build clean output ────────────────────────────────────────
    out = df[[date_col, usd_col]].copy()
    out.columns = ["date", "usd_sell"]

    out["date"]     = pd.to_datetime(out["date"], errors="coerce")
    out["usd_sell"] = pd.to_numeric(out["usd_sell"], errors="coerce")

    out = (out.dropna(subset=["date", "usd_sell"])
              .sort_values("date")
              .drop_duplicates(subset=["date"])
              .reset_index(drop=True))

    return out


# =========================================================
# Merge with Existing File
# =========================================================
def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    if not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0:
        return new_df
    existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
    merged = (pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"])
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
    print("[INFO] NRB Exchange Rate Downloader — Fixed Version")
    print("=" * 60)

    # Check if update needed
    from_date, to_date = get_fetch_range()

    if from_date is None:
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        print(f"\n  Rows        : {len(existing)}")
        print(f"  Date range  : "
              f"{existing['date'].min().date()} → {existing['date'].max().date()}")
        print(f"  Latest rate : {existing.iloc[-1]['usd_sell']} NPR/USD")
        return

    # Download
    downloaded_file = download_forex_csv(from_date, to_date)

    # Extract
    new_df = extract_usd_sell(downloaded_file)
    print(f"[INFO] Extracted {len(new_df)} rows from downloaded file")

    # Merge
    final = merge_with_existing(new_df)

    # Save
    final.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Saved → {OUTPUT_FILE}")
    print(f"  Total rows  : {len(final)}")
    print(f"  Date range  : "
          f"{final['date'].min().date()} → {final['date'].max().date()}")
    print(f"  Latest rate : {final.iloc[-1]['usd_sell']} NPR/USD  "
          f"({final.iloc[-1]['date'].strftime('%d %b %Y')})")

    # Check for missing days (weekends/holidays are expected gaps)
    all_days = pd.date_range(
        start=final["date"].min(),
        end=final["date"].max(),
        freq="D"
    )
    missing = sorted(set(all_days) - set(pd.to_datetime(final["date"])))
    if missing:
        print(f"\n  ℹ Missing days: {len(missing)} "
              f"(weekends and public holidays are normal gaps)")
    else:
        print("  ✅ No missing days")

    print("\n[INFO] Latest 5 rows:")
    print(final.tail(5).to_string(index=False))
    print("=" * 60)

    # Cleanup temp download
    try:
        downloaded_file.unlink(missing_ok=True)
        print(f"[INFO] Deleted temp file: {downloaded_file.name}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
