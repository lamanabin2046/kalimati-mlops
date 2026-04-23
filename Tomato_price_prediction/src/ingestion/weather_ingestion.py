import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "weather.csv"

# ---------------------------------------------------------
# District coordinates
# ---------------------------------------------------------
DISTRICTS = {
    "Dhading": {"lat": 27.8667, "lon": 84.9167},
    "Kathmandu": {"lat": 27.7172, "lon": 85.3240},
    "Kavre": {"lat": 27.6240, "lon": 85.5475},
    "Sarlahi": {"lat": 26.9833, "lon": 85.5500},
}

# ---------------------------------------------------------
# Variables for daily dataset
# ---------------------------------------------------------
DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "surface_pressure_max,surface_pressure_min,"
    "wind_speed_10m_max,precipitation_sum"
)

def today_nepal_date():
    now_utc = datetime.now(timezone.utc)
    return (now_utc + timedelta(hours=5, minutes=45)).date()

def fetch_weather(lat, lon, start_date, end_date):
    """Fetch daily historical data for one district."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "Asia/Kathmandu"
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        print(f"[WARN] No daily data for lat={lat}, lon={lon}")
        return pd.DataFrame(columns=["date"])

    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])

    # Derived metrics
    df["Temperature"] = df[["temperature_2m_max", "temperature_2m_min"]].mean(axis=1)
    df["Air_Pressure"] = df[["surface_pressure_max", "surface_pressure_min"]].mean(axis=1)

    df.rename(columns={
        "wind_speed_10m_max": "Wind_Speed",
        "precipitation_sum": "Precipitation",
    }, inplace=True)

    df["Rainfall_MM"] = df["Precipitation"]

    return df[["date", "Temperature", "Air_Pressure", "Wind_Speed", "Precipitation", "Rainfall_MM"]]

def merge_districts(start_date, end_date):
    """Fetch and merge weather for all districts."""
    frames = []

    for name, coords in DISTRICTS.items():
        print(f"[INFO] Fetching weather for {name}...")
        df = fetch_weather(coords["lat"], coords["lon"], start_date, end_date)
        df = df.add_prefix(f"{name}_")
        df.rename(columns={f"{name}_date": "date"}, inplace=True)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_final = frames[0]
    for df in frames[1:]:
        df_final = pd.merge(df_final, df, on="date", how="outer")

    return df_final.sort_values("date")

def main():
    today = today_nepal_date()
    yesterday = today - timedelta(days=1)

    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 0:
        old = pd.read_csv(OUT_FILE)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        last_date = old["date"].max().date()
        print(f"[INFO] Last recorded weather date: {last_date}")
    else:
        old = pd.DataFrame()
        last_date = datetime.strptime("2021-12-31", "%Y-%m-%d").date()
        print("[INFO] No previous weather file found. Starting fresh.")

    if last_date >= yesterday:
        print("[INFO] Weather data already up to date.")
        return

    start_date = last_date + timedelta(days=1)
    end_date = yesterday

    print(f"[INFO] Downloading weather data from {start_date} to {end_date}...")

    df_new = merge_districts(start_date, end_date)

    if not df_new.empty:
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        df.to_csv(OUT_FILE, index=False)
        print(f"[SUCCESS] Weather data saved to: {OUT_FILE}")
    else:
        print("[WARN] No new weather data returned.")

if __name__ == "__main__":
    main()