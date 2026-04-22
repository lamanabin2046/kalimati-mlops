"""
weather_ingestion - AWS Lambda Version
----------------------------------------
- Reads existing weather.csv from S3
- Fetches only new data (incremental)
- Writes updated weather.csv back to S3

S3 path: s3://kalimati-price-prediction/raw/weather/weather.csv
"""

import io
import requests
import pandas as pd
import boto3
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET = "kalimati-price-prediction"
S3_KEY    = "raw/weather/weather.csv"

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------
# District coordinates
# ---------------------------------------------------------
DISTRICTS = {
    "Dhading":   {"lat": 27.8667, "lon": 84.9167},
    "Kathmandu": {"lat": 27.7172, "lon": 85.3240},
    "Kavre":     {"lat": 27.6240, "lon": 85.5475},
    "Sarlahi":   {"lat": 26.9833, "lon": 85.5500},
}

DAILY_VARS = (
    "temperature_2m_max,temperature_2m_min,"
    "surface_pressure_max,surface_pressure_min,"
    "wind_speed_10m_max,precipitation_sum"
)

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read CSV from S3 and return DataFrame. Returns empty DataFrame if not found."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"[INFO] Loaded existing data from s3://{bucket}/{key} — {len(df)} rows")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"[INFO] No existing file found at s3://{bucket}/{key}. Starting fresh.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Could not read from S3: {e}. Starting fresh.")
        return pd.DataFrame()


def write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    """Write DataFrame as CSV to S3."""
    s3 = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Date Helper
# ---------------------------------------------------------
def today_nepal_date():
    now_utc = datetime.now(timezone.utc)
    return (now_utc + timedelta(hours=5, minutes=45)).date()


# ---------------------------------------------------------
# Weather Fetch
# ---------------------------------------------------------
def fetch_weather(lat, lon, start_date, end_date):
    """Fetch daily historical data for one district."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "daily":      DAILY_VARS,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "timezone":   "Asia/Kathmandu"
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        print(f"[WARN] No daily data for lat={lat}, lon={lon}")
        return pd.DataFrame(columns=["date"])

    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])

    df["Temperature"] = df[["temperature_2m_max", "temperature_2m_min"]].mean(axis=1)
    df["Air_Pressure"] = df[["surface_pressure_max", "surface_pressure_min"]].mean(axis=1)

    df.rename(columns={
        "wind_speed_10m_max": "Wind_Speed",
        "precipitation_sum":  "Precipitation",
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


# ---------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------
def lambda_handler(event, context):
    print("[INFO] Weather ingestion Lambda started")

    today     = today_nepal_date()
    yesterday = today - timedelta(days=1)

    # Step 1: Read existing data from S3
    old = read_csv_from_s3(S3_BUCKET, S3_KEY)

    if not old.empty:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        last_date   = old["date"].max().date()
        print(f"[INFO] Last recorded weather date: {last_date}")
    else:
        last_date = datetime(2021, 12, 31).date()
        print("[INFO] No previous data. Starting from 2022-01-01.")

    # Step 2: Check if already up to date
    if last_date >= yesterday:
        print("[INFO] Weather data already up to date.")
        return {"statusCode": 200, "body": "Already up to date"}

    start_date = last_date + timedelta(days=1)
    end_date   = yesterday
    print(f"[INFO] Fetching weather from {start_date} to {end_date}...")

    # Step 3: Fetch new data
    df_new = merge_districts(start_date, end_date)

    if df_new.empty:
        print("[WARN] No new weather data returned.")
        return {"statusCode": 200, "body": "No new data"}

    # Step 4: Merge old + new and write back to S3
    df_combined = pd.concat([old, df_new], ignore_index=True)
    df_combined = (df_combined
                   .drop_duplicates(subset=["date"])
                   .sort_values("date")
                   .reset_index(drop=True))

    write_csv_to_s3(df_combined, S3_BUCKET, S3_KEY)

    return {
        "statusCode": 200,
        "body": f"Added {len(df_new)} new rows. Total: {len(df_combined)} rows."
    }
