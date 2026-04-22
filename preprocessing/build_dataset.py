"""
build_dataset.py — EC2 + S3 Version
--------------------------------------
- Reads all raw CSVs from S3
- Merges and builds features
- Saves processed datasets back to S3

S3 inputs:
  raw/kalimati/veg_price_list.csv
  raw/kalimati/supply_volume.csv
  raw/weather/weather.csv
  raw/macro/diesel.csv
  raw/macro/inflation.csv
  raw/macro/exchange_rate_usd_sell.csv
  processed/daily_event_risk.csv

S3 outputs:
  processed/tomato_base_data.csv
  features/tomato_time_series_features.csv
"""

import io
import boto3
import pandas as pd
import numpy as np
from utils import clean_number, clean_commodity

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET = "kalimati-price-prediction"

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket, key, **kwargs):
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()), **kwargs)
        print(f"[INFO] Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df
    except Exception as e:
        print(f"[WARN] Could not read {key}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()


def write_csv_to_s3(df, bucket, key):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Load Functions
# ---------------------------------------------------------
def load_price_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/kalimati/veg_price_list.csv", encoding_errors="replace")
    if df.empty:
        raise FileNotFoundError("veg_price_list.csv not found in S3")

    cols = df.columns.tolist()
    print(f"[INFO] Price columns: {cols}")

    df = df[["Date", "कृषि उपज", "औसत"]].copy()
    df.rename(columns={"कृषि उपज": "commodity", "औसत": "Average_Price"}, inplace=True)

    df["commodity"]     = df["commodity"].apply(clean_commodity)
    df["Average_Price"] = df["Average_Price"].apply(clean_number)
    df["Date"]          = pd.to_datetime(df["Date"], errors="coerce")

    tomato_df = df[df["commodity"].isin(["Tomato", "Tomato_Big", "Tomato_Small"])]
    tomato_df = tomato_df.dropna(subset=["Date", "Average_Price"])
    tomato_df = tomato_df.groupby("Date", as_index=False)["Average_Price"].mean()

    print(f"[INFO] Price date range: {tomato_df['Date'].min().date()} → {tomato_df['Date'].max().date()}")
    return tomato_df.sort_values("Date")


def load_supply_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/kalimati/supply_volume.csv", encoding_errors="replace")
    if df.empty:
        return pd.DataFrame(columns=["Date", "Supply_Volume"])

    df = df[["Date", "कृषि उपज", "आगमन"]].copy()
    df.rename(columns={"कृषि उपज": "commodity", "आगमन": "Supply_Volume"}, inplace=True)

    df["commodity"]     = df["commodity"].apply(clean_commodity)
    df["Supply_Volume"] = df["Supply_Volume"].apply(clean_number)
    df["Date"]          = pd.to_datetime(df["Date"], errors="coerce")

    df = df[df["commodity"].isin(["Tomato", "Tomato_Big", "Tomato_Small"])]
    df = df.groupby("Date", as_index=False)["Supply_Volume"].sum()

    print(f"[INFO] Supply date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df.sort_values("Date")


def load_weather_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/weather/weather.csv")
    if df.empty:
        return pd.DataFrame()

    df.rename(columns={"date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    temp_cols = [c for c in df.columns if "temp" in c.lower()]
    rain_cols = [c for c in df.columns if "rain" in c.lower() or "precipitation" in c.lower()]

    for col in temp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate()
    for col in rain_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"[INFO] Weather date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df.sort_values("Date")


def load_fuel_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/macro/diesel.csv")
    if df.empty:
        return pd.DataFrame(columns=["Date", "Diesel"])
    df.rename(columns={"date": "Date", "diesel": "Diesel"}, inplace=True)
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
    df["Diesel"] = pd.to_numeric(df["Diesel"], errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
    print(f"[INFO] Diesel date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_inflation_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/macro/inflation.csv")
    if df.empty:
        return pd.DataFrame(columns=["Date", "Inflation"])
    df.rename(columns={"Date": "Date", "Inflation": "Inflation"}, inplace=True)
    df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
    df["Inflation"] = pd.to_numeric(df["Inflation"], errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
    print(f"[INFO] Inflation date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_exchange_data():
    df = read_csv_from_s3(S3_BUCKET, "raw/macro/exchange_rate_usd_sell.csv")
    if df.empty:
        return pd.DataFrame(columns=["Date", "USD_TO_NPR"])
    df.rename(columns={"date": "Date", "usd_sell": "USD_TO_NPR"}, inplace=True)
    df["Date"]       = pd.to_datetime(df["Date"], errors="coerce")
    df["USD_TO_NPR"] = pd.to_numeric(df["USD_TO_NPR"], errors="coerce")
    df = df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
    print(f"[INFO] Exchange date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_event_risk_data():
    df = read_csv_from_s3(S3_BUCKET, "processed/daily_event_risk.csv")
    if df.empty:
        return pd.DataFrame(columns=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date")


# ---------------------------------------------------------
# Merge All
# ---------------------------------------------------------
def merge_all(price_df, supply_df, weather_df,
              fuel_df, inflation_df, exchange_df, event_df):
    df = price_df.copy()

    for other_df in [supply_df, exchange_df, fuel_df, inflation_df, weather_df, event_df]:
        if not other_df.empty:
            df = pd.merge(df, other_df, on="Date", how="left")

    df = df.sort_values("Date")
    df = df.dropna(subset=["Date"])

    # Fill full date range
    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = (df.set_index("Date")
            .reindex(full_dates)
            .rename_axis("Date")
            .reset_index())

    # Macro: forward fill then backward fill
    for col in ["USD_TO_NPR", "Diesel", "Inflation"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Price: forward fill
    df["Average_Price"] = df["Average_Price"].ffill()

    # Supply: interpolate
    if "Supply_Volume" in df.columns:
        df["Supply_Volume"] = df["Supply_Volume"].interpolate(method="linear").ffill().bfill()

    # Weather: interpolate temperature, fill rain with 0
    for col in [c for c in df.columns if "Temperature" in c or "Pressure" in c or "Wind" in c]:
        df[col] = df[col].interpolate(method="linear").ffill().bfill()

    for col in [c for c in df.columns if "Rain" in c or "Precipitation" in c]:
        df[col] = df[col].fillna(0)

    # Event risk: fill 0
    for col in [c for c in df.columns if "Risk" in c]:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------
def add_time_features(df):
    df["day"]         = df["Date"].dt.day
    df["month"]       = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.weekday
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_festival_feature(df):
    df["is_festival"] = 0
    df.loc[(df["month"] == 3) & (df["day"].between(1, 20)), "is_festival"] = 1
    df.loc[(df["month"] == 4) & (df["day"].between(10, 20)), "is_festival"] = 1
    df.loc[((df["month"] == 9) & (df["day"] >= 25)) | ((df["month"] == 10) & (df["day"] <= 15)), "is_festival"] = 1
    df.loc[(df["month"] == 11) & (df["day"].between(1, 15)), "is_festival"] = 1
    return df


def add_lag_features(df):
    df["price_lag1"] = df["Average_Price"].shift(1)
    df["price_lag3"] = df["Average_Price"].shift(3)
    df["price_lag7"] = df["Average_Price"].shift(7)
    return df


def add_rolling_features(df):
    df["price_roll_mean_3"] = df["Average_Price"].shift(1).rolling(3).mean()
    df["price_roll_mean_7"] = df["Average_Price"].shift(1).rolling(7).mean()
    df["price_roll_std_7"]  = df["Average_Price"].shift(1).rolling(7).std()
    return df


def print_quality_report(df, label):
    print(f"\n===== DATA QUALITY REPORT: {label} =====")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print("Nulls:", nulls.to_dict() if len(nulls) > 0 else "None ✅")


# ---------------------------------------------------------
# Build Datasets
# ---------------------------------------------------------
def build_base_dataset():
    price     = load_price_data()
    supply    = load_supply_data()
    weather   = load_weather_data()
    fuel      = load_fuel_data()
    inflation = load_inflation_data()
    exchange  = load_exchange_data()
    event     = load_event_risk_data()

    base_df = merge_all(price, supply, weather, fuel, inflation, exchange, event)
    print_quality_report(base_df, "Base Dataset")
    write_csv_to_s3(base_df, S3_BUCKET, "processed/tomato_base_data.csv")
    return base_df


def build_time_series_dataset(base_df):
    df = base_df.copy()
    df = add_time_features(df)
    df = add_festival_feature(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Only drop rows where core features are missing
    df = df.dropna(subset=["Average_Price", "price_lag1", "price_lag3", "price_lag7"]).reset_index(drop=True)

    print_quality_report(df, "Time Series Dataset")
    write_csv_to_s3(df, S3_BUCKET, "features/tomato_time_series_features.csv")
    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("[INFO] Starting preprocessing pipeline...")
    base_df = build_base_dataset()
    build_time_series_dataset(base_df)
    print("\n[DONE] Dataset build complete!")


if __name__ == "__main__":
    main()

