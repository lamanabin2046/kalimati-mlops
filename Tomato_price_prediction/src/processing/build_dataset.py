

"""
build_dataset.py
----------------
Build:
1. Base tomato dataset
2. Time-series tomato dataset

Features:
- Weather
- Supply
- Diesel price
- Inflation
- Exchange rate
- Knowledge Graph event risk
- Time features
- Lag features
- Rolling features
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.utils import clean_number, clean_commodity


DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW_KALIMATI = DATA_RAW / "kalimati"
DATA_RAW_MACRO = DATA_RAW / "macro"
DATA_RAW_WEATHER = DATA_RAW / "weather"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_FEATURES.mkdir(parents=True, exist_ok=True)


# =========================================================
# PRICE DATA
# =========================================================

def load_price_data():

    path = DATA_RAW_KALIMATI / "veg_price_list.csv"

    if not path.exists():
        raise FileNotFoundError("veg_price_list.csv not found")

    df = pd.read_csv(path, encoding="utf-8-sig")

    df = df[["Date", "कृषि उपज", "औसत"]].copy()

    df.rename(
        columns={
            "कृषि उपज": "commodity",
            "औसत": "Average_Price"
        },
        inplace=True
    )

    df["commodity"] = df["commodity"].apply(clean_commodity)
    df["Average_Price"] = df["Average_Price"].apply(clean_number)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    tomato_df = df[
        df["commodity"].isin(["Tomato", "Tomato_Big", "Tomato_Small"])
    ]

    tomato_df = tomato_df.dropna(subset=["Date", "Average_Price"])

    tomato_df = tomato_df.groupby("Date", as_index=False)["Average_Price"].mean()

    return tomato_df.sort_values("Date")


# =========================================================
# SUPPLY DATA
# =========================================================

def load_supply_data():

    path = DATA_RAW_KALIMATI / "supply_volume.csv"

    if not path.exists():
        return pd.DataFrame(columns=["Date", "Supply_Volume"])

    df = pd.read_csv(path, encoding="utf-8-sig")

    df = df[["Date", "कृषि उपज", "आगमन"]]

    df.rename(
        columns={
            "कृषि उपज": "commodity",
            "आगमन": "Supply_Volume"
        },
        inplace=True
    )

    df["commodity"] = df["commodity"].apply(clean_commodity)
    df["Supply_Volume"] = df["Supply_Volume"].apply(clean_number)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df[df["commodity"].isin(["Tomato", "Tomato_Big", "Tomato_Small"])]

    df = df.groupby("Date", as_index=False)["Supply_Volume"].sum()

    return df.sort_values("Date")


# =========================================================
# WEATHER DATA
# =========================================================

def load_weather_data():

    path = DATA_RAW_WEATHER / "weather.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    df.rename(columns={"date": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"])

    temp_cols = [c for c in df.columns if "temp" in c.lower()]
    rain_cols = [c for c in df.columns if "rain" in c.lower()]

    for col in temp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].ffill()

    for col in rain_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    return df.sort_values("Date")


# =========================================================
# DIESEL DATA
# =========================================================

def load_fuel_data():

    path = DATA_RAW_MACRO / "diesel.csv"

    if not path.exists():
        return pd.DataFrame(columns=["Date", "Diesel"])

    df = pd.read_csv(path)

    df.rename(columns={"date": "Date", "diesel": "Diesel"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Diesel"] = pd.to_numeric(df["Diesel"], errors="coerce")

    df = df.dropna(subset=["Date"])

    return df.sort_values("Date")


# =========================================================
# INFLATION DATA
# =========================================================

def load_inflation_data():

    path = DATA_RAW_MACRO / "inflation.csv"

    if not path.exists():
        return pd.DataFrame(columns=["Date", "Inflation"])

    df = pd.read_csv(path)

    df.rename(columns={"date": "Date", "inflation": "Inflation"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Inflation"] = pd.to_numeric(df["Inflation"], errors="coerce")

    df = df.dropna(subset=["Date"])

    return df.sort_values("Date")


# =========================================================
# EXCHANGE RATE
# =========================================================

def load_exchange_data():

    path = DATA_RAW_MACRO / "exchange_rate_usd_sell.csv"

    if not path.exists():
        return pd.DataFrame(columns=["Date", "USD_TO_NPR"])

    df = pd.read_csv(path)

    df.rename(
        columns={"date": "Date", "usd_sell": "USD_TO_NPR"},
        inplace=True
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["USD_TO_NPR"] = pd.to_numeric(df["USD_TO_NPR"], errors="coerce")

    df = df.dropna(subset=["Date"])

    return df.sort_values("Date")


# =========================================================
# EVENT RISK (KG)
# =========================================================

def load_event_risk_data():

    path = DATA_PROCESSED / "daily_event_risk.csv"

    if not path.exists():
        return pd.DataFrame(columns=["Date"])

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df.sort_values("Date")


# =========================================================
# MERGE DATA
# =========================================================

def merge_all(price_df, supply_df, weather_df,
              fuel_df, inflation_df, exchange_df,
              event_df):

    df = price_df.copy()

    if not supply_df.empty:
        df = pd.merge(df, supply_df, on="Date", how="left")

    if not exchange_df.empty:
        df = pd.merge(df, exchange_df, on="Date", how="left")

    if not fuel_df.empty:
        df = pd.merge(df, fuel_df, on="Date", how="left")

    if not inflation_df.empty:
        df = pd.merge(df, inflation_df, on="Date", how="left")

    if not weather_df.empty:
        df = pd.merge(df, weather_df, on="Date", how="left")

    if not event_df.empty:
        df = pd.merge(df, event_df, on="Date", how="left")

    df = df.sort_values("Date")

    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")

    df = (
        df.set_index("Date")
        .reindex(full_dates)
        .rename_axis("Date")
        .reset_index()
    )

    df["Average_Price"] = df["Average_Price"].ffill()

    if "Supply_Volume" in df.columns:
        df["Supply_Volume"] = df["Supply_Volume"].ffill()

    macro_cols = ["USD_TO_NPR", "Diesel", "Inflation"]

    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    rain_cols = [c for c in df.columns if "Rainfall" in c]

    for col in rain_cols:
        df[col] = df[col].fillna(0)

    risk_cols = [c for c in df.columns if "Risk" in c]

    for col in risk_cols:
        df[col] = df[col].fillna(0)

    return df


# =========================================================
# QUALITY REPORT
# =========================================================

def print_quality_report(df, label):

    print("\n===== DATA QUALITY REPORT =====")
    print(label)

    print("Shape:", df.shape)
    print("Date range:", df["Date"].min(), "→", df["Date"].max())

    print("Duplicates:", df.duplicated().sum())

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]

    if len(nulls) == 0:
        print("No null values")
    else:
        print(nulls)


# =========================================================
# TIME FEATURES
# =========================================================

def add_time_features(df):

    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.weekday

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# =========================================================
# FESTIVAL FEATURE
# =========================================================

def add_festival_feature(df):

    df["is_festival"] = 0

    df.loc[(df["month"] == 3) & (df["day"].between(1, 20)), "is_festival"] = 1
    df.loc[(df["month"] == 4) & (df["day"].between(10, 20)), "is_festival"] = 1
    df.loc[((df["month"] == 9) & (df["day"] >= 25)) | ((df["month"] == 10) & (df["day"] <= 15)), "is_festival"] = 1
    df.loc[(df["month"] == 11) & (df["day"].between(1, 15)), "is_festival"] = 1

    return df


# =========================================================
# LAG FEATURES
# =========================================================

def add_lag_features(df):

    df["price_lag1"] = df["Average_Price"].shift(1)
    df["price_lag3"] = df["Average_Price"].shift(3)
    df["price_lag7"] = df["Average_Price"].shift(7)

    return df


# =========================================================
# ROLLING FEATURES
# =========================================================

def add_rolling_features(df):

    df["price_roll_mean_3"] = df["Average_Price"].shift(1).rolling(3).mean()
    df["price_roll_mean_7"] = df["Average_Price"].shift(1).rolling(7).mean()
    df["price_roll_std_7"] = df["Average_Price"].shift(1).rolling(7).std()

    return df


# =========================================================
# BUILD DATASETS
# =========================================================

def build_base_dataset():

    price = load_price_data()
    supply = load_supply_data()
    weather = load_weather_data()
    fuel = load_fuel_data()
    inflation = load_inflation_data()
    exchange = load_exchange_data()
    event = load_event_risk_data()

    base_df = merge_all(
        price, supply, weather,
        fuel, inflation, exchange,
        event
    )

    print_quality_report(base_df, "Base Dataset")

    path = DATA_PROCESSED / "tomato_base_data.csv"

    base_df.to_csv(path, index=False)

    return base_df


def build_time_series_dataset(base_df):

    df = base_df.copy()

    df = add_time_features(df)
    df = add_festival_feature(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = df.dropna().reset_index(drop=True)

    print_quality_report(df, "Time Series Dataset")

    path = DATA_FEATURES / "tomato_time_series_features.csv"

    df.to_csv(path, index=False)

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    base_df = build_base_dataset()

    build_time_series_dataset(base_df)

    print("\nDataset build complete")


if __name__ == "__main__":
    main()