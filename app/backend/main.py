"""
main.py — FastAPI Backend (EC2 + S3 Version)
---------------------------------------------
Loads model and data directly from S3.
No local CSV files needed.
"""

import io
import json
import math
import pickle
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
S3_BUCKET        = "kalimati-price-prediction"
S3_FEATURES_KEY  = "features/tomato_time_series_features.csv"
S3_BASE_DATA_KEY = "processed/tomato_base_data.csv"
S3_MODELS_PREFIX = "models/"

app = FastAPI(title="Kalimati Tomato Price API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def clean_value(v):
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return v


def clean_list(lst):
    return [clean_value(v) for v in lst]


# ---------------------------------------------------------
# Load Latest Model from S3
# ---------------------------------------------------------
model_bundle = None


def get_latest_model_key():
    """Find the latest model.tar.gz in S3."""
    s3       = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODELS_PREFIX)
    objects  = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].endswith("model.tar.gz")
    ]
    if not objects:
        return None
    latest = sorted(objects, key=lambda x: x["LastModified"], reverse=True)[0]
    return latest["Key"]


def load_model_from_s3():
    global model_bundle
    try:
        key = get_latest_model_key()
        if not key:
            print("[ERROR] No model found in S3!")
            return

        print(f"[INFO] Loading model from s3://{S3_BUCKET}/{key}")
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)

        # Extract model.tar.gz → model.pkl
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "model.tar.gz"
            tar_path.write_bytes(obj["Body"].read())

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            # Find model.pkl
            pkl_files = list(Path(tmpdir).rglob("model.pkl"))
            if not pkl_files:
                print("[ERROR] model.pkl not found in archive!")
                return

            with open(pkl_files[0], "rb") as f:
                model_bundle = pickle.load(f)

        print(f"[INFO] Model loaded! Keys: {list(model_bundle['models'].keys())}")

    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")


@app.on_event("startup")
def startup():
    load_model_from_s3()


# ---------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------
def load_base_data():
    df = read_csv_from_s3(S3_BASE_DATA_KEY)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


def load_features():
    df = read_csv_from_s3(S3_FEATURES_KEY)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model_bundle is not None
    }


@app.get("/api/stats")
def get_stats():
    df               = load_base_data()
    latest_price     = clean_value(round(float(df["Average_Price"].iloc[-1]), 2))
    avg_price        = clean_value(round(float(df["Average_Price"].mean()), 2))
    max_price        = clean_value(round(float(df["Average_Price"].max()), 2))
    min_price        = clean_value(round(float(df["Average_Price"].min()), 2))
    total_days       = len(df)
    last_7           = df.tail(7)
    price_change     = clean_value(round(float(last_7["Average_Price"].iloc[-1] - last_7["Average_Price"].iloc[0]), 2))
    price_change_pct = clean_value(round(float(price_change / last_7["Average_Price"].iloc[0] * 100), 2))

    return {
        "latest_price":     latest_price,
        "avg_price":        avg_price,
        "max_price":        max_price,
        "min_price":        min_price,
        "total_days":       total_days,
        "date_from":        str(df["Date"].min().date()),
        "date_to":          str(df["Date"].max().date()),
        "price_change_7d":  price_change,
        "price_change_pct": price_change_pct,
    }


@app.get("/api/historical")
def get_historical(days: int = 365):
    df = load_base_data()
    if days > 0:
        df = df.tail(days)
    return {
        "dates":  df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": clean_list(df["Average_Price"].round(2).tolist()),
    }


@app.get("/api/supply")
def get_supply(days: int = 365):
    df = load_base_data()
    if "Supply_Volume" not in df.columns:
        raise HTTPException(status_code=404, detail="Supply data not available")
    if days > 0:
        df = df.tail(days)
    df["Supply_Volume"] = pd.to_numeric(df["Supply_Volume"], errors="coerce").fillna(0)
    return {
        "dates":  df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "supply": clean_list(df["Supply_Volume"].round(2).tolist()),
    }


@app.get("/api/macro")
def get_macro(days: int = 365):
    df = load_base_data()
    if days > 0:
        df = df.tail(days)
    result = {"dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()}
    if "Diesel" in df.columns:
        result["diesel"] = clean_list(pd.to_numeric(df["Diesel"], errors="coerce").round(2).tolist())
    if "USD_TO_NPR" in df.columns:
        result["usd_to_npr"] = clean_list(pd.to_numeric(df["USD_TO_NPR"], errors="coerce").round(2).tolist())
    if "Inflation" in df.columns:
        result["inflation"] = clean_list(pd.to_numeric(df["Inflation"], errors="coerce").round(2).tolist())
    return result


@app.get("/api/weather")
def get_weather(days: int = 365):
    df = load_base_data()
    if days > 0:
        df = df.tail(days)
    result = {"dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()}
    if "Kathmandu_Temperature" in df.columns:
        result["temperature"] = clean_list(pd.to_numeric(df["Kathmandu_Temperature"], errors="coerce").round(2).tolist())
    if "Kathmandu_Precipitation" in df.columns:
        result["precipitation"] = clean_list(pd.to_numeric(df["Kathmandu_Precipitation"], errors="coerce").fillna(0).round(2).tolist())
    return result


@app.get("/api/predict")
def predict():
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    models        = model_bundle["models"]
    feature_names = model_bundle["feature_names"]

    df         = load_features()
    latest_row = df[feature_names].iloc[-1:].copy()
    latest_row = latest_row.ffill().fillna(0)

    last_date  = df["Date"].iloc[-1]
    last_price = float(df["Average_Price"].iloc[-1])

    predictions = []
    for h in range(1, 8):
        model = models[f"day_{h}"]
        pred  = float(model.predict(latest_row)[0])
        pred  = round(max(pred, 0), 2)
        pred_date = last_date + pd.Timedelta(days=h)
        predictions.append({
            "day":        h,
            "date":       pred_date.strftime("%Y-%m-%d"),
            "price":      pred,
            "change":     round(pred - last_price, 2),
            "change_pct": round((pred - last_price) / last_price * 100, 2),
        })

    avg_pred  = float(np.mean([p["price"] for p in predictions]))
    trend     = "up" if avg_pred > last_price else "down"
    trend_pct = round((avg_pred - last_price) / last_price * 100, 2)

    return {
        "last_known_date":  last_date.strftime("%Y-%m-%d"),
        "last_known_price": round(last_price, 2),
        "predictions":      predictions,
        "trend":            trend,
        "trend_pct":        trend_pct,
    }


@app.get("/api/metrics")
def get_metrics():
    try:
        # Load metrics from model bundle
        key = get_latest_model_key()
        if not key:
            raise HTTPException(status_code=404, detail="No model found")

        # Try to find metrics.json in S3
        metrics_key = key.replace("model.tar.gz", "metrics.json")
        s3  = boto3.client("s3")
        try:
            obj     = s3.get_object(Bucket=S3_BUCKET, Key=metrics_key)
            metrics = json.loads(obj["Body"].read())
        except Exception:
            # fallback — load local metrics.json if exists
            local = Path(__file__).parent / "model" / "metrics.json"
            if local.exists():
                with open(local) as f:
                    metrics = json.load(f)
            else:
                metrics = []

        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload-model")
def reload_model():
    """Reload the latest model from S3."""
    load_model_from_s3()
    return {"status": "ok", "model_loaded": model_bundle is not None}

