"""
train.py — SageMaker XGBoost Training Script
----------------------------------------------
Trains 7 separate XGBoost models for next 7 day
tomato price prediction.

SageMaker passes these environment variables:
  SM_CHANNEL_TRAIN  → input data directory
  SM_MODEL_DIR      → where to save model
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# SageMaker Paths
# ---------------------------------------------------------
TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# ---------------------------------------------------------
# Features to use
# ---------------------------------------------------------
FEATURE_COLS = [
    "Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation",
    "day", "month", "day_of_week", "month_sin", "month_cos",
    "is_festival",
    "price_lag1", "price_lag3", "price_lag7",
    "price_roll_mean_3", "price_roll_mean_7", "price_roll_std_7",
    "Dhading_Temperature", "Dhading_Precipitation",
    "Kathmandu_Temperature", "Kathmandu_Precipitation",
    "Kavre_Temperature", "Kavre_Precipitation",
    "Sarlahi_Temperature", "Sarlahi_Precipitation",
    "Kathmandu_Risk", "Sarlahi_Risk", "Dhading_Risk",
    "Kavre_Risk", "Market_Risk",
]

TARGET_COL  = "Average_Price"
FORECAST_HORIZON = 7   # predict next 7 days

# ---------------------------------------------------------
# XGBoost Parameters
# ---------------------------------------------------------
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        6,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
def load_data():
    csv_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {TRAIN_DIR}")

    path = os.path.join(TRAIN_DIR, csv_files[0])
    df   = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} rows from {path}")
    print(f"[INFO] Date range: {df['Date'].min()} → {df['Date'].max()}")

    return df


# ---------------------------------------------------------
# Prepare Features
# ---------------------------------------------------------
def prepare_features(df):
    # Keep only available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        print(f"[WARN] Missing feature columns: {missing}")

    print(f"[INFO] Using {len(available)} features: {available}")

    X = df[available].copy()
    y = df[TARGET_COL].copy()

    # Fill any remaining nulls
    X = X.ffill().fillna(0)

    return X, y, available


# ---------------------------------------------------------
# Train / Test Split (time-based)
# ---------------------------------------------------------
def time_split(X, y, test_size=0.15):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[INFO] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Train One Model for Horizon h
# ---------------------------------------------------------
def train_model_for_horizon(X_train, X_test, y_train, y_test, horizon):
    """
    Shift target by `horizon` days to train a model
    that predicts `horizon` days ahead.
    """
    print(f"\n[INFO] Training model for horizon = {horizon} day(s) ahead...")

    y_train_shifted = y_train.shift(-horizon).dropna()
    y_test_shifted  = y_test.shift(-horizon).dropna()

    X_train_h = X_train.iloc[:len(y_train_shifted)]
    X_test_h  = X_test.iloc[:len(y_test_shifted)]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train_h, y_train_shifted,
        eval_set=[(X_test_h, y_test_shifted)],
        verbose=50,
    )

    # Evaluate
    preds = model.predict(X_test_h)
    mae   = mean_absolute_error(y_test_shifted, preds)
    rmse  = np.sqrt(mean_squared_error(y_test_shifted, preds))
    mape  = np.mean(np.abs((y_test_shifted - preds) / (y_test_shifted + 1e-8))) * 100

    print(f"[METRICS] Horizon={horizon} | MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

    return model, {"horizon": horizon, "mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("[INFO] SageMaker XGBoost Training Started")
    print("=" * 60)

    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = time_split(X, y)

    # Train 7 models (one per forecast horizon)
    models  = {}
    metrics = []

    for h in range(1, FORECAST_HORIZON + 1):
        model, metric = train_model_for_horizon(
            X_train, X_test, y_train, y_test, horizon=h
        )
        models[f"day_{h}"] = model
        metrics.append(metric)

    # Save all models as a single pickle file
    model_bundle = {
        "models":        models,
        "feature_names": feature_names,
        "forecast_horizon": FORECAST_HORIZON,
    }

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n[SUCCESS] Model bundle saved → {model_path}")

    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Metrics saved → {metrics_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("[SUMMARY] Training Complete")
    print("=" * 60)
    for m in metrics:
        print(f"  Day {m['horizon']:2d}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  MAPE={m['mape']:.2f}%")

    avg_mae = np.mean([m["mae"] for m in metrics])
    print(f"\n  Average MAE across 7 days: {avg_mae:.2f} NPR")
    print("=" * 60)


if __name__ == "__main__":
    main()

