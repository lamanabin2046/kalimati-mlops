"""
Tomato Price Shock Intelligence System
Complete training and evaluation script for CEAI publication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats as scipy_stats

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, classification_report, roc_auc_score)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)


# ================================================================
# CONFIG
# ================================================================

HORIZONS        = [1, 2, 3, 4, 5, 6, 7]
PRIMARY_HORIZON = 7
SPLIT           = 0.80
SPIKE_SIGMA     = 1.5
RANDOM_STATE    = 42

# District supply shares — from KG construction
SUPPLY_SHARES = {
    "Sarlahi":   0.40,
    "Dhading":   0.25,
    "Kavre":     0.20,
    "Kathmandu": 0.15,
}


# ================================================================
# STEP 1 — LOAD AND CLEAN
# ================================================================

print("=" * 60)
print("STEP 1: DATA LOADING AND CLEANING")
print("=" * 60)

df = pd.read_csv("data/processed/tomato_base_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date").asfreq("D")

# ── Drop useless metadata columns from exchange rate scraper ──
metadata_cols = [
    "currency_name", "currency_iso3", "unit",
    "usd_buy", "published_on", "modified_on"
]
dropped = [c for c in metadata_cols if c in df.columns]
df.drop(columns=dropped, inplace=True)
print(f"Dropped metadata columns: {dropped}")

# ── Remove zero / negative prices ──
bad = df["Average_Price"] <= 0
if bad.sum() > 0:
    print(f"Removing {bad.sum()} rows with zero/negative price")
    df = df[~bad]

print(f"\nShape after cleaning: {df.shape}")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")

# ── Missing value report ──
print("\nMissing values before imputation:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

# ── Imputation ──
# Weather & risk: linear interpolation (physically continuous signals)
weather_cols = [c for c in df.columns if any(
    tag in c for tag in [
        "Temperature", "Air_Pressure", "Wind_Speed",
        "Precipitation", "Rainfall_MM"
    ]
)]
risk_cols = [c for c in df.columns if "Risk" in c]

for col in weather_cols + risk_cols:
    df[col] = df[col].interpolate(method="linear").ffill().bfill()

# Macroeconomic: forward-fill then back-fill (published monthly)
for col in ["Diesel", "Inflation", "USD_TO_NPR"]:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

# Price and supply: forward-fill only
df["Average_Price"]  = df["Average_Price"].ffill().bfill()
df["Supply_Volume"]  = df["Supply_Volume"].ffill().bfill()

# Verify no remaining nulls in key columns
key_cols = ["Average_Price", "Supply_Volume",
            "USD_TO_NPR", "Diesel", "Inflation"] + weather_cols + risk_cols
remaining = df[key_cols].isnull().sum().sum()
print(f"\nRemaining nulls in key columns after imputation: {remaining}")


# ================================================================
# STEP 2 — DERIVE FULL KG FEATURES FROM DISTRICT RISK COLUMNS
# ================================================================
# Your CSV already has: Kathmandu_Risk, Kavre_Risk,
#                       Sarlahi_Risk, Dhading_Risk, Market_Risk
# We derive the remaining 4 KG features from these.

print("\n" + "=" * 60)
print("STEP 2: DERIVING FULL KG FEATURE SET")
print("=" * 60)

district_risk_cols = {
    "Kathmandu": "Kathmandu_Risk",
    "Kavre":     "Kavre_Risk",
    "Sarlahi":   "Sarlahi_Risk",
    "Dhading":   "Dhading_Risk",
}

# Verify all district risk columns exist
for district, col in district_risk_cols.items():
    if col not in df.columns:
        print(f"WARNING: {col} not found — filling with 0")
        df[col] = 0.0

# 1. disruption_flag — any active disruption on this day
df["disruption_flag"] = (df["Market_Risk"] > 0).astype(int)

# 2. active_disruption_count — number of districts with active risk
risk_matrix = df[[col for col in district_risk_cols.values()]]
df["active_disruption_count"] = (risk_matrix > 0).sum(axis=1)

# 3. max_severity — highest single district severity
df["max_severity"] = risk_matrix.max(axis=1)

# 4. weighted_route_blockage — supply-weighted fraction of routes blocked
df["weighted_route_blockage"] = sum(
    (df[col] > 0).astype(float) * SUPPLY_SHARES[district]
    for district, col in district_risk_cols.items()
)

print("KG features derived:")
print(f"  disruption_flag:         {df['disruption_flag'].sum()} active days "
      f"({100*df['disruption_flag'].mean():.1f}%)")
print(f"  active_disruption_count: mean={df['active_disruption_count'].mean():.2f}")
print(f"  max_severity:            mean={df['max_severity'].mean():.3f}  "
      f"max={df['max_severity'].max():.3f}")
print(f"  weighted_route_blockage: mean={df['weighted_route_blockage'].mean():.3f}")
print(f"  Market_Risk:             mean={df['Market_Risk'].mean():.3f}")


# ================================================================
# STEP 3 — FEATURE ENGINEERING
# ================================================================

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

month       = df.index.month
day         = df.index.day
day_of_year = df.index.dayofyear

# ── Lag features ──
for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
    df[f"lag{lag}"] = df["Average_Price"].shift(lag)

# ── Rolling statistics ──
for window in [7, 14, 21]:
    df[f"roll_mean{window}"] = df["Average_Price"].shift(1).rolling(window).mean()
    df[f"roll_std{window}"]  = df["Average_Price"].shift(1).rolling(window).std()

# ── Momentum & volatility ──
df["momentum7"]    = df["lag1"] - df["lag7"]
df["momentum14"]   = df["lag1"] - df["lag14"]
df["momentum21"]   = df["lag1"] - df["lag21"]
df["accel"]        = df["momentum7"] - (df["lag7"] - df["lag14"])
df["zscore14"]     = ((df["lag1"] - df["roll_mean14"])
                      / df["roll_std14"].replace(0, 1))
df["price_range7"] = (
    df["Average_Price"].shift(1).rolling(7).max() -
    df["Average_Price"].shift(1).rolling(7).min()
)
df["volatility7"] = df["roll_std7"] / (df["roll_mean7"] + 1e-6)

# ── Weather aggregation ──
temp_cols = [c for c in df.columns if "Temperature" in c]
rain_cols = [c for c in df.columns if "Rainfall_MM" in c]
pres_cols = [c for c in df.columns if "Air_Pressure" in c]
wind_cols = [c for c in df.columns if "Wind_Speed" in c]
prec_cols = [c for c in df.columns if "Precipitation" in c]

if temp_cols:
    df["Temp_mean"] = df[temp_cols].mean(axis=1)
    df["Temp_std"]  = df[temp_cols].std(axis=1)
if rain_cols:
    df["Rainfall_mean"] = df[rain_cols].mean(axis=1)
    df["Rainfall_max"]  = df[rain_cols].max(axis=1)
if pres_cols:
    df["Pressure_mean"] = df[pres_cols].mean(axis=1)
if wind_cols:
    df["Wind_mean"]     = df[wind_cols].mean(axis=1)
    df["Wind_max"]      = df[wind_cols].max(axis=1)
if prec_cols:
    df["Precip_mean"]   = df[prec_cols].mean(axis=1)
    df["Precip_max"]    = df[prec_cols].max(axis=1)

# ── Nepal-specific temporal features ──
df["month_sin"]        = np.sin(2 * np.pi * month / 12)
df["month_cos"]        = np.cos(2 * np.pi * month / 12)
df["month_progress"]   = day / 30
df["is_monsoon"]       = month.isin([6, 7, 8, 9]).astype(int)
df["is_dashain_tihar"] = month.isin([10, 11]).astype(int)
df["is_terai_harvest"] = month.isin([12, 1, 2]).astype(int)
df["day_of_week"]      = df.index.dayofweek
df["is_weekend"]       = df["day_of_week"].isin([5, 6]).astype(int)

dist_dashain           = np.minimum(
    np.abs(day_of_year - 280),
    365 - np.abs(day_of_year - 280)
)
df["weeks_to_dashain"] = dist_dashain / 7

# ── Interaction terms ──
df["mom7_x_monsoon"]  = df["momentum7"] * df["is_monsoon"]
df["mom7_x_festival"] = df["momentum7"] * df["is_dashain_tihar"]

print(f"Total columns after feature engineering: {df.shape[1]}")


# ================================================================
# STEP 4 — FEATURE GROUPS
# ================================================================

macro_features = [c for c in
    ["Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation"]
    if c in df.columns]

# Extended weather: now includes pressure, wind, precipitation
weather_features = [c for c in
    ["Temp_mean", "Temp_std",
     "Rainfall_mean", "Rainfall_max",
     "Pressure_mean",
     "Wind_mean", "Wind_max",
     "Precip_mean", "Precip_max"]
    if c in df.columns]

# Full 5-feature KG set (derived in Step 2)
kg_features = [c for c in
    ["Market_Risk", "disruption_flag",
     "active_disruption_count", "max_severity",
     "weighted_route_blockage"]
    if c in df.columns]

lag_features = [c for c in
    ["lag1", "lag2", "lag3", "lag5",
     "lag7", "lag14", "lag21", "lag28"]
    if c in df.columns]

rolling_features = [c for c in
    ["roll_mean7", "roll_mean14", "roll_mean21",
     "roll_std7",  "roll_std14",  "roll_std21"]
    if c in df.columns]

momentum_features = [c for c in
    ["momentum7", "momentum14", "momentum21",
     "accel", "zscore14", "price_range7", "volatility7"]
    if c in df.columns]

temporal_features = [c for c in
    ["month_sin", "month_cos", "month_progress",
     "is_monsoon", "is_dashain_tihar", "is_terai_harvest",
     "day_of_week", "is_weekend", "weeks_to_dashain"]
    if c in df.columns]

interaction_features = [c for c in
    ["mom7_x_monsoon", "mom7_x_festival"]
    if c in df.columns]

experiments = {
    "BASE":
        macro_features + weather_features + kg_features,

    "BASE + LAG":
        macro_features + weather_features + kg_features +
        lag_features,

    "BASE + LAG + ROLL":
        macro_features + weather_features + kg_features +
        lag_features + rolling_features,

    "BASE + LAG + ROLL + MOM":
        macro_features + weather_features + kg_features +
        lag_features + rolling_features + momentum_features,

    "BASE + LAG + ROLL + MOM + TEMP":
        macro_features + weather_features + kg_features +
        lag_features + rolling_features + momentum_features +
        temporal_features,

    "FULL MODEL":
        macro_features + weather_features + kg_features +
        lag_features + rolling_features + momentum_features +
        temporal_features + interaction_features,
}

full_features = [f for f in experiments["FULL MODEL"] if f in df.columns]

print("\nFeature group sizes:")
print(f"  Macro:       {len(macro_features)}")
print(f"  Weather:     {len(weather_features)}")
print(f"  KG:          {len(kg_features)}")
print(f"  Lag:         {len(lag_features)}")
print(f"  Rolling:     {len(rolling_features)}")
print(f"  Momentum:    {len(momentum_features)}")
print(f"  Temporal:    {len(temporal_features)}")
print(f"  Interaction: {len(interaction_features)}")
print(f"  FULL MODEL:  {len(full_features)}")


# ================================================================
# STEP 5 — MODELS
# ================================================================

def make_models():
    return {
        "Ridge": Ridge(alpha=100),

        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6,
            min_samples_leaf=30, max_features=0.4,
            random_state=RANDOM_STATE, n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=2,
            min_samples_leaf=20, subsample=0.7,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=2,
            subsample=0.6, colsample_bytree=0.6,
            min_child_weight=20, reg_alpha=5, reg_lambda=10,
            verbosity=0, random_state=RANDOM_STATE
        ),
    }


# ================================================================
# UTILITIES
# ================================================================

def evaluate(y_true_log, y_pred_log):
    """All metrics computed in NPR space after back-transforming log."""
    y_true = np.exp(np.asarray(y_true_log))
    y_pred = np.exp(np.asarray(y_pred_log))
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    return r2, rmse, mae, mape


def diebold_mariano(e1, e2):
    """
    One-sided DM test (MSE-based loss differential).
    H1: model producing e1 is more accurate than model producing e2.
    Errors must be in NPR space (already back-transformed).
    """
    d   = e1 ** 2 - e2 ** 2
    T   = len(d)
    dm  = d.mean() / (d.std(ddof=1) / np.sqrt(T))
    p   = float(scipy_stats.t.cdf(dm, df=T - 1))   # one-sided
    return round(dm, 3), round(p, 4)


def fit_predict(model_name, model, X_tr, X_te, y_tr,
                X_tr_sc, X_te_sc):
    if model_name == "Ridge":
        model.fit(X_tr_sc, y_tr)
        return model.predict(X_tr_sc), model.predict(X_te_sc)
    model.fit(X_tr, y_tr)
    return model.predict(X_tr), model.predict(X_te)


def build_dataset(feature_list, horizon):
    """Return train/test splits for a given feature list and horizon."""
    tcol = f"target_h{horizon}"
    df[tcol] = np.log(df["Average_Price"].shift(-horizon))
    feats = [f for f in feature_list if f in df.columns]
    data  = df[feats + [tcol]].dropna()
    sp    = int(len(data) * SPLIT)

    scaler     = StandardScaler()
    X_tr       = data[feats].iloc[:sp]
    X_te       = data[feats].iloc[sp:]
    y_tr       = data[tcol].iloc[:sp]
    y_te       = data[tcol].iloc[sp:]
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, feats


# ================================================================
# STEP 6 — NEPAL SEASONAL ANALYSIS
# ================================================================

print("\n" + "=" * 60)
print("STEP 6: NEPAL SEASONAL PRICE ANALYSIS")
print("=" * 60)

annual_mean = df["Average_Price"].mean()
annual_std  = df["Average_Price"].std()

print(f"\nAnnual mean price: {annual_mean:.2f} NPR/kg")
print(f"Annual std:        {annual_std:.2f} NPR/kg\n")

month_stats = df.groupby(df.index.month)["Average_Price"].agg(
    ["mean", "std", "count"]
)
print("Monthly statistics:")
for m in range(1, 13):
    row  = month_stats.loc[m]
    prem = (row["mean"] - annual_mean) / annual_mean * 100
    print(f"  Month {m:2d}: mean={row['mean']:5.1f}  "
          f"std={row['std']:4.1f}  premium={prem:+5.1f}%")

# Festival effect (Oct–Nov)
festival    = df[df.index.month.isin([10, 11])]["Average_Price"]
non_festival= df[~df.index.month.isin([10, 11])]["Average_Price"]
fest_prem   = (festival.mean() - annual_mean) / annual_mean * 100
t_fest, p_fest = scipy_stats.ttest_ind(festival, non_festival)

# Monsoon effect (Jun–Sep)
monsoon     = df[df.index.month.isin([6, 7, 8, 9])]["Average_Price"]
non_monsoon = df[~df.index.month.isin([6, 7, 8, 9])]["Average_Price"]
mon_lift    = monsoon.mean() - non_monsoon.mean()
t_mon, p_mon = scipy_stats.ttest_ind(monsoon, non_monsoon)

# Terai harvest effect (Dec–Feb)
terai       = df[df.index.month.isin([12, 1, 2])]["Average_Price"]
non_terai   = df[~df.index.month.isin([12, 1, 2])]["Average_Price"]
terai_prem  = (terai.mean() - annual_mean) / annual_mean * 100
t_terai, p_terai = scipy_stats.ttest_ind(terai, non_terai)

print(f"\nDashain/Tihar (Oct-Nov):  mean={festival.mean():.2f}  "
      f"premium={fest_prem:+.1f}%  t={t_fest:.3f}  p={p_fest:.4f}")
print(f"Monsoon (Jun-Sep):        mean={monsoon.mean():.2f}  "
      f"lift=+{mon_lift:.2f} NPR  t={t_mon:.3f}  p={p_mon:.4f}")
print(f"Terai harvest (Dec-Feb):  mean={terai.mean():.2f}  "
      f"premium={terai_prem:+.1f}%  t={t_terai:.3f}  p={p_terai:.4f}")

seasonal_df = pd.DataFrame({
    "Period":        ["Dashain/Tihar (Oct-Nov)",
                      "Monsoon (Jun-Sep)",
                      "Terai Harvest (Dec-Feb)"],
    "Mean_NPR":      [round(festival.mean(), 2),
                      round(monsoon.mean(), 2),
                      round(terai.mean(), 2)],
    "Annual_Mean":   [round(annual_mean, 2)] * 3,
    "Premium_pct":   [round(fest_prem, 1),
                      round(mon_lift / non_monsoon.mean() * 100, 1),
                      round(terai_prem, 1)],
    "t_stat":        [round(t_fest, 3), round(t_mon, 3), round(t_terai, 3)],
    "p_value":       [round(p_fest, 4), round(p_mon, 4), round(p_terai, 4)],
    "Significant":   [p_fest < 0.05,    p_mon < 0.05,    p_terai < 0.05],
})
seasonal_df.to_csv("outputs/seasonal_analysis.csv", index=False)
print("\nSeasonal analysis saved.")


# ================================================================
# STEP 7 — ABLATION STUDY (PRIMARY HORIZON = 7)
# ================================================================

print("\n" + "=" * 60)
print("STEP 7: ABLATION STUDY  —  t+7")
print("=" * 60)

ablation_results = []
predictions      = {}

for exp_name, feature_list in experiments.items():
    X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, feats = \
        build_dataset(feature_list, PRIMARY_HORIZON)

    print(f"\n{'='*50}")
    print(f"{exp_name}  |  features: {len(feats)}")

    for model_name, model in make_models().items():
        tr_pred, te_pred = fit_predict(
            model_name, model, X_tr, X_te, y_tr, X_tr_sc, X_te_sc
        )

        tr_r2, _, _, _          = evaluate(y_tr, tr_pred)
        te_r2, rmse, mae, mape  = evaluate(y_te, te_pred)

        print(f"  {model_name:20s}  "
              f"Train R²={tr_r2:.3f}  Test R²={te_r2:.3f}  "
              f"MAE={mae:.2f} NPR  MAPE={mape:.1f}%  "
              f"Gap={tr_r2-te_r2:+.3f}")

        # Store as pd.Series with datetime index for DM test
        predictions[f"{exp_name}|{model_name}"] = {
            "y_test": y_te,
            "y_pred": pd.Series(te_pred, index=y_te.index)
        }

        ablation_results.append({
            "Experiment":  exp_name,
            "Model":       model_name,
            "N_Features":  len(feats),
            "Train_R2":    round(tr_r2,        3),
            "Test_R2":     round(te_r2,        3),
            "Overfit_Gap": round(tr_r2 - te_r2,3),
            "RMSE":        round(rmse,         3),
            "MAE_NPR":     round(mae,          2),
            "MAPE_pct":    round(mape,         1),
        })

ablation_df = pd.DataFrame(ablation_results)


# ================================================================
# STEP 8 — BASELINES
# ================================================================

print("\n" + "=" * 60)
print("STEP 8: BASELINES  —  t+7")
print("=" * 60)

tcol = "target_h7"
df[tcol] = np.log(df["Average_Price"].shift(-PRIMARY_HORIZON))
base_data = df[[tcol, "Average_Price"]].dropna()

sp          = int(len(base_data) * SPLIT)
y_train_raw = base_data["Average_Price"].iloc[:sp]
y_test_raw  = base_data["Average_Price"].iloc[sp:]
y_test_log  = base_data[tcol].iloc[sp:]

# ── Persistence ──
pers_log  = np.log(base_data["Average_Price"].shift(PRIMARY_HORIZON).clip(lower=1))
valid_idx = y_test_log.index.intersection(pers_log.dropna().index)

pers_r2, pers_rmse, pers_mae, pers_mape = evaluate(
    y_test_log.loc[valid_idx], pers_log.loc[valid_idx]
)
predictions["PERSISTENCE"] = {
    "y_test": y_test_log.loc[valid_idx],
    "y_pred": pd.Series(pers_log.loc[valid_idx].values, index=valid_idx)
}
print(f"\nPersistence   R²={pers_r2:.3f}  RMSE={pers_rmse:.3f}  "
      f"MAE={pers_mae:.2f} NPR  MAPE={pers_mape:.1f}%")

# Sanity check
print("\nPersistence sanity check (10 test dates):")
sc = pd.DataFrame({
    "Actual_NPR":    np.exp(y_test_log.loc[valid_idx]).round(2),
    "Predicted_NPR": np.exp(pers_log.loc[valid_idx]).round(2)
}).head(10)
print(sc.to_string())

# ── ARIMA ──
ar_r2 = ar_rmse = ar_mae = ar_mape = np.nan
try:
    arima_fit  = ARIMA(y_train_raw, order=(5, 1, 2)).fit()
    arima_fc   = arima_fit.forecast(steps=len(y_test_raw))
    arima_log  = np.log(np.maximum(arima_fc.values, 1))
    n          = min(len(arima_log), len(y_test_log))
    ar_r2, ar_rmse, ar_mae, ar_mape = evaluate(
        y_test_log.values[:n], arima_log[:n]
    )
    print(f"\nARIMA(5,1,2)  R²={ar_r2:.3f}  RMSE={ar_rmse:.3f}  "
          f"MAE={ar_mae:.2f} NPR  MAPE={ar_mape:.1f}%")
except Exception as e:
    print(f"ARIMA failed: {e}")

# ── SARIMA ──
sa_r2 = sa_rmse = sa_mae = sa_mape = np.nan
try:
    sarima_fit = SARIMAX(y_train_raw, order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    sarima_fc  = sarima_fit.forecast(steps=len(y_test_raw))
    sarima_log = np.log(np.maximum(sarima_fc.values, 1))
    n          = min(len(sarima_log), len(y_test_log))
    sa_r2, sa_rmse, sa_mae, sa_mape = evaluate(
        y_test_log.values[:n], sarima_log[:n]
    )
    print(f"SARIMA        R²={sa_r2:.3f}  RMSE={sa_rmse:.3f}  "
          f"MAE={sa_mae:.2f} NPR  MAPE={sa_mape:.1f}%")
except Exception as e:
    print(f"SARIMA failed: {e}")

baseline_df = pd.DataFrame([
    {"Model": "Persistence", "Test_R2": round(pers_r2, 3),
     "RMSE": round(pers_rmse, 3), "MAE_NPR": round(pers_mae, 2),
     "MAPE_pct": round(pers_mape, 1)},
    {"Model": "ARIMA(5,1,2)", "Test_R2": round(ar_r2, 3),
     "RMSE": round(ar_rmse, 3), "MAE_NPR": round(ar_mae, 2),
     "MAPE_pct": round(ar_mape, 1)},
    {"Model": "SARIMA(1,1,1)(1,1,1,7)", "Test_R2": round(sa_r2, 3),
     "RMSE": round(sa_rmse, 3), "MAE_NPR": round(sa_mae, 2),
     "MAPE_pct": round(sa_mape, 1)},
])


# ================================================================
# STEP 9 — DIEBOLD-MARIANO TESTS
# ================================================================

print("\n" + "=" * 60)
print("STEP 9: DIEBOLD-MARIANO TESTS  —  all models vs Persistence")
print("=" * 60)

dm_results = []
pers       = predictions["PERSISTENCE"]

for model_name in make_models():
    key = f"FULL MODEL|{model_name}"
    if key not in predictions:
        continue

    ml     = predictions[key]
    common = ml["y_test"].index.intersection(pers["y_test"].index)

    if len(common) < 20:
        print(f"  {model_name}: too few common dates ({len(common)})")
        continue

    e_ml   = (np.exp(ml["y_pred"].loc[common])
              - np.exp(ml["y_test"].loc[common]))
    e_pers = (np.exp(pers["y_pred"].loc[common])
              - np.exp(pers["y_test"].loc[common]))

    dm, pv = diebold_mariano(e_ml.values, e_pers.values)
    sig    = "✓ sig" if pv < 0.05 else "✗ not sig"
    print(f"  {model_name:20s}  DM={dm:+.3f}  p={pv:.4f}  {sig}")

    dm_results.append({
        "Model": model_name, "DM_stat": dm,
        "p_value": pv, "Significant_5pct": pv < 0.05
    })

dm_df = pd.DataFrame(dm_results)


# ================================================================
# STEP 10 — PER-HORIZON ANALYSIS (RF FULL MODEL)
# ================================================================

print("\n" + "=" * 60)
print("STEP 10: PER-HORIZON ANALYSIS  —  RF FULL MODEL")
print("=" * 60)

horizon_results = []

for h in HORIZONS:
    X_tr, X_te, y_tr, y_te, _, _, feats = \
        build_dataset(full_features, h)

    # Persistence at this horizon
    pers_h_log = np.log(
        df["Average_Price"].shift(h)
        .reindex(X_te.index).clip(lower=1)
    )
    common_h   = y_te.index.intersection(pers_h_log.dropna().index)
    pr2, _, pmae, pmape = evaluate(
        y_te.loc[common_h], pers_h_log.loc[common_h]
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=30,
        max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    pred_h = rf.predict(X_te)

    r2, rmse, mae, mape = evaluate(y_te, pred_h)

    # DM test at this horizon
    pred_s = pd.Series(pred_h, index=y_te.index)
    e_rf_h   = (np.exp(pred_s.loc[common_h])
                - np.exp(y_te.loc[common_h]))
    e_pers_h = (np.exp(pers_h_log.loc[common_h])
                - np.exp(y_te.loc[common_h]))
    dm_h, p_h = diebold_mariano(e_rf_h.values, e_pers_h.values)
    sig_h = "✓" if p_h < 0.05 else "✗"

    print(f"  t+{h}: RF R²={r2:.3f} MAE={mae:.2f}  "
          f"| Pers R²={pr2:.3f} MAE={pmae:.2f}  "
          f"| DM p={p_h:.4f} {sig_h}")

    horizon_results.append({
        "Horizon":      f"t+{h}",
        "RF_R2":        round(r2,   3),
        "RF_MAE_NPR":   round(mae,  2),
        "RF_MAPE_pct":  round(mape, 1),
        "Pers_R2":      round(pr2,  3),
        "Pers_MAE_NPR": round(pmae, 2),
        "Delta_R2":     round(r2 - pr2, 3),
        "DM_stat":      dm_h,
        "DM_p_value":   p_h,
        "DM_sig":       p_h < 0.05,
    })

horizon_df = pd.DataFrame(horizon_results)


# ================================================================
# STEP 11 — WALK-FORWARD CROSS-VALIDATION (RF FULL MODEL)
# ================================================================

print("\n" + "=" * 60)
print("STEP 11: WALK-FORWARD CROSS-VALIDATION  —  RF FULL MODEL")
print("=" * 60)

tcol = "target_h7"
df[tcol] = np.log(df["Average_Price"].shift(-PRIMARY_HORIZON))
data_wf   = df[full_features + [tcol]].dropna()

X_wf = data_wf[full_features].values
y_wf = data_wf[tcol].values

tscv = TimeSeriesSplit(n_splits=5, gap=PRIMARY_HORIZON)

wf_r2, wf_mae, wf_mape = [], [], []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_wf)):
    rf_wf = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=30,
        max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_wf.fit(X_wf[tr_idx], y_wf[tr_idx])
    pred_wf = rf_wf.predict(X_wf[te_idx])

    r2_f, _, mae_f, mape_f = evaluate(y_wf[te_idx], pred_wf)
    wf_r2.append(r2_f)
    wf_mae.append(mae_f)
    wf_mape.append(mape_f)

    print(f"  Fold {fold+1} (n_test={len(te_idx)}):  "
          f"R²={r2_f:.3f}  MAE={mae_f:.2f} NPR  MAPE={mape_f:.1f}%")

print(f"\n  Cross-validation summary:")
print(f"  Mean R²   = {np.mean(wf_r2):.3f}  ±  {np.std(wf_r2):.3f}")
print(f"  Mean MAE  = {np.mean(wf_mae):.2f}  ±  {np.std(wf_mae):.2f} NPR")
print(f"  Mean MAPE = {np.mean(wf_mape):.1f}  ±  {np.std(wf_mape):.1f}%")

wf_df = pd.DataFrame({
    "Fold":     [f"Fold {i+1}" for i in range(5)] + ["Mean", "Std"],
    "R2":       wf_r2 + [round(np.mean(wf_r2), 3), round(np.std(wf_r2), 3)],
    "MAE_NPR":  wf_mae + [round(np.mean(wf_mae), 2), round(np.std(wf_mae), 2)],
    "MAPE_pct": wf_mape + [round(np.mean(wf_mape), 1), round(np.std(wf_mape), 1)],
})


# ================================================================
# STEP 12 — SPIKE DETECTION
# ================================================================

print("\n" + "=" * 60)
print("STEP 12: PRICE SHOCK DETECTION  —  FULL MODEL  t+7")
print("=" * 60)

X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, feats = \
    build_dataset(full_features, PRIMARY_HORIZON)

train_prices_sp = np.exp(y_tr)
spike_thresh    = (train_prices_sp.mean()
                   + SPIKE_SIGMA * train_prices_sp.std())
y_te_price      = np.exp(y_te)
true_binary     = (y_te_price > spike_thresh).astype(int)
n_shocks        = true_binary.sum()

print(f"Spike threshold ({SPIKE_SIGMA}σ above training mean): "
      f"{spike_thresh:.2f} NPR/kg")
print(f"Shock events in test set: {n_shocks} / {len(true_binary)} days "
      f"({100*true_binary.mean():.1f}%)")

if n_shocks < 30:
    print("NOTE: Fewer than 30 shock events — report as limitation.")

spike_results = []

for model_name, model in make_models().items():
    _, te_pred = fit_predict(
        model_name, model, X_tr, X_te, y_tr, X_tr_sc, X_te_sc
    )
    pred_price  = np.exp(te_pred)
    pred_binary = (pred_price > spike_thresh).astype(int)

    rpt = classification_report(
        true_binary, pred_binary, output_dict=True, zero_division=0
    )
    auc = (roc_auc_score(true_binary, pred_price)
           if n_shocks > 0 else np.nan)

    prec = rpt.get("1", {}).get("precision", 0)
    rec  = rpt.get("1", {}).get("recall",    0)
    f1   = rpt.get("1", {}).get("f1-score",  0)

    print(f"  {model_name:20s}  "
          f"Prec={prec:.3f}  Rec={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")

    spike_results.append({
        "Model":     model_name,
        "Precision": round(prec, 3),
        "Recall":    round(rec,  3),
        "F1":        round(f1,   3),
        "ROC_AUC":   round(auc,  3) if not np.isnan(auc) else np.nan,
    })

# Persistence spike detection
pers_price_te = df["Average_Price"].shift(PRIMARY_HORIZON).reindex(X_te.index)
pers_binary   = (pers_price_te > spike_thresh).astype(int)
aligned       = pers_binary.dropna().index.intersection(true_binary.index)

if len(aligned) > 0:
    rpt_p = classification_report(
        true_binary.loc[aligned], pers_binary.loc[aligned],
        output_dict=True, zero_division=0
    )
    auc_p = (roc_auc_score(true_binary.loc[aligned],
                            pers_price_te.loc[aligned])
             if true_binary.loc[aligned].sum() > 0 else np.nan)
    pp = rpt_p.get("1", {}).get("precision", 0)
    rp = rpt_p.get("1", {}).get("recall",    0)
    fp = rpt_p.get("1", {}).get("f1-score",  0)
    print(f"  {'Persistence':20s}  "
          f"Prec={pp:.3f}  Rec={rp:.3f}  "
          f"F1={fp:.3f}  AUC={auc_p:.3f}")
    spike_results.append({
        "Model": "Persistence",
        "Precision": round(pp, 3), "Recall": round(rp, 3),
        "F1": round(fp, 3),
        "ROC_AUC": round(auc_p, 3) if not np.isnan(auc_p) else np.nan,
    })

spike_df = pd.DataFrame(spike_results)


# ================================================================
# STEP 13 — KG EVENT-PERIOD ANALYSIS
# ================================================================

print("\n" + "=" * 60)
print("STEP 13: KG EVENT-PERIOD ANALYSIS  —  RF  t+7")
print("=" * 60)

no_kg_features = [f for f in full_features if f not in kg_features]
kg_analysis    = []

for feat_set, label in [
    (full_features,  "WITH KG"),
    (no_kg_features, "WITHOUT KG"),
]:
    X_tr_k, X_te_k, y_tr_k, y_te_k, _, _, fts = \
        build_dataset(feat_set, PRIMARY_HORIZON)

    disrupt_te = df["disruption_flag"].reindex(X_te_k.index).fillna(0)

    rf_k = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=30,
        max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_k.fit(X_tr_k, y_tr_k)
    pred_k = pd.Series(rf_k.predict(X_te_k), index=X_te_k.index)

    print(f"\n  {label}  ({len(fts)} features)")

    for subset_name, mask in [
        ("All test dates",        X_te_k.index),
        ("Disruption dates",      X_te_k.index[disrupt_te == 1]),
        ("Non-disruption dates",  X_te_k.index[disrupt_te == 0]),
    ]:
        if len(mask) < 10:
            print(f"    {subset_name:25s}: too few samples ({len(mask)})")
            continue

        r2_k, _, mae_k, mape_k = evaluate(
            y_te_k.loc[mask], pred_k.loc[mask]
        )
        print(f"    {subset_name:25s}  n={len(mask):4d}  "
              f"R²={r2_k:.3f}  MAE={mae_k:.2f} NPR  MAPE={mape_k:.1f}%")

        kg_analysis.append({
            "Feature_Set": label, "Subset": subset_name,
            "N": len(mask), "R2": round(r2_k, 3),
            "MAE_NPR": round(mae_k, 2), "MAPE_pct": round(mape_k, 1),
        })

kg_df = pd.DataFrame(kg_analysis)


# ================================================================
# STEP 14 — FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 60)
print("STEP 14: FEATURE IMPORTANCE  —  RF FULL MODEL")
print("=" * 60)

X_tr_fi, _, y_tr_fi, _, _, _, feats_fi = \
    build_dataset(full_features, PRIMARY_HORIZON)

rf_fi = RandomForestRegressor(
    n_estimators=400, max_depth=6, min_samples_leaf=30,
    max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1
)
rf_fi.fit(X_tr_fi, y_tr_fi)

imp_df = pd.DataFrame({
    "Feature":    feats_fi,
    "Importance": rf_fi.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(imp_df.head(20).to_string(index=False))


# ================================================================
# STEP 15 — SUMMARY PRINT
# ================================================================

print("\n" + "=" * 60)
print("SUMMARY: ABLATION  —  Random Forest")
print("=" * 60)
rf_abl = ablation_df[ablation_df["Model"] == "RandomForest"][
    ["Experiment", "N_Features", "Train_R2",
     "Test_R2", "Overfit_Gap", "MAE_NPR", "MAPE_pct"]
]
print(rf_abl.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: ALL MODELS  —  FULL MODEL  t+7")
print("=" * 60)
full_cmp = ablation_df[ablation_df["Experiment"] == "FULL MODEL"][
    ["Model", "Train_R2", "Test_R2",
     "Overfit_Gap", "RMSE", "MAE_NPR", "MAPE_pct"]
]
print(full_cmp.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: PER-HORIZON")
print("=" * 60)
print(horizon_df.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: WALK-FORWARD CROSS-VALIDATION")
print("=" * 60)
print(wf_df.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: SPIKE DETECTION")
print("=" * 60)
print(spike_df.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: DIEBOLD-MARIANO")
print("=" * 60)
print(dm_df.to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY: KG EVENT-PERIOD ANALYSIS")
print("=" * 60)
print(kg_df.to_string(index=False))


# ================================================================
# STEP 16 — SAVE ALL OUTPUTS
# ================================================================

ablation_df.to_csv("outputs/ablation_results.csv",    index=False)
baseline_df.to_csv("outputs/baseline_results.csv",    index=False)
horizon_df.to_csv("outputs/horizon_results.csv",      index=False)
wf_df.to_csv("outputs/walkforward_cv.csv",            index=False)
spike_df.to_csv("outputs/spike_detection.csv",        index=False)
dm_df.to_csv("outputs/diebold_mariano.csv",           index=False)
kg_df.to_csv("outputs/kg_event_analysis.csv",         index=False)
imp_df.to_csv("outputs/feature_importance.csv",       index=False)
seasonal_df.to_csv("outputs/seasonal_analysis.csv",   index=False)

print("\n✓ All CSVs saved to outputs/")


# ================================================================
# STEP 17 — FIGURES
# ================================================================

# ── Figure 1: 4-panel model evaluation ──
X_tr_pl, X_te_pl, y_tr_pl, y_te_pl, _, _, feats_pl = \
    build_dataset(full_features, PRIMARY_HORIZON)

rf_pl = RandomForestRegressor(
    n_estimators=400, max_depth=6, min_samples_leaf=30,
    max_features=0.4, random_state=RANDOM_STATE, n_jobs=-1
)
rf_pl.fit(X_tr_pl, y_tr_pl)
pred_pl     = rf_pl.predict(X_te_pl)
actual_pl   = np.exp(y_te_pl.values)
pred_npr_pl = np.exp(pred_pl)
test_dates  = y_te_pl.index

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Random Forest FULL MODEL — t+7 Evaluation",
    fontsize=14, fontweight="bold"
)

# (a) Time series
ax = axes[0, 0]
ax.plot(test_dates, actual_pl,   label="Actual",
        linewidth=1.2, color="#2E4057")
ax.plot(test_dates, pred_npr_pl, label="Predicted",
        linewidth=1.0, color="#E84855", alpha=0.8, linestyle="--")
ax.set_title("(a) Actual vs Predicted — Test Set")
ax.set_ylabel("Price (NPR/kg)")
ax.legend(); ax.grid(alpha=0.3)

# (b) Residuals
residuals = actual_pl - pred_npr_pl
ax = axes[0, 1]
ax.scatter(pred_npr_pl, residuals, alpha=0.4, s=10, color="#2E4057")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_title("(b) Residual Plot")
ax.set_xlabel("Predicted (NPR/kg)")
ax.set_ylabel("Residual (NPR/kg)")
ax.grid(alpha=0.3)

# (c) R² by horizon
ax = axes[1, 0]
ax.plot(
    [r["Horizon"] for r in horizon_results],
    [r["RF_R2"]   for r in horizon_results],
    marker="o", color="#2E4057", linewidth=2, label="Random Forest"
)
ax.plot(
    [r["Horizon"] for r in horizon_results],
    [r["Pers_R2"] for r in horizon_results],
    marker="s", color="#E84855", linewidth=2,
    linestyle="--", label="Persistence"
)
ax.set_title("(c) Test R² by Forecast Horizon")
ax.set_ylabel("Test R²"); ax.set_xlabel("Horizon")
ax.legend(); ax.grid(alpha=0.3)

# (d) Feature importance top 15
ax = axes[1, 1]
top15 = imp_df.head(15)
ax.barh(top15["Feature"][::-1], top15["Importance"][::-1],
        color="#2E4057")
ax.set_title("(d) Top 15 Feature Importances (RF)")
ax.set_xlabel("Mean Decrease Impurity")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("outputs/fig1_model_evaluation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved: outputs/fig1_model_evaluation.png")

# ── Figure 2: Ablation bar chart ──
rf_abl_plot = ablation_df[ablation_df["Model"] == "RandomForest"].copy()

fig, ax = plt.subplots(figsize=(11, 5))
colors  = ["#E84855" if r < 0 else "#2E4057"
           for r in rf_abl_plot["Test_R2"]]
bars    = ax.bar(range(len(rf_abl_plot)),
                 rf_abl_plot["Test_R2"], color=colors)
ax.axhline(pers_r2, color="orange", linestyle="--",
           linewidth=1.5,
           label=f"Persistence (R²={pers_r2:.3f})")
ax.set_xticks(range(len(rf_abl_plot)))
ax.set_xticklabels(rf_abl_plot["Experiment"],
                   rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Test R²")
ax.set_title(
    "Ablation Study — Random Forest Test R² by "
    "Feature Configuration (t+7)"
)
ax.legend(); ax.grid(alpha=0.3, axis="y")

for bar, val in zip(bars, rf_abl_plot["Test_R2"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/fig2_ablation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved: outputs/fig2_ablation.png")

# ── Figure 3: Seasonal price patterns ──
monthly_mean = df.groupby(df.index.month)["Average_Price"].mean()
monthly_std  = df.groupby(df.index.month)["Average_Price"].std()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
colors_m     = ["#E84855" if m in [6,7,8,9,10,11]
                else "#2E4057" for m in range(1,13)]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(12), monthly_mean.values,
              color=colors_m, alpha=0.85,
              yerr=monthly_std.values, capsize=4)
ax.axhline(annual_mean, color="gray", linestyle="--",
           linewidth=1.2, label=f"Annual mean ({annual_mean:.1f} NPR)")
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_ylabel("Average Price (NPR/kg)")
ax.set_title("Average Tomato Price by Month — Seasonal Cycles\n"
             "(Red = Monsoon + Festival period)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("outputs/fig3_seasonal.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved: outputs/fig3_seasonal.png")

print("\n" + "=" * 60)
print("ALL DONE — check outputs/ folder for all results and figures")
print("=" * 60)
