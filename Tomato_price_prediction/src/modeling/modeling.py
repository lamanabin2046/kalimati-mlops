import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

HORIZON = 7
SPLIT = 0.80
MEAN_PRICE = 52.86


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/tomato_base_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date").asfreq("D")

df = df.dropna(subset=["Average_Price"]).ffill().bfill()

print("Dataset shape:", df.shape)


# ============================================================
# WEATHER AGGREGATION
# ============================================================

temp_cols = [c for c in df.columns if "Temperature" in c]
rain_cols = [c for c in df.columns if "Rainfall_MM" in c]

if len(temp_cols) > 0:
    df["Temp_mean"] = df[temp_cols].mean(axis=1)
    df["Temp_std"] = df[temp_cols].std(axis=1)

if len(rain_cols) > 0:
    df["Rainfall_mean"] = df[rain_cols].mean(axis=1)
    df["Rainfall_max"] = df[rain_cols].max(axis=1)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

month = df.index.month
day = df.index.day

for lag in [1,2,3,5,7,14,21,28]:
    df[f"lag{lag}"] = df["Average_Price"].shift(lag)


df["roll_mean7"] = df["Average_Price"].shift(1).rolling(7).mean()
df["roll_mean14"] = df["Average_Price"].shift(1).rolling(14).mean()
df["roll_mean21"] = df["Average_Price"].shift(1).rolling(21).mean()

df["roll_std7"] = df["Average_Price"].shift(1).rolling(7).std()
df["roll_std14"] = df["Average_Price"].shift(1).rolling(14).std()


df["momentum7"] = df["lag1"] - df["lag7"]
df["momentum14"] = df["lag1"] - df["lag14"]
df["momentum21"] = df["lag1"] - df["lag21"]

df["accel"] = df["momentum7"] - (df["lag7"] - df["lag14"])

df["zscore14"] = (df["lag1"] - df["roll_mean14"]) / df["roll_std14"].replace(0,1)

df["price_range7"] = (
    df["Average_Price"].shift(1).rolling(7).max() -
    df["Average_Price"].shift(1).rolling(7).min()
)

df["volatility7"] = df["roll_std7"] / (df["roll_mean7"] + 1e-6)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

df["month_sin"] = np.sin(2*np.pi*month/12)
df["month_cos"] = np.cos(2*np.pi*month/12)

df["month_progress"] = day / 30

df["is_monsoon"] = month.isin([6,7,8,9]).astype(int)
df["is_dashain_tihar"] = month.isin([10,11]).astype(int)

df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)


df["mom7_x_monsoon"] = df["momentum7"] * df["is_monsoon"]
df["mom7_x_festival"] = df["momentum7"] * df["is_dashain_tihar"]


day_of_year = df.index.dayofyear
dist_dashain = np.minimum(np.abs(day_of_year-280),365-np.abs(day_of_year-280))
df["weeks_to_dashain"] = dist_dashain/7


# ============================================================
# TARGET
# ============================================================

df["target"] = np.log(df["Average_Price"].shift(-HORIZON))

df = df.dropna()

print("After feature engineering:", df.shape)


# ============================================================
# FEATURE GROUPS
# ============================================================

macro_features = ["Supply_Volume","USD_TO_NPR","Diesel","Inflation"]
weather_features = ["Temp_mean","Temp_std","Rainfall_mean","Rainfall_max"]
kg_features = ["Market_Risk"]

lag_features = ["lag1","lag2","lag3","lag5","lag7","lag14","lag21"]

rolling_features = ["roll_mean7","roll_mean14","roll_mean21","roll_std7","roll_std14"]

momentum_features = [
"momentum7","momentum14","momentum21",
"accel","zscore14","price_range7","volatility7"
]

temporal_features = [
"month_sin","month_cos","month_progress",
"is_monsoon","is_dashain_tihar",
"day_of_week","is_weekend","weeks_to_dashain"
]

interaction_features = [
"mom7_x_monsoon","mom7_x_festival"
]


# ============================================================
# EXPERIMENT SETUPS
# ============================================================

experiments = {

"BASE":
macro_features + weather_features + kg_features,

"BASE + LAG":
macro_features + weather_features + kg_features + lag_features,

"BASE + LAG + ROLL":
macro_features + weather_features + kg_features + lag_features + rolling_features,

"BASE + LAG + ROLL + MOM":
macro_features + weather_features + kg_features + lag_features + rolling_features + momentum_features,

"BASE + LAG + ROLL + MOM + TEMP":
macro_features + weather_features + kg_features + lag_features + rolling_features + momentum_features + temporal_features,

"FULL MODEL":
macro_features + weather_features + kg_features + lag_features + rolling_features + momentum_features + temporal_features + interaction_features

}


# ============================================================
# MODELS
# ============================================================

models = {

"Ridge":
Ridge(alpha=100),

"RandomForest":
RandomForestRegressor(
n_estimators=400,
max_depth=6,
min_samples_leaf=30,
max_features=0.4,
random_state=42
),

"GradientBoosting":
GradientBoostingRegressor(
n_estimators=500,
learning_rate=0.01,
max_depth=2,
min_samples_leaf=20,
subsample=0.7
),

"XGBoost":
XGBRegressor(
n_estimators=500,
learning_rate=0.01,
max_depth=2,
subsample=0.6,
colsample_bytree=0.6,
min_child_weight=20,
reg_alpha=5,
reg_lambda=10,
verbosity=0
)

}


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true, y_pred):

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    npr  = MEAN_PRICE * (np.exp(mae) - 1)

    return r2, rmse, mae, npr


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

for exp_name, features in experiments.items():

    features = [f for f in features if f in df.columns]

    print("\n===============================")
    print("Experiment:", exp_name)
    print("Features:", len(features))

    X = df[features]
    y = df["target"]

    split_index = int(len(df) * SPLIT)

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    for name, model in models.items():

        if name == "Ridge":
            model.fit(X_train_sc, y_train)
            train_pred = model.predict(X_train_sc)   # ← added
            test_pred  = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)      # ← added
            test_pred  = model.predict(X_test)

        train_r2, _, _, _          = evaluate(y_train, train_pred)   # ← added
        test_r2, rmse, mae, npr    = evaluate(y_test,  test_pred)

        overfit_gap = train_r2 - test_r2                             # ← added

        print(f"{name:20s}  Train R2: {train_r2:.3f}  |  Test R2: {test_r2:.3f}  |  Gap: {overfit_gap:+.3f}")

        results.append({
            "Experiment" : exp_name,
            "Model"      : name,
            "Train_R2"   : train_r2,     # ← added
            "Test_R2"    : test_r2,
            "Overfit_Gap": overfit_gap,  # ← added
            "RMSE"       : rmse,
            "MAE"        : mae,
            "Error_NPR"  : npr
        })


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\nFINAL RESULTS (sorted by Test R2)")
print(
    results_df[["Experiment","Model","Train_R2","Test_R2","Overfit_Gap","RMSE","MAE","Error_NPR"]]
    .sort_values("Test_R2", ascending=False)
    .to_string(index=False)
)