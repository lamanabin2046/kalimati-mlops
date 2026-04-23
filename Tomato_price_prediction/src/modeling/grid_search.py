"""
random_forest_gridsearch.py
----------------------------
Grid search hyperparameter tuning for Random Forest
on the tomato price 7-day-ahead forecasting task.

Output:
  - Best parameters
  - Best Train R² / Test R² / Gap
  - Top 10 configs ranked by Test R²
  - Feature importance chart
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# CONFIGURATION
# ============================================================

HORIZON    = 7       # days ahead to forecast
SPLIT      = 0.80    # 80% train / 20% test
MEAN_PRICE = 52.86   # average tomato price in NPR


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/tomato_base_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date").asfreq("D")
df = df.dropna(subset=["Average_Price"]).ffill().bfill()

print(f"Dataset loaded: {df.shape[0]} rows")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

m = df.index.month
p = df["Average_Price"]

# Lag features
for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
    df[f"lag{lag}"] = p.shift(lag)

# Rolling statistics
for w in [7, 14, 21, 30]:
    df[f"roll_mean{w}"] = p.shift(1).rolling(w).mean()
for w in [7, 14]:
    df[f"roll_std{w}"] = p.shift(1).rolling(w).std()

# Momentum
df["mom7"]  = df["lag1"] - df["lag7"]
df["mom14"] = df["lag1"] - df["lag14"]
df["mom21"] = df["lag1"] - df["lag21"]
df["accel"] = df["mom7"] - (df["lag7"] - df["lag14"])

# Price position
df["zscore14"]     = (df["lag1"] - df["roll_mean14"]) / (df["roll_std14"].replace(0, 1))
df["price_range7"] = p.shift(1).rolling(7).max() - p.shift(1).rolling(7).min()

# Nepal seasonal features
df["month_sin"]        = np.sin(2 * np.pi * m / 12)
df["month_cos"]        = np.cos(2 * np.pi * m / 12)
df["is_monsoon"]       = m.isin([6, 7, 8, 9]).astype(int)
df["is_monsoon_peak"]  = m.isin([7, 8]).astype(int)
df["is_dashain_tihar"] = m.isin([10, 11]).astype(int)
df["is_terai_harvest"] = m.isin([11, 12, 1, 2]).astype(int)
df["is_offseason"]     = m.isin([3, 4]).astype(int)
df["is_pre_festival"]  = (m == 9).astype(int)
df["is_winter"]        = m.isin([12, 1, 2]).astype(int)
df["is_post_monsoon"]  = m.isin([10, 11]).astype(int)

# Interaction features
df["mom7_x_monsoon"]  = df["mom7"] * df["is_monsoon"]
df["mom7_x_festival"] = df["mom7"] * df["is_dashain_tihar"]
df["mom7_x_harvest"]  = df["mom7"] * df["is_terai_harvest"]

# Weeks to Dashain (continuous festival proximity)
doy = df.index.dayofyear
df["weeks_to_dashain"] = np.minimum(
    np.abs(doy - 280), 365 - np.abs(doy - 280)
) / 7.0

# Target: log price 7 days ahead
df["target"] = np.log(p.shift(-HORIZON))
df = df.dropna()

print(f"After feature engineering: {df.shape[0]} rows")


# ============================================================
# FEATURE SELECTION
# ============================================================

FEATURES = [
    # Price lags
    "lag1", "lag2", "lag3", "lag5", "lag7", "lag14", "lag21", "lag28",
    # Rolling stats
    "roll_mean7", "roll_mean14", "roll_mean21", "roll_mean30",
    "roll_std7", "roll_std14",
    # Momentum
    "mom7", "mom14", "mom21", "accel",
    "zscore14", "price_range7",
    # Nepal seasonal
    "month_sin", "month_cos",
    "is_monsoon", "is_monsoon_peak", "is_dashain_tihar",
    "is_terai_harvest", "is_offseason", "is_pre_festival",
    "is_winter", "is_post_monsoon",
    # Interactions
    "mom7_x_monsoon", "mom7_x_festival", "mom7_x_harvest",
    "weeks_to_dashain",
    # Macro
    "Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation",
    # Weather
    "Sarlahi_Rainfall_MM", "Dhading_Rainfall_MM",
    "Kathmandu_Rainfall_MM", "Market_Risk",
]

FEATURES = [f for f in FEATURES if f in df.columns]
print(f"Features used: {len(FEATURES)}")

X = df[FEATURES]
y = df["target"]


# ============================================================
# TRAIN / TEST SPLIT (chronological)
# ============================================================

split_index = int(len(df) * SPLIT)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_test  = y.iloc[split_index:]

print(f"\nTrain: {len(X_train)} rows  "
      f"({X_train.index.min().date()} → {X_train.index.max().date()})")
print(f"Test : {len(X_test)} rows  "
      f"({X_test.index.min().date()} → {X_test.index.max().date()})")


# ============================================================
# GRID SEARCH PARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators":     [200, 300, 400, 500],
    "max_depth":        [3, 4, 5, 6],
    "min_samples_leaf": [20, 30, 40, 50],
    "max_features":     [0.3, 0.4, 0.5, "sqrt"],
}

# TimeSeriesSplit — prevents future data leaking into training folds
tscv = TimeSeriesSplit(n_splits=5)

print(f"\n{'='*58}")
print(f"  GRID SEARCH — Random Forest")
print(f"  Parameter combinations: "
      f"{len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")
print(f"  Cross-validation: TimeSeriesSplit(n_splits=5)")
print(f"  Scoring: R²")
print(f"  This may take 5–10 minutes...")
print(f"{'='*58}\n")


grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)


# ============================================================
# BEST MODEL EVALUATION
# ============================================================

best_rf   = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate on train and test
train_pred = best_rf.predict(X_train)
test_pred  = best_rf.predict(X_test)

train_r2  = r2_score(y_train, train_pred)
test_r2   = r2_score(y_test,  test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae  = mean_absolute_error(y_test, test_pred)
test_npr  = MEAN_PRICE * (np.exp(test_mae) - 1)
gap       = train_r2 - test_r2

print(f"\n{'='*58}")
print(f"  BEST PARAMETERS FOUND")
print(f"{'='*58}")
for param, value in best_params.items():
    print(f"  {param:<22}: {value}")

print(f"\n{'='*58}")
print(f"  BEST MODEL PERFORMANCE  (t+{HORIZON} forecast)")
print(f"{'='*58}")
print(f"  CV R² (best fold)  : {grid_search.best_score_:.4f}")
print(f"  Train R²           : {train_r2:.4f}")
print(f"  Test R²            : {test_r2:.4f}")
print(f"  Gap (train - test) : {gap:.4f}  "
      f"({'✅ GOOD' if gap < 0.10 else '⚠ ACCEPTABLE' if gap < 0.15 else '❌ HIGH'})")
print(f"  RMSE               : {test_rmse:.4f}")
print(f"  MAE                : {test_mae:.4f}  ≈ ±{test_npr:.2f} NPR/kg")
print(f"{'='*58}")


# ============================================================
# TOP 10 CONFIGURATIONS
# ============================================================

cv_results = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score descending
cv_results = cv_results.sort_values("mean_test_score", ascending=False)

# Evaluate every config on the held-out test set
print(f"\n{'='*70}")
print(f"  TOP 10 CONFIGURATIONS — ranked by CV R²")
print(f"{'='*70}")
print(f"  {'Rank':<5} {'n_est':>5} {'depth':>6} {'leaf':>5} "
      f"{'feat':>6} {'CV R²':>8} {'Test R²':>8} {'Gap':>7}")
print(f"  {'─'*60}")

top10 = cv_results.head(10)

for rank, (_, row) in enumerate(top10.iterrows(), 1):
    p = row["params"]
    # Re-train this config and test it
    rf_tmp = RandomForestRegressor(
        n_estimators    = p["n_estimators"],
        max_depth       = p["max_depth"],
        min_samples_leaf= p["min_samples_leaf"],
        max_features    = p["max_features"],
        random_state    = 42,
        n_jobs          = -1,
    )
    rf_tmp.fit(X_train, y_train)
    tr2 = round(r2_score(y_train, rf_tmp.predict(X_train)), 3)
    te2 = round(r2_score(y_test,  rf_tmp.predict(X_test)),  3)
    gap_r = round(tr2 - te2, 3)
    cv_r2 = round(row["mean_test_score"], 3)

    flag = " ← BEST" if rank == 1 else ""
    print(f"  {rank:<5} {p['n_estimators']:>5} {p['max_depth']:>6} "
          f"{p['min_samples_leaf']:>5} {str(p['max_features']):>6} "
          f"{cv_r2:>8} {te2:>8} {gap_r:>7}{flag}")

print(f"{'='*70}")


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

importances = best_rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

# Colour seasonal vs non-seasonal features
seasonal_keywords = [
    "monsoon", "dashain", "festival", "harvest", "offseason",
    "winter", "holi", "month_sin", "month_cos", "weeks_to_dashain",
    "post_monsoon", "pre_festival", "mom7_x",
]
feat_df["is_seasonal"] = feat_df["Feature"].apply(
    lambda x: any(k in x for k in seasonal_keywords)
)

top20 = feat_df.tail(20)
colors = ["#ED7D31" if s else "#2E75B6" for s in top20["is_seasonal"]]

plt.figure(figsize=(10, 7))
plt.barh(top20["Feature"], top20["Importance"], color=colors, alpha=0.85)
plt.xlabel("Importance Score", fontsize=11)
plt.title(
    f"Top 20 Feature Importances — Random Forest (Best Config)\n"
    f"Orange = seasonal  ·  Blue = price/macro/weather",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=120)
plt.show()
print("Saved: rf_feature_importance.png")


# ============================================================
# ACTUAL vs PREDICTED PLOT
# ============================================================

actual_npr = np.exp(y_test.values)
pred_npr   = np.exp(test_pred)
residuals  = actual_npr - pred_npr

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(actual_npr, label="Actual price (NPR)",
             color="#C00000", linewidth=1.2)
axes[0].plot(pred_npr, label=f"Predicted (best RF)",
             color="#70AD47", linewidth=1.2, linestyle="--", alpha=0.9)
axes[0].set_ylabel("Average Price (NPR/kg)")
axes[0].set_title(
    f"Actual vs Predicted — Best Random Forest  "
    f"(t+{HORIZON}, Test R²={test_r2:.3f})",
    fontweight="bold"
)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].bar(range(len(residuals)), residuals,
            color="#ED7D31", alpha=0.7, width=1.0)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Residual (NPR/kg)")
axes[1].set_xlabel("Test observations (days)")
axes[1].set_title(
    f"Residuals  mean={residuals.mean():.2f}  std={residuals.std():.2f} NPR/kg",
    fontweight="bold"
)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rf_actual_vs_predicted.png", dpi=120)
plt.show()
print("Saved: rf_actual_vs_predicted.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*58}")
print(f"  FINAL SUMMARY")
print(f"{'='*58}")
print(f"  Model        : Random Forest (grid search tuned)")
print(f"  Best params  : {best_params}")
print(f"  Train R²     : {train_r2:.4f}")
print(f"  Test R²      : {test_r2:.4f}")
print(f"  Gap          : {gap:.4f}")
print(f"  MAE          : ≈ ±{test_npr:.2f} NPR/kg")
print(f"  Horizon      : t+{HORIZON} (7-day-ahead)")
print(f"  Features     : {len(FEATURES)}")
print(f"{'='*58}")
