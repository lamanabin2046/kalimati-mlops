import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

HORIZON    = 7
SPLIT      = 0.80
MEAN_PRICE = 52.86
TOP_N      = 20          # how many features to show in the chart


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

if temp_cols:
    df["Temp_mean"] = df[temp_cols].mean(axis=1)
    df["Temp_std"]  = df[temp_cols].std(axis=1)

if rain_cols:
    df["Rainfall_mean"] = df[rain_cols].mean(axis=1)
    df["Rainfall_max"]  = df[rain_cols].max(axis=1)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

month      = df.index.month
day        = df.index.day
day_of_year = df.index.dayofyear

# --- Lag features ---
for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
    df[f"lag{lag}"] = df["Average_Price"].shift(lag)

# --- Rolling features ---
df["roll_mean7"]  = df["Average_Price"].shift(1).rolling(7).mean()
df["roll_mean14"] = df["Average_Price"].shift(1).rolling(14).mean()
df["roll_mean21"] = df["Average_Price"].shift(1).rolling(21).mean()
df["roll_std7"]   = df["Average_Price"].shift(1).rolling(7).std()
df["roll_std14"]  = df["Average_Price"].shift(1).rolling(14).std()

# --- Momentum features ---
df["momentum7"]  = df["lag1"] - df["lag7"]
df["momentum14"] = df["lag1"] - df["lag14"]
df["momentum21"] = df["lag1"] - df["lag21"]
df["accel"]      = df["momentum7"] - (df["lag7"] - df["lag14"])
df["zscore14"]   = (df["lag1"] - df["roll_mean14"]) / df["roll_std14"].replace(0, 1)
df["price_range7"] = (
    df["Average_Price"].shift(1).rolling(7).max() -
    df["Average_Price"].shift(1).rolling(7).min()
)
df["volatility7"] = df["roll_std7"] / (df["roll_mean7"] + 1e-6)

# --- Temporal features ---
df["month_sin"]       = np.sin(2 * np.pi * month / 12)
df["month_cos"]       = np.cos(2 * np.pi * month / 12)
df["month_progress"]  = day / 30
df["is_monsoon"]      = month.isin([6, 7, 8, 9]).astype(int)
df["is_dashain_tihar"]= month.isin([10, 11]).astype(int)
df["day_of_week"]     = df.index.dayofweek
df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)

dist_dashain          = np.minimum(np.abs(day_of_year - 280), 365 - np.abs(day_of_year - 280))
df["weeks_to_dashain"]= dist_dashain / 7

# --- Interaction features ---
df["mom7_x_monsoon"]  = df["momentum7"] * df["is_monsoon"]
df["mom7_x_festival"] = df["momentum7"] * df["is_dashain_tihar"]


# ============================================================
# TARGET
# ============================================================

df["target"] = np.log(df["Average_Price"].shift(-HORIZON))
df = df.dropna()

print("After feature engineering:", df.shape)


# ============================================================
# FEATURE GROUPS
# ============================================================

macro_features = ["Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation"]
weather_features = ["Temp_mean", "Temp_std", "Rainfall_mean", "Rainfall_max"]
kg_features = ["Market_Risk"]

lag_features = ["lag1", "lag2", "lag3", "lag5", "lag7", "lag14", "lag21"]

rolling_features = [
    "roll_mean7", "roll_mean14", "roll_mean21",
    "roll_std7", "roll_std14"
]

momentum_features = [
    "momentum7", "momentum14", "momentum21",
    "accel", "zscore14", "price_range7", "volatility7"
]

temporal_features = [
    "month_sin", "month_cos", "month_progress",
    "is_monsoon", "is_dashain_tihar",
    "day_of_week", "is_weekend", "weeks_to_dashain"
]

interaction_features = ["mom7_x_monsoon", "mom7_x_festival"]

# All features combined (FULL MODEL)
all_features = (
    macro_features + weather_features + kg_features +
    lag_features + rolling_features + momentum_features +
    temporal_features + interaction_features
)

# Keep only columns that actually exist in the dataset
all_features = [f for f in all_features if f in df.columns]

print(f"\nFULL MODEL feature count: {len(all_features)}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df[all_features]
y = df["target"]

split_index = int(len(df) * SPLIT)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")


# ============================================================
# FIT BEST MODEL (Random Forest — FULL MODEL config)
# ============================================================

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=30,
    max_features=0.4,
    random_state=42
)

model.fit(X_train, y_train)

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2  = r2_score(y_test,  model.predict(X_test))
gap      = train_r2 - test_r2

print(f"\nTrain R²: {train_r2:.3f}  |  Test R²: {test_r2:.3f}  |  Overfit gap: {gap:+.3f}")


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

importances = pd.DataFrame({
    "Feature":    all_features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n===== FEATURE IMPORTANCE (Top 20) =====")
print(importances.head(20).to_string(index=False))
print()

importances.to_csv("feature_importance_full.csv", index=False)
print("[SAVED] feature_importance_full.csv")


# ============================================================
# COLOUR MAP BY GROUP
# ============================================================

group_colours = {
    "Lag":         "#2196F3",   # blue
    "Rolling":     "#4CAF50",   # green
    "Momentum":    "#FF9800",   # orange
    "Temporal":    "#9C27B0",   # purple
    "Macro":       "#F44336",   # red
    "Weather":     "#00BCD4",   # cyan
    "KG Risk":     "#795548",   # brown
    "Interaction": "#E91E63",   # pink
}

def get_group(feat):
    if feat in lag_features:          return "Lag"
    if feat in rolling_features:      return "Rolling"
    if feat in momentum_features:     return "Momentum"
    if feat in temporal_features:     return "Temporal"
    if feat in macro_features:        return "Macro"
    if feat in weather_features:      return "Weather"
    if feat in kg_features:           return "KG Risk"
    if feat in interaction_features:  return "Interaction"
    return "Other"

importances["Group"]  = importances["Feature"].apply(get_group)
importances["Colour"] = importances["Group"].map(group_colours).fillna("#9E9E9E")


# ============================================================
# PLOT 1 — TOP N HORIZONTAL BAR CHART
# ============================================================

top = importances.head(TOP_N).copy()

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top["Feature"][::-1],
    top["Importance"][::-1],
    color=top["Colour"].tolist()[::-1],
    edgecolor="white",
    linewidth=0.6
)

# Value labels on bars
for bar, val in zip(bars, top["Importance"][::-1]):
    ax.text(
        bar.get_width() + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", fontsize=8, color="#444"
    )

ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title(
    f"Top {TOP_N} Feature Importances\n"
    f"Random Forest — FULL MODEL  |  t+7 Horizon  |  Test R² = {test_r2:.3f}",
    fontsize=13, fontweight="bold", pad=14
)
ax.grid(axis="x", linestyle="--", alpha=0.45, color="#ccc")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, top["Importance"].max() * 1.18)

# Legend
legend_handles = [
    mpatches.Patch(facecolor=colour, label=group)
    for group, colour in group_colours.items()
    if group in top["Group"].values
]
ax.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=9,
    framealpha=0.8,
    title="Feature Group",
    title_fontsize=9
)

plt.tight_layout()
plt.savefig("feature_importance_top20.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] feature_importance_top20.png")


# ============================================================
# PLOT 2 — GROUPED BAR CHART (total importance per group)
# ============================================================

group_total = (
    importances.groupby("Group")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
group_total["Colour"] = group_total["Group"].map(group_colours).fillna("#9E9E9E")

fig2, ax2 = plt.subplots(figsize=(9, 5))

bars2 = ax2.bar(
    group_total["Group"],
    group_total["Importance"],
    color=group_total["Colour"],
    edgecolor="white",
    linewidth=0.8
)

for bar, val in zip(bars2, group_total["Importance"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=9, color="#333"
    )

ax2.set_ylabel("Total Feature Importance", fontsize=11)
ax2.set_title(
    "Feature Importance by Group\nRandom Forest — FULL MODEL  |  t+7 Horizon",
    fontsize=13, fontweight="bold", pad=12
)
ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#ccc")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(0, group_total["Importance"].max() * 1.18)

plt.tight_layout()
plt.savefig("feature_importance_by_group.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] feature_importance_by_group.png")


# ============================================================
# SUMMARY PRINTOUT
# ============================================================

print("\n===== IMPORTANCE BY GROUP =====")
print(group_total[["Group", "Importance"]].to_string(index=False))

print("\n===== TOP 5 FEATURES PER GROUP =====")
for group in group_total["Group"]:
    subset = importances[importances["Group"] == group].head(5)
    print(f"\n{group}:")
    print(subset[["Feature", "Importance"]].to_string(index=False))
