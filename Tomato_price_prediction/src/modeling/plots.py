import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/tomato_base_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Date range:", df["Date"].min(), "→", df["Date"].max())
print("Columns:", df.columns.tolist())


# ============================================================
# PLOT 1 — TOMATO PRICE TREND OVER TIME
# ============================================================

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(df["Date"], df["Average_Price"], color="#CC0000", linewidth=0.9)

ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("Average_Price", fontsize=11)
ax1.set_title("Tomato Price Trend Over Time", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax1.spines[["top", "right"]].set_visible(False)

# Format x-axis dates
import matplotlib.dates as mdates
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("plot1_price_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] plot1_price_trend.png")


# ============================================================
# PLOT 2 — SUPPLY VOLUME OVER TIME
# ============================================================

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(df["Date"], df["Supply_Volume"], color="#1565C0", linewidth=0.8)

ax2.set_xlabel("Date", fontsize=11)
ax2.set_ylabel("Supply_Volume", fontsize=11)
ax2.set_title("Supply Volume Over Time", fontsize=13, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax2.spines[["top", "right"]].set_visible(False)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=0)

# Format y-axis with comma separators
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{int(x):,}")
)

plt.tight_layout()
plt.savefig("plot2_supply_volume.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] plot2_supply_volume.png")


# ============================================================
# PLOT 3 — FEATURE CORRELATION HEATMAP
# ============================================================

# Select numeric columns relevant for correlation
# Includes price, supply, macro, and weather variables
keep_cols = (
    ["Average_Price", "Supply_Volume", "USD_TO_NPR", "Diesel", "Inflation"] +
    [c for c in df.columns if "Temperature" in c or "Rainfall_MM" in c]
)
keep_cols = [c for c in keep_cols if c in df.columns]

corr_df = df[keep_cols].corr().round(2)

# Shorten column labels for readability (keep prefix only)
short_labels = []
for col in corr_df.columns:
    if "_Temperature" in col:
        short_labels.append(col.replace("_Temperature", "\nTemperature"))
    elif "_Rainfall_MM" in col:
        short_labels.append(col.replace("_Rainfall_MM", "\nRainfall_MM"))
    else:
        short_labels.append(col)

fig3, ax3 = plt.subplots(figsize=(14, 11))

sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    linewidths=0.4,
    linecolor="white",
    annot_kws={"size": 7},
    xticklabels=short_labels,
    yticklabels=short_labels,
    ax=ax3,
    cbar_kws={"shrink": 0.8}
)

ax3.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=14)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig("plot3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] plot3_correlation_heatmap.png")


# ============================================================
# PLOT 4 — AVERAGE TOMATO PRICE BY MONTH
# ============================================================

df["Month"] = df["Date"].dt.month

monthly = df.groupby("Month")["Average_Price"].agg(["mean", "sem"]).reset_index()
monthly.columns = ["Month", "Mean", "SEM"]

# Viridis-style colour per bar (matching your image)
cmap = plt.get_cmap("viridis")
colours = [cmap(i / 11) for i in range(12)]

fig4, ax4 = plt.subplots(figsize=(11, 6))

bars = ax4.bar(
    monthly["Month"],
    monthly["Mean"],
    yerr=monthly["SEM"],
    color=colours,
    edgecolor="none",
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "#333333", "capthick": 1.2}
)

ax4.set_xlabel("Month", fontsize=11)
ax4.set_ylabel("Average_Price", fontsize=11)
ax4.set_title("Average Tomato Price by Month", fontsize=13, fontweight="bold")
ax4.set_xticks(range(1, 13))
ax4.grid(axis="y", linestyle="--", alpha=0.5, color="#cccccc")
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_ylim(0, monthly["Mean"].max() * 1.2)

plt.tight_layout()
plt.savefig("plot4_price_by_month.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] plot4_price_by_month.png")


# ============================================================
# PLOT 5 — LAG PLOT (Price vs Previous Day)
# ============================================================

df["lag1"] = df["Average_Price"].shift(1)
lag_df = df[["lag1", "Average_Price"]].dropna()

fig5, ax5 = plt.subplots(figsize=(8, 7))

ax5.scatter(
    lag_df["lag1"],
    lag_df["Average_Price"],
    alpha=0.45,
    s=18,
    color="#1565C0",
    edgecolors="none"
)

ax5.set_xlabel("lag1", fontsize=11)
ax5.set_ylabel("Average_Price", fontsize=11)
ax5.set_title("Lag Plot (Price vs Previous Day)", fontsize=13, fontweight="bold")
ax5.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
ax5.spines[["top", "right"]].set_visible(False)

# Print autocorrelation
r = lag_df["lag1"].corr(lag_df["Average_Price"])
print(f"\nLag-1 autocorrelation: {r:.3f}")
ax5.text(
    0.05, 0.93,
    f"Autocorrelation (lag-1) = {r:.3f}",
    transform=ax5.transAxes,
    fontsize=9,
    color="#333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

plt.tight_layout()
plt.savefig("plot5_lag_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("[SAVED] plot5_lag_plot.png")


# ============================================================
# SUMMARY
# ============================================================

print("\n===== ALL PLOTS SAVED =====")
print("  plot1_price_trend.png")
print("  plot2_supply_volume.png")
print("  plot3_correlation_heatmap.png")
print("  plot4_price_by_month.png")
print("  plot5_lag_plot.png")
