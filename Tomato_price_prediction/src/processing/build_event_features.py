"""
build_event_features.py
-----------------------
Convert district event data into daily risk features.

Input:
data/raw/events/district_events.csv

Output:
data/processed/daily_event_risk.csv
"""

from pathlib import Path
import pandas as pd


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_INPUT = PROJECT_ROOT / "data" / "raw" / "events" / "district_events.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "daily_event_risk.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# Config
# =========================================================
DISTRICTS = ["Kathmandu", "Sarlahi", "Dhading", "Kavre"]

# Example supply weights for market-risk propagation
DISTRICT_WEIGHTS = {
    "Kathmandu": 0.15,
    "Sarlahi": 0.40,
    "Dhading": 0.25,
    "Kavre": 0.20,
}


# =========================================================
# Helpers
# =========================================================
def severity_to_risk(severity: float) -> float:
    """Convert severity 1-5 to 0.2-1.0"""
    if pd.isna(severity):
        return 0.0
    return float(severity) / 5.0


# =========================================================
# Main Transform
# =========================================================
def build_daily_event_risk():
    print(f"[INFO] Reading event file: {EVENT_INPUT}")

    if not EVENT_INPUT.exists():
        raise FileNotFoundError(f"Event file not found: {EVENT_INPUT}")

    df = pd.read_csv(EVENT_INPUT)

    if df.empty:
        print("[WARN] district_events.csv is empty.")
        out = pd.DataFrame(columns=["Date"] + [f"{d}_Risk" for d in DISTRICTS] + ["Market_Risk"])
        out.to_csv(OUTPUT_PATH, index=False)
        print(f"[SUCCESS] Saved empty risk file -> {OUTPUT_PATH}")
        return out

    required_cols = ["Date", "District", "Event_Type", "Severity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in district_events.csv: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")
    df = df.dropna(subset=["Date", "District", "Severity"]).copy()

    df = df[df["District"].isin(DISTRICTS)].copy()
    df["Risk"] = df["Severity"].apply(severity_to_risk)

    # If multiple events happen in the same district on same day, keep max risk
    daily_district = (
        df.groupby(["Date", "District"], as_index=False)["Risk"]
        .max()
    )

    # Pivot to wide format
    wide = daily_district.pivot(index="Date", columns="District", values="Risk").reset_index()

    # Ensure all district columns exist
    for d in DISTRICTS:
        if d not in wide.columns:
            wide[d] = 0.0

    # Rename columns
    wide = wide.rename(columns={d: f"{d}_Risk" for d in DISTRICTS})

    # Fill missing dates from min to max with 0 risk
    min_date = wide["Date"].min()
    max_date = wide["Date"].max()

    full_dates = pd.DataFrame({"Date": pd.date_range(min_date, max_date, freq="D")})
    out = full_dates.merge(wide, on="Date", how="left")

    risk_cols = [f"{d}_Risk" for d in DISTRICTS]
    out[risk_cols] = out[risk_cols].fillna(0.0)

    # Market risk = weighted sum
    out["Market_Risk"] = (
        out["Kathmandu_Risk"] * DISTRICT_WEIGHTS["Kathmandu"] +
        out["Sarlahi_Risk"] * DISTRICT_WEIGHTS["Sarlahi"] +
        out["Dhading_Risk"] * DISTRICT_WEIGHTS["Dhading"] +
        out["Kavre_Risk"] * DISTRICT_WEIGHTS["Kavre"]
    )

    out = out.sort_values("Date").reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[SUCCESS] Daily event risk saved -> {OUTPUT_PATH}")
    print(out.head(10))

    return out


if __name__ == "__main__":
    build_daily_event_risk()