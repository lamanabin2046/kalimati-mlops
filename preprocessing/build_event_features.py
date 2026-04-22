"""
build_event_features.py — EC2 + S3 Version
--------------------------------------------
- Reads district_events.csv from S3
- Builds daily risk features
- Saves daily_event_risk.csv to S3

S3 inputs:  s3://kalimati-price-prediction/raw/events/district_events.csv
S3 output:  s3://kalimati-price-prediction/processed/daily_event_risk.csv
"""

import io
import boto3
import pandas as pd

# ---------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------
S3_BUCKET   = "kalimati-price-prediction"
S3_INPUT    = "raw/events/district_events.csv"
S3_OUTPUT   = "processed/daily_event_risk.csv"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DISTRICTS = ["Kathmandu", "Sarlahi", "Dhading", "Kavre"]

DISTRICT_WEIGHTS = {
    "Kathmandu": 0.15,
    "Sarlahi":   0.40,
    "Dhading":   0.25,
    "Kavre":     0.20,
}

# ---------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------
def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"[INFO] Loaded {len(df)} rows from s3://{bucket}/{key}")
        return df
    except Exception as e:
        print(f"[WARN] Could not read {key}: {e}")
        return pd.DataFrame()


def write_csv_to_s3(df, bucket, key):
    s3     = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"[SUCCESS] Saved {len(df)} rows to s3://{bucket}/{key}")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def severity_to_risk(severity):
    if pd.isna(severity):
        return 0.0
    return float(severity) / 5.0


# ---------------------------------------------------------
# Main Transform
# ---------------------------------------------------------
def build_daily_event_risk():
    print("[INFO] Building daily event risk features...")

    df = read_csv_from_s3(S3_BUCKET, S3_INPUT)

    if df.empty:
        print("[WARN] district_events.csv is empty. Saving empty risk file.")
        out = pd.DataFrame(columns=["Date"] + [f"{d}_Risk" for d in DISTRICTS] + ["Market_Risk"])
        write_csv_to_s3(out, S3_BUCKET, S3_OUTPUT)
        return out

    df["Date"]     = pd.to_datetime(df["Date"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")
    df = df.dropna(subset=["Date", "District", "Severity"])
    df = df[df["District"].isin(DISTRICTS)].copy()
    df["Risk"] = df["Severity"].apply(severity_to_risk)

    # Max risk per district per day
    daily_district = (df.groupby(["Date", "District"], as_index=False)["Risk"].max())

    # Pivot to wide
    wide = daily_district.pivot(index="Date", columns="District", values="Risk").reset_index()

    # Ensure all district columns exist
    for d in DISTRICTS:
        if d not in wide.columns:
            wide[d] = 0.0

    wide = wide.rename(columns={d: f"{d}_Risk" for d in DISTRICTS})

    # Fill full date range
    full_dates = pd.DataFrame({"Date": pd.date_range(wide["Date"].min(), wide["Date"].max(), freq="D")})
    out        = full_dates.merge(wide, on="Date", how="left")

    risk_cols      = [f"{d}_Risk" for d in DISTRICTS]
    out[risk_cols] = out[risk_cols].fillna(0.0)

    # Weighted market risk
    out["Market_Risk"] = (
        out["Kathmandu_Risk"] * DISTRICT_WEIGHTS["Kathmandu"] +
        out["Sarlahi_Risk"]   * DISTRICT_WEIGHTS["Sarlahi"]   +
        out["Dhading_Risk"]   * DISTRICT_WEIGHTS["Dhading"]   +
        out["Kavre_Risk"]     * DISTRICT_WEIGHTS["Kavre"]
    )

    out = out.sort_values("Date").reset_index(drop=True)

    write_csv_to_s3(out, S3_BUCKET, S3_OUTPUT)
    print(f"[INFO] Daily event risk rows: {len(out)}")

    return out


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    build_daily_event_risk()

