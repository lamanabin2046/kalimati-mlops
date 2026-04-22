"""
nrb_inflation_check Lambda
---------------------------
Since NRB website data is unavailable, this Lambda:
- Reads existing inflation.csv from S3
- Forward-fills to today's date
- Saves back to S3

S3 path: s3://kalimati-price-prediction/raw/macro/inflation.csv
"""

import io
import boto3
import pandas as pd
from datetime import datetime

S3_BUCKET = "kalimati-price-prediction"
S3_KEY    = "raw/macro/inflation.csv"

def lambda_handler(event, context):
    print("[INFO] NRB Inflation Check Lambda started")
    s3 = boto3.client("s3")

    try:
        # Read existing inflation data from S3
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        last_date  = df["Date"].max()
        today      = pd.Timestamp(datetime.now().date())
        last_value = df["Inflation"].iloc[-1]

        print(f"[INFO] Last inflation date: {last_date.date()}")
        print(f"[INFO] Last inflation value: {last_value}")

        # Forward fill to today if needed
        if last_date < today:
            new_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                end=today,
                freq="MS"  # Monthly
            )
            if len(new_dates) > 0:
                new_rows = pd.DataFrame({
                    "Date":      new_dates,
                    "Inflation": last_value  # Forward fill with last known value
                })
                df = pd.concat([df, new_rows], ignore_index=True)
                df = df.drop_duplicates(subset=["Date"]).sort_values("Date")

                # Save back to S3
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=buffer.getvalue())
                print(f"[SUCCESS] Forward-filled {len(new_dates)} months to {today.date()}")
            else:
                print("[INFO] Inflation data already up to date")
        else:
            print("[INFO] Inflation data already up to date")

        return {
            "statusCode": 200,
            "body": f"Inflation data OK. Last value: {last_value}% as of {last_date.date()}"
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        # Don't fail pipeline - just return success with warning
        return {
            "statusCode": 200,
            "body": f"Warning: Could not update inflation data: {str(e)}. Using existing S3 data."
        }
