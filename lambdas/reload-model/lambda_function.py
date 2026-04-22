"""
reload_model Lambda
--------------------
Called by Step Functions after training completes.
Calls the FastAPI /api/reload-model endpoint on EC2.
"""

import json
import urllib.request

EC2_URL = "http://52.201.197.75/api/reload-model"

def lambda_handler(event, context):
    print("[INFO] Reloading model on EC2...")
    try:
        req  = urllib.request.Request(EC2_URL, method="POST")
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read().decode())
        print(f"[SUCCESS] Model reloaded: {body}")
        return {"statusCode": 200, "body": body}
    except Exception as e:
        print(f"[ERROR] Failed to reload model: {e}")
        raise
