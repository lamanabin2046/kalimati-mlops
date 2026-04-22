"""
build_event_features Lambda
-----------------------------
Called by Step Functions after scraping completes.
Uses SSM Run Command to execute build_event_features.py on EC2.
"""

import boto3
import time

EC2_INSTANCE_ID = "i-0491674c8535b4bc7"  # Your EC2 instance ID
REGION          = "us-east-1"

def lambda_handler(event, context):
    ssm = boto3.client("ssm", region_name=REGION)

    print("[INFO] Running build_event_features.py on EC2...")

    response = ssm.send_command(
        InstanceIds=[EC2_INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [
                "/home/ubuntu/scraper-env/bin/python /home/ubuntu/preprocessing/build_event_features.py"
            ]
        },
        TimeoutSeconds=300,
    )

    command_id = response["Command"]["CommandId"]
    print(f"[INFO] SSM Command ID: {command_id}")

    # Wait for command to complete
    time.sleep(10)
    for _ in range(30):
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=EC2_INSTANCE_ID,
        )
        status = result["Status"]
        print(f"[INFO] Command status: {status}")

        if status == "Success":
            print(f"[SUCCESS] Output: {result['StandardOutputContent']}")
            return {"statusCode": 200, "body": "build_event_features completed"}
        elif status in ["Failed", "Cancelled", "TimedOut"]:
            raise Exception(f"Command failed: {result['StandardErrorContent']}")

        time.sleep(10)

    raise Exception("Command timed out waiting for completion")
