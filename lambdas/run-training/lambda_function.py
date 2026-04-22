"""
run_training Lambda
--------------------
Called by Step Functions to trigger SageMaker training.
Uses SSM Run Command to execute run_training.py on EC2.
"""

import boto3
import time

EC2_INSTANCE_ID = "i-0491674c8535b4bc7"
REGION          = "us-east-1"

def lambda_handler(event, context):
    ssm = boto3.client("ssm", region_name=REGION)

    print("[INFO] Running run_training.py on EC2 via SSM...")

    response = ssm.send_command(
        InstanceIds=[EC2_INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [
                "/home/ubuntu/scraper-env/bin/python /home/ubuntu/training/run_training.py"
            ]
        },
        TimeoutSeconds=900,
    )

    command_id = response["Command"]["CommandId"]
    print(f"[INFO] SSM Command ID: {command_id}")

    # Wait for command to complete (training takes ~5-10 minutes)
    time.sleep(30)
    for _ in range(40):
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=EC2_INSTANCE_ID,
        )
        status = result["Status"]
        print(f"[INFO] Command status: {status}")

        if status == "Success":
            print(f"[SUCCESS] Training completed!")
            print(f"Output: {result['StandardOutputContent'][-2000:]}")
            return {"statusCode": 200, "body": "Training completed successfully"}
        elif status in ["Failed", "Cancelled", "TimedOut"]:
            raise Exception(f"Training failed: {result['StandardErrorContent']}")

        time.sleep(15)

    raise Exception("Training timed out")
