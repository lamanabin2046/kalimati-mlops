"""
run_training.py — SageMaker Training Job Launcher
---------------------------------------------------
Runs from EC2 to launch a SageMaker training job.
- Reads features from S3
- Launches XGBoost training on SageMaker
- Saves model to S3
- Registers model in SageMaker Model Registry

Run: python3 run_training.py
"""

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
from datetime import datetime

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
S3_BUCKET           = "kalimati-price-prediction"
S3_TRAIN_INPUT      = f"s3://{S3_BUCKET}/features/"
S3_MODEL_OUTPUT     = f"s3://{S3_BUCKET}/models/"
ROLE                = "arn:aws:iam::135723932609:role/LabRole"
REGION              = "us-east-1"
MODEL_PACKAGE_GROUP = "tomato-price-xgboost"

# ---------------------------------------------------------
# Setup SageMaker Session
# ---------------------------------------------------------
boto_session      = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# ---------------------------------------------------------
# Create Model Package Group (first time only)
# ---------------------------------------------------------
def create_model_package_group():
    sm_client = boto3.client("sagemaker", region_name=REGION)
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName       = MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription= "Tomato price prediction XGBoost models"
        )
        print(f"[INFO] Created model package group: {MODEL_PACKAGE_GROUP}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"[INFO] Model package group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise


# ---------------------------------------------------------
# Launch Training Job
# ---------------------------------------------------------
def run_training():
    print("=" * 60)
    print("[INFO] Launching SageMaker Training Job")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name  = f"tomato-xgboost-{timestamp}"

    print(f"[INFO] Job name     : {job_name}")
    print(f"[INFO] Input data   : {S3_TRAIN_INPUT}")
    print(f"[INFO] Model output : {S3_MODEL_OUTPUT}")
    print(f"[INFO] Role         : {ROLE}")

    # Get XGBoost container image
    container = image_uris.retrieve("xgboost", REGION, version="1.7-1")
    print(f"[INFO] Container    : {container}")

    # Define Estimator
    estimator = Estimator(
        image_uri         = container,
        role              = ROLE,
        instance_count    = 1,
        instance_type     = "ml.m5.large",
        output_path       = S3_MODEL_OUTPUT,
        sagemaker_session = sagemaker_session,
        base_job_name     = "tomato-xgboost",
        hyperparameters   = {
            "max_depth":    "6",
            "eta":          "0.05",
            "num_round":    "500",
            "objective":    "reg:squarederror",
            "subsample":    "0.8",
        },
    )

    # Upload train.py as a source
    estimator.entry_point = "train.py"
    estimator.source_dir  = "/home/ubuntu/training"

    # Define training input
    train_input = TrainingInput(
        s3_data      = S3_TRAIN_INPUT,
        content_type = "text/csv",
    )

    # Launch training job
    print("\n[INFO] Starting training job... this may take 5-10 minutes.")
    estimator.fit({"train": train_input}, wait=True, logs="All")

    print(f"\n[SUCCESS] Training job completed: {job_name}")
    print(f"[INFO] Model artifacts: {estimator.model_data}")

    return estimator, job_name


# ---------------------------------------------------------
# Register Model in Model Registry
# ---------------------------------------------------------
def register_model(estimator, job_name):
    print("\n[INFO] Registering model in SageMaker Model Registry...")

    sm_client = boto3.client("sagemaker", region_name=REGION)

    try:
        container = image_uris.retrieve("xgboost", REGION, version="1.7-1")

        response = sm_client.create_model_package(
            ModelPackageGroupName   = MODEL_PACKAGE_GROUP,
            ModelPackageDescription = f"Tomato price XGBoost trained {job_name}",
            InferenceSpecification  = {
                "Containers": [{
                    "Image":        container,
                    "ModelDataUrl": estimator.model_data,
                }],
                "SupportedContentTypes":      ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            ModelApprovalStatus = "Approved",
        )
        model_package_arn = response["ModelPackageArn"]
        print(f"[SUCCESS] Model registered: {model_package_arn}")
        return model_package_arn

    except Exception as e:
        print(f"[WARN] Could not register model: {e}")
        return None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Create model package group if needed
    create_model_package_group()

    # Run training
    estimator, job_name = run_training()

    # Register model
    register_model(estimator, job_name)

    print("\n" + "=" * 60)
    print("[DONE] Training pipeline complete!")
    print(f"  Model artifacts : {estimator.model_data}")
    print("=" * 60)


if __name__ == "__main__":
    main()

