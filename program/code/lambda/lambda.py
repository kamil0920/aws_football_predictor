import os
import json
import boto3
import time

sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):
    if "detail" in event:
        approval_status = event["detail"]["ModelApprovalStatus"]
    else:
        approval_status = "Approved"

    if approval_status != "Approved":
        response = {
            "message": "Skipping deployment.",
            "approval_status": approval_status,
        }

        print(response)
        return {"statusCode": 200, "body": json.dumps(response)}

    data_capture_percentage = int(os.environ["DATA_CAPTURE_PERCENTAGE"])
    data_capture_destination = os.environ["DATA_CAPTURE_DESTINATION"]
    model_package_group = os.environ["MODEL_PACKAGE_GROUP"]
    endpoint_name = os.environ["ENDPOINT"]
    role = os.environ["ROLE"]

    response = sagemaker.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        MaxResults=2,
    )

    if response["ModelPackageSummaryList"]:
        production_package = response["ModelPackageSummaryList"][1]["ModelPackageArn"]
        shadow_package = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    else:
        production_package = None
        shadow_package = None

    print(f"Production package: {production_package}")
    print(f"Shadow package: {shadow_package}")

    timestamp = time.strftime("%m%d%H%M%S", time.localtime())
    prod_model_name = f"{endpoint_name}-model-prod-{timestamp}"
    shadow_model_name = f"{endpoint_name}-model-shadow-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    sagemaker.create_model(ModelName=prod_model_name, ExecutionRoleArn=role , Containers=[{"ModelPackageName": production_package}])
    sagemaker.create_model(ModelName=shadow_model_name, ExecutionRoleArn=role, Containers=[{"ModelPackageName": shadow_package}])

    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "ModelName": prod_model_name,
                "InstanceType": "ml.m5.xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "VariantName": "production-variant",
            }
        ],
        ShadowProductionVariants=[
            {
                "ModelName": shadow_model_name,
                "InstanceType": "ml.m5.xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "VariantName": "shadow-variant"
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": data_capture_percentage,
            "DestinationS3Uri": data_capture_destination,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": ["text/csv", "application/octect-stream"],
                "JsonContentTypes": ["application/json", "application/octect-stream"],
            },
        },
    )

    response = sagemaker.list_endpoints(NameContains=endpoint_name, MaxResults=1)

    if len(response["Endpoints"]) == 0:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    else:
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

    return {"statusCode": 200, "body": json.dumps("Endpoint deployed successfully")}
