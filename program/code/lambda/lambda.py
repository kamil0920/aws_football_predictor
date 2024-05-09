import os
import json
import boto3
import time

sagemaker = boto3.client("sagemaker")


def lambda_handler(event, context):
    # If we are calling this function from EventBridge,
    # we need to extract the model package ARN and the
    # approval status from the event details. If we are
    # calling this function from the pipeline, we can
    # assume the model is approved and we can get the
    # model package ARN as a direct parameter.
    if "detail" in event:
        model_package_arn = event["detail"]["ModelPackageArn"]
        approval_status = event["detail"]["ModelApprovalStatus"]
    else:
        model_package_arn = event["model_package_arn"]
        approval_status = "Approved"

    print(f"Model: {model_package_arn}")
    print(f"Approval status: {approval_status}")

    if approval_status != "Approved":
        response = {
            "message": "Skipping deployment.",
            "approval_status": approval_status,
        }

        print(response)
        return {"statusCode": 200, "body": json.dumps(response)}

    endpoint_name = os.environ["ENDPOINT"]
    data_capture_percentage = int(os.environ["DATA_CAPTURE_PERCENTAGE"])
    data_capture_destination = os.environ["DATA_CAPTURE_DESTINATION"]
    role = os.environ["ROLE"]
    pacakge_group = os.environ["MODEL_PACKAGE_GROUP"]

    timestamp = time.strftime("%m%d%H%M%S", time.localtime())
    model_name = f"{endpoint_name}-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    response = sagemaker.list_model_packages(
        ModelPackageGroupName=pacakge_group,
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

    production_model_name = f"{endpoint_name}-production-{timestamp}"

    sagemaker.create_model(
        ModelName=production_model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"ModelPackageName": production_package}
    )

    shadow_model_name = f"{endpoint_name}-shadow-{timestamp}"

    sagemaker.create_model(
        ModelName=shadow_model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"ModelPackageName": shadow_package}
    )

    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "ModelName": production_model_name,
                "InstanceType": "ml.c5.4xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        ShadowProductionVariants=[
            {
                "ModelName": shadow_model_name,
                "InstanceType": "ml.c5.4xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "VariantName": "ShadowTraffic",
            },
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
