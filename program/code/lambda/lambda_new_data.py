import os
import boto3

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')
dynamodb = boto3.client('dynamodb')

table_name = 'S3FileUploadStatus'


def lambda_handler(event, context):
    print("Lambda triggered by S3 event")
    bucket_name = event['detail']['bucket']['name']
    object_key = event['detail']['object']['key']

    folder_path = os.path.dirname(object_key) + "/"
    print(f'folder_path: {folder_path}')

    if not object_key.startswith("data/") or not object_key.endswith(".csv"):
        print(f"Skipping object: {object_key}")
        return {"message": "Object is not in the 'data/' folder or not a CSV file."}

    response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
    metadata = response.get('Metadata', {})
    batch_id = metadata.get('batch-id')

    if not batch_id:
        print("batch-id metadata is missing")
        return {"message": "batch-id metadata is missing"}

    file_type = 'df' if 'df.csv' in object_key else 'y' if 'y.csv' in object_key else None
    if not file_type:
        return {'statusCode': 400, 'body': 'File type not recognized'}

    response = dynamodb.get_item(
        TableName=table_name,
        Key={'BatchId': {'S': batch_id}}
    )

    if 'Item' in response:
        other_file_type = 'y' if file_type == 'df' else 'df'
        if response['Item'].get(other_file_type):
            trigger_pipeline(bucket_name, folder_path)

            dynamodb.delete_item(
                TableName=table_name,
                Key={'BatchId': {'S': batch_id}}
            )

            return {"message": "Both files uploaded, pipeline triggered and DynamoDB entry cleaned."}
        else:
            dynamodb.update_item(
                TableName=table_name,
                Key={'BatchId': {'S': batch_id}},
                UpdateExpression=f"SET {file_type} = :val",
                ExpressionAttributeValues={':val': {'BOOL': True}}
            )
            return {"message": f"{file_type}.csv uploaded, waiting for the other file."}

    dynamodb.put_item(
        TableName=table_name,
        Item={
            'BatchId': {'S': batch_id},
            file_type: {'BOOL': True}
        }
    )

    return {"message": f"{file_type}.csv uploaded, waiting for the other file."}


def trigger_pipeline(bucket_name, folder_path):
    dataset_location = f"s3://{bucket_name}/{folder_path}"
    print(f'dataset_location: {dataset_location}')

    response = sagemaker_client.start_pipeline_execution(
        PipelineName=os.environ['PIPELINE_NAME'],
        PipelineExecutionDisplayName='TriggeredByS3Upload',
        PipelineParameters=[
            {
                'Name': 'dataset_location',
                'Value': dataset_location
            }
        ]
    )

    print("SageMaker Pipeline Execution ARN:", response['PipelineExecutionArn'])