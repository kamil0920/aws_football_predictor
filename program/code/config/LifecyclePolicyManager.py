import json

import boto3
from botocore.exceptions import ClientError


class LifecyclePolicyManager:
    def __init__(self, account_id):
        self.account_id = account_id
        self.session = boto3.Session()
        self.ecr_client = self.session.client('ecr')

    def put_lifecycle_policy(self, ecr_client, image_name, lifecycle_policy):
        try:
            response = ecr_client.put_lifecycle_policy(
                registryId=self.account_id,
                repositoryName=image_name,
                lifecyclePolicyText=json.dumps(lifecycle_policy)
            )
            print(f"Lifecycle policy attached to repository {image_name} successfully.")
        except ClientError as e:
            print(f"Error putting lifecycle policy: {e}")
            raise

    # def assume_role_and_put_lifecycle_policy(self):
    #     max_retries = 3
    #     for attempt in range(max_retries):
    #         try:
    #             credentials = self.assume_role()
    #             ecr_client = self.create_ecr_client(credentials)
    #             self.put_lifecycle_policy(ecr_client)
    #             break
    #         except ClientError as e:
    #             if attempt < max_retries - 1:
    #                 print(f"Retrying in 5 seconds... ({attempt + 1}/{max_retries})")
    #                 time.sleep(5)
    #             else:
    #                 print(f"Failed after {max_retries} attempts: {e}")
    #                 raise