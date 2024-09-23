import os
import boto3
import logging
from botocore.exceptions import ClientError


class AWSClientManager:
    def __init__(self, account_id, role_name, region):
        self.account_id = account_id
        self.role_name = role_name
        self.region = region
        self.session = boto3.Session()
        self.sts_client = self.session.client('sts')
        self.logger = logging.getLogger(__name__)

    def assume_role(self):
        """
        Assumes the specified IAM role and returns temporary AWS credentials.
        """
        role_arn = f"arn:aws:iam::{self.account_id}:role/{self.role_name}"
        self.logger.info(f'Attempting to assume role: {role_arn}')

        try:
            response = self.sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="AssumedRoleSession"
            )
            self.logger.info(f"Assumed role {self.role_name} successfully.")
            return response['Credentials']
        except ClientError as e:
            self.logger.error(f"Error assuming role {self.role_name}: {e}")
            raise

    def get_client(self, service_name):
        """
        Returns a boto3 client for the specified AWS service using assumed role credentials.
        """
        credentials = self.assume_role()

        return boto3.client(
            service_name,
            region_name=self.region,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )