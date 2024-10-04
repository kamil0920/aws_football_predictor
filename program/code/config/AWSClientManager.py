import logging

import boto3
from botocore.exceptions import ClientError


class AWSClientManager:
    def __init__(self, region, access_key_id=None, secret_access_key=None, session_token=None, account_id=None, role_name=None):
        """
        Initialize the AWSClientManager.

        Parameters:
        - region: AWS region to use.
        - access_key_id: IAM user's access key ID (for user-based credentials).
        - secret_access_key: IAM user's secret access key (for user-based credentials).
        - session_token: Optional session token (for user-based or temporary credentials).
        - account_id: AWS Account ID (used for role assumption).
        - role_name: Name of the role to assume (used for role assumption).
        """
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.account_id = account_id
        self.role_name = role_name
        self.logger = logging.getLogger(self.__class__.__name__)

        self.session = boto3.Session()
        self.sts_client = self.session.client(
            'sts',
            region_name=self.region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token
        )
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
                RoleSessionName="AssumedFootballRoleSession"
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
