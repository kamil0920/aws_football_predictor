import json
import logging
import boto3
from botocore.exceptions import ClientError

from program.code.config.PolicyManager import PolicyManager


class RoleManager:
    def __init__(self, account_id, user_name, role_name, policy_name, region):
        self.account_id = account_id
        self.user_name = user_name
        self.role_name = role_name
        self.policy_name = policy_name
        self.region = region
        self.logger = logging.getLogger(__name__)
        self.session = boto3.Session()
        self.iam_client = self.session.client('iam')

    def update_role_policy(self):
        """
        Updates the trust relationship policy of the IAM role.
        """
        trust_relationship_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": [
                            "sagemaker.amazonaws.com",
                            "events.amazonaws.com"
                        ],
                    },
                    "Action": "sts:AssumeRole"
                },
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": f"arn:aws:iam::{self.account_id}:user/{self.user_name}"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        try:
            self.iam_client.update_assume_role_policy(
                RoleName=self.role_name,
                PolicyDocument=json.dumps(trust_relationship_policy)
            )
            print(f"Successfully updated trust relationship for role: {self.role_name}")
        except ClientError as e:
            print(f"Error updating trust relationship: {e}")
            raise

    def create_role_and_policy(self):
        """
        Orchestrates the creation of policies and role updates.
        """
        policy_manager = PolicyManager(self.account_id, self.user_name)

        # Step 1: Update Trust Relationship
        self.update_role_policy()

        # Step 2: Create Permissions Policy
        policy_arn = policy_manager.create_permissions_policy(self.iam_client, self.policy_name)

        # Step 3: Attach Policy to Role
        policy_manager.attach_policy_to_role(self.iam_client, policy_arn, self.role_name)

        # Step 4: Attach Inline Policy to User
        policy_manager.attach_inline_policy_to_user(self.iam_client, self.role_name)
