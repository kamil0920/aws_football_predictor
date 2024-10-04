import json
import logging

import boto3
from botocore.exceptions import ClientError


class IAMUserManager:
    def __init__(self):
        """Initialize the IAM client."""
        self.iam_client = boto3.client('iam')
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_user(self, username):
        """Create a new IAM user."""
        try:
            response = self.iam_client.create_user(UserName=username)
            print(f"User {username} created successfully.")
            return response

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            print(f"User {username} already exists.")
            return username

        except ClientError as error:
            print(f"Error creating user {username}: {error}")
            return None

    def attach_inline_policy(self, username, role_arn, policy_name="AssumeRolePolicy"):
        """
        Attach an inline policy to the user that allows them to assume the given role.

        Parameters:
        - username: IAM user to whom the policy is being attached
        - role_arn: The ARN of the role the user should be able to assume
        - policy_name: Name of the inline policy (default: 'AssumeRolePolicy')
        """

        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "sts:AssumeRole",
                    "Resource": role_arn
                }
            ]
        }

        try:
            self.iam_client.put_user_policy(
                UserName=username,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(assume_role_policy)
            )
            print(f"Inline policy {policy_name} attached to user {username}.")
        except ClientError as error:
            print(f"Error attaching policy to user {username}: {error}")

    def create_access_key(self, username):
        """Create and return access keys for the IAM user."""
        try:
            response = self.iam_client.create_access_key(UserName=username)
            print(f"Access keys created for user {username}.")
            return response['AccessKey']
        except ClientError as error:
            print(f"Error creating access keys for user {username}: {error}")
            return None

    def create_user_with_inline_policy_and_secrets(self, username, role_arn):
        """
        Full process to create a user and attach an inline policy for role assumption.

        Parameters:
        - username: The IAM username to be created
        - role_arn: The ARN of the role the user should be able to assume
        """

        user_response = self.create_user(username)
        if user_response is not None:
            self.attach_inline_policy(username, role_arn)
            return self.create_access_key(username)
        else:
            print(f"User {username} could not be created, skipping policy attachment.")


# Example usage of the class
# if __name__ == "__main__":
#     # Provide the username and role ARN
#     username = 'new-username'
#     role_arn = 'arn:aws:iam::123456789012:role/MyRole'
#
#     # Initialize the IAMUserManager class
#     iam_manager = IAMUserManager()
#
#     # Create a user and attach an inline policy
#     iam_manager.create_user_with_policy(username, role_arn)