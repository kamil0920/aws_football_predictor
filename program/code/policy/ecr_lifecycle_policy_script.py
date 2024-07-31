import boto3
import json
import os
import time


class ECRLifecyclePolicyService:
    def __init__(self, account_id, user_name, role_name, policy_name, image_name):
        self.account_id = account_id
        self.user_name = user_name
        self.role_name = role_name
        self.policy_name = policy_name
        self.image_name = image_name
        self.session = boto3.Session()
        self.iam_client = self.session.client('iam')
        self.sts_client = self.session.client('sts')

    def create_role_and_policy(self):
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": f"arn:aws:iam::{self.account_id}:user/{self.user_name}"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:PutLifecyclePolicy"
                    ],
                    "Resource": "*"
                }
            ]
        }

        try:
            response = self.iam_client.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role to allow putting lifecycle policy in ECR registry'
            )
            print(f"Role {self.role_name} created successfully.")
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Role {self.role_name} already exists.")
            response = self.iam_client.get_role(RoleName=self.role_name)

        policy_arn = f"arn:aws:iam::{self.account_id}:policy/{self.policy_name}"

        try:
            response = self.iam_client.create_policy(
                PolicyName=self.policy_name,
                PolicyDocument=json.dumps(permissions_policy),
                Description='Policy to allow putting lifecycle policy in ECR registry'
            )
            print(f"Policy {self.policy_name} created successfully.")
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Policy {self.policy_name} already exists.")
            response = self.iam_client.get_policy(PolicyArn=policy_arn)

        # Attach the policy to the role
        self.iam_client.attach_role_policy(
            RoleName=self.role_name,
            PolicyArn=policy_arn
        )
        print(f"Policy {self.policy_name} attached to role {self.role_name} successfully.")

        inline_policy_name = 'AssumeECRLifecyclePolicyRole'
        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "sts:AssumeRole",
                    "Resource": f"arn:aws:iam::{self.account_id}:role/{self.role_name}"
                }
            ]
        }

        response_put_user_policy = self.iam_client.put_user_policy(
            UserName=self.user_name,
            PolicyName=inline_policy_name,
            PolicyDocument=json.dumps(inline_policy)
        )

        print(f"Inline policy {inline_policy_name} attached to user {self.user_name} successfully.")

    def assume_role_and_put_lifecycle_policy(self):
        # TODO: check why for first time I get error and for second time it works
        try:
            assumed_role = self.sts_client.assume_role(
                RoleArn=f"arn:aws:iam::{self.account_id}:role/{self.role_name}",
                RoleSessionName="ECRLifecycleSession"
            )
        except Exception as e:
            print(f"Unable to assume role {self.role_name}, retrying in 2 seconds...")
            time.sleep(2)

            assumed_role = self.sts_client.assume_role(
                RoleArn=f"arn:aws:iam::{self.account_id}:role/{self.role_name}",
                RoleSessionName="ECRLifecycleSession"
            )

        credentials = assumed_role['Credentials']

        # Use the assumed role credentials to create an ECR client
        ecr_client = boto3.client(
            'ecr',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        lifecycle_policy = {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Keep only one untagged image, expire all others",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "imageCountMoreThan",
                        "countNumber": 1
                    },
                    "action": {
                        "type": "expire"
                    }
                }
            ]
        }

        response = ecr_client.put_lifecycle_policy(
            registryId=self.account_id,
            repositoryName=self.image_name,
            lifecyclePolicyText=json.dumps(lifecycle_policy)
        )

        print(f"Lifecycle policy attached to repository {self.image_name} successfully.")


# Example usage:
if __name__ == "__main__":
    account_id = os.getenv('ACCOUNT_ID')
    user_name = os.getenv('USER_PROFILE')
    role_name = 'ECRLifecyclePolicyRole'
    policy_name = 'ECRLifecyclePolicyPermission'
    image_name = 'xgb-clf-custom-training-container'

    manager = ECRLifecyclePolicyService(account_id, user_name, role_name, policy_name, image_name)
    manager.create_role_and_policy()
    manager.assume_role_and_put_lifecycle_policy()
