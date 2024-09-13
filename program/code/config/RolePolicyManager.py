import boto3
import json
import os
import time

from botocore.exceptions import ClientError


class RolePolicyManager:
    def __init__(self, account_id, user_name, role_name, policy_name, region, bucket):
        self.account_id = account_id
        self.bucket = bucket
        self.region = region
        self.user_name = user_name
        self.role_name = role_name
        self.policy_name = policy_name
        self.session = boto3.Session()
        self.iam_client = self.session.client('iam')
        self.sts_client = self.session.client('sts')

    def create_role_and_policy(self):
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
            response = self.iam_client.update_assume_role_policy(
                RoleName=self.role_name,
                PolicyDocument=json.dumps(trust_relationship_policy)
            )
            print(f"Successfully updated trust relationship for role: {self.role_name}")
        except Exception as e:
            print(f"Error updating trust relationship: {str(e)}")
            return

        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ECR",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:CreateRepository",
                        "ecr:DescribeRepositories",
                        "ecr:DeleteRepository",
                        "ecr:ListImages",
                        "ecr:PutImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:TagResource",
                        "ecr:PutLifecyclePolicy",
                        "ecr:GetAuthorizationToken"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "S3",
                    "Effect": "Allow",
                    "Action": [
                        "s3:CreateBucket",
                        "s3:GetBucketLocation",
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "S3ListBucket",
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": "*"
                },
                {
                    "Sid": "EventBridge",
                    "Effect": "Allow",
                    "Action": [
                        "events:PutRule",
                        "events:PutTargets"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "IAM0",
                    "Effect": "Allow",
                    "Action": [
                        "iam:CreateServiceLinkedRole"
                    ],
                    "Resource": "*",
                    "Condition": {
                        "StringEquals": {
                            "iam:AWSServiceName": [
                                "autoscaling.amazonaws.com",
                                "ec2scheduled.amazonaws.com",
                                "elasticloadbalancing.amazonaws.com",
                                "spot.amazonaws.com",
                                "spotfleet.amazonaws.com",
                                "transitgateway.amazonaws.com"
                            ]
                        }
                    }
                },
                {
                    "Sid": "Lambda",
                    "Effect": "Allow",
                    "Action": [
                        "lambda:CreateFunction",
                        "lambda:DeleteFunction",
                        "lambda:InvokeFunctionUrl",
                        "lambda:InvokeFunction",
                        "lambda:UpdateFunctionCode",
                        "lambda:InvokeAsync",
                        "lambda:AddPermission",
                        "lambda:RemovePermission"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "SageMaker",
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:UpdateDomain",
                        "sagemaker:UpdateUserProfile"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "CloudWatch",
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData",
                        "cloudwatch:GetMetricData",
                        "cloudwatch:DescribeAlarmsForMetric",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:CreateLogGroup",
                        "logs:DescribeLogStreams"
                    ],
                    "Resource": "*"
                }
            ]
        }

        policy_arn = f"arn:aws:iam::{self.account_id}:policy/{self.policy_name}"

        try:
            response = self.iam_client.create_policy(
                PolicyName=self.policy_name,
                PolicyDocument=json.dumps(permissions_policy),
                Description='Policy for football match predictions project.'
            )
            print(f"Policy {response['Policy']['PolicyName']} created successfully.")
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Policy {self.policy_name} already exists.")
            print(f"Policy arn: {policy_arn}.")
            return
        except ClientError as e:
            print(f"Error creating policy: {e}")
            return

        try:
            self.iam_client.attach_role_policy(
                RoleName=self.role_name,
                PolicyArn=policy_arn
            )
            print(f"Policy {self.policy_name} attached to role {self.role_name} successfully.")
            print(f"Policy arn: {policy_arn}.")
        except ClientError as e:
            print(f"Error attaching policy: {e}")
            return


        inline_policy_name = 'FootballMatchPolicy'
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

        try:
            response_put_user_policy = self.iam_client.put_user_policy(
                UserName=self.user_name,
                PolicyName=inline_policy_name,
                PolicyDocument=json.dumps(inline_policy)
            )
            print(f"Inline policy {inline_policy_name} attached to user {self.user_name} successfully.")
        except ClientError as e:
            print(f"Error attaching inline policy: {e}")
            return

    def create_client(self, resource_type):
        try:
            assumed_role = self.sts_client.assume_role(
                RoleArn=f"arn:aws:iam::{self.account_id}:role/{self.role_name}",
                RoleSessionName="FootballPredictorRoleSession"
            )
        except Exception as e:
            print(f"Unable to assume role {self.role_name}, retrying in 2 seconds...")
            time.sleep(2)

            assumed_role = self.sts_client.assume_role(
                RoleArn=f"arn:aws:iam::{self.account_id}:role/{self.role_name}",
                RoleSessionName="ECRLifecycleSession"
            )

        credentials = assumed_role['Credentials']

        os.environ['AWS_ACCESS_KEY_ID'] = credentials['AccessKeyId']
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['SecretAccessKey']
        os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']

        return boto3.client(
            resource_type,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

    # def put_lifecycle_policy(self, ecr_client):
    #     lifecycle_policy = {
    #         "rules": [
    #             {
    #                 "rulePriority": 1,
    #                 "description": "Keep only one untagged image, expire all others",
    #                 "selection": {
    #                     "tagStatus": "any",
    #                     "countType": "imageCountMoreThan",
    #                     "countNumber": 1
    #                 },
    #                 "action": {
    #                     "type": "expire"
    #                 }
    #             }
    #         ]
    #     }
    #
    #     try:
    #         response = ecr_client.put_lifecycle_policy(
    #             registryId=self.account_id,
    #             repositoryName=self.image_name,
    #             lifecyclePolicyText=json.dumps(lifecycle_policy)
    #         )
    #         print(f"Lifecycle policy attached to repository {response['Policy']['PolicyName']} successfully.")
    #     except ClientError as e:
    #         print(f"Error putting lifecycle policy: {e}")
    #         raise
    #
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

# Example usage:
# if __name__ == "__main__":
#     account_id = os.getenv('ACCOUNT_ID')
#     user_name = os.getenv('USER_PROFILE')
#     role_name = 'ECRLifecyclePolicyRole'
#     policy_name = 'ECRLifecyclePolicyPermission'
#     image_name = 'xgb-clf-custom-training-container'
#
#     manager = RoleService(account_id, user_name, role_name, policy_name, image_name)
#     manager.create_role_and_policy()
#     manager.assume_role_and_put_lifecycle_policy()
