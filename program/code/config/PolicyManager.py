import json
import logging

import boto3
from botocore.exceptions import ClientError


class PolicyManager:
    def __init__(self, account_id, user_name):
        self.account_id = account_id
        self.user_name = user_name
        self.session = boto3.Session()
        self.logger = logging.getLogger(__name__)

    def create_permissions_policy(self, iam_client, policy_name):
        """
        Creates a permissions policy and returns its ARN.
        """

        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                # {
                #     "Sid": "ECR",
                #     "Effect": "Allow",
                #     "Action": [
                #         "ecr:CreateRepository",
                #         "ecr:DescribeRepositories",
                #         "ecr:DeleteRepository",
                #         "ecr:ListImages",
                #         "ecr:PutImage",
                #         "ecr:InitiateLayerUpload",
                #         "ecr:UploadLayerPart",
                #         "ecr:CompleteLayerUpload",
                #         "ecr:BatchCheckLayerAvailability",
                #         "ecr:TagResource",
                #         "ecr:PutLifecyclePolicy",
                #         "ecr:GetAuthorizationToken"
                #     ],
                #     "Resource": "*"
                # },
                # {
                #     "Sid": "S3",
                #     "Effect": "Allow",
                #     "Action": [
                #         "s3:CreateBucket",
                #         "s3:GetBucketLocation",
                #         "s3:PutObject",
                #         "s3:GetObject",
                #         "s3:DeleteObject"
                #     ],
                #     "Resource": "*"
                # },
                # {
                #     "Sid": "S3ListBucket",
                #     "Effect": "Allow",
                #     "Action": "s3:ListBucket",
                #     "Resource": "*"
                # },
                # {
                #     "Sid": "EventBridge",
                #     "Effect": "Allow",
                #     "Action": [
                #         "events:PutRule",
                #         "events:PutTargets"
                #     ],
                #     "Resource": "*"
                # },
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
                # {
                #     "Sid": "Lambda",
                #     "Effect": "Allow",
                #     "Action": [
                #         "lambda:CreateFunction",
                #         "lambda:DeleteFunction",
                #         "lambda:InvokeFunctionUrl",
                #         "lambda:InvokeFunction",
                #         "lambda:UpdateFunctionCode",
                #         "lambda:InvokeAsync",
                #         "lambda:AddPermission",
                #         "lambda:RemovePermission"
                #     ],
                #     "Resource": "*"
                # },
                # {
                #     "Sid": "SageMaker",
                #     "Effect": "Allow",
                #     "Action": [
                #         "sagemaker:UpdateDomain",
                #         "sagemaker:UpdateUserProfile"
                #     ],
                #     "Resource": "*"
                # },
                # {
                #     "Sid": "CloudWatch",
                #     "Effect": "Allow",
                #     "Action": [
                #         "cloudwatch:PutMetricData",
                #         "cloudwatch:GetMetricData",
                #         "cloudwatch:DescribeAlarmsForMetric",
                #         "logs:CreateLogStream",
                #         "logs:PutLogEvents",
                #         "logs:CreateLogGroup",
                #         "logs:DescribeLogStreams"
                #     ],
                #     "Resource": "*"
                # },
                {
                    "Sid": "IAM",
                    "Effect": "Allow",
                    "Action": "iam:UpdateAssumeRolePolicy",
                    "Resource": "*"
                }
            ]
        }

        policy_arn = f"arn:aws:iam::{self.account_id}:policy/{policy_name}"

        try:
            response = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(permissions_policy),
                Description='Policy for football match predictions project.'
            )
            self.logger.info(f"Policy {policy_name} created successfully.")
            return response['Policy']['Arn']
        except iam_client.exceptions.EntityAlreadyExistsException:
            self.logger.warning(f"Policy {policy_name} already exists.")
            return policy_arn
        except ClientError as e:
            self.logger.error(f"Error creating policy: {e}")
            raise


    def attach_policy_to_role(self, iam_client, policy_arn, role_name):
        """
        Attaches a policy to an IAM role.
        """
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            self.logger.info(f"Policy {policy_arn} attached to role {role_name} successfully.")
        except ClientError as e:
            self.logger.error(f"Error attaching policy: {e}")
            raise

    def attach_inline_policy_to_user(self, iam_client, role_name):
        """
        Attaches an inline policy to an IAM user to allow role assumption.
        """
        inline_policy_name = 'FootballMatchPolicy'

        inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "sts:AssumeRole",
                    "Resource": f"arn:aws:iam::{self.account_id}:role/{role_name}"
                }
            ]
        }

        try:
            iam_client.put_user_policy(
                UserName=self.user_name,
                PolicyName=inline_policy_name,
                PolicyDocument=json.dumps(inline_policy)
            )
            self.logger.info(f"Inline policy {inline_policy_name} attached to user {self.user_name} successfully.")
        except ClientError as e:
            self.logger.error(f"Error attaching inline policy: {e}")
            raise
