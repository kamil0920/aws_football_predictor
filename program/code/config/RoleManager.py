import json
import logging
import time

import boto3


class RoleManager:
    def __init__(self, account_id, user_name):
        self.account_id = account_id
        self.user_name = user_name
        self.session = boto3.Session()
        self.iam_client = self.session.client('iam')
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_role_and_policy(self, role_name, policy_name, region):
        """
        Orchestrates the creation of policies and role updates.
        """

        try:
            self.logger.info(f"Attempting to create role '{role_name}' in account '{self.account_id}' for user '{self.user_name}'.")

            user = self.iam_client.get_user(
                UserName=self.user_name,
            )

            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": ["ec2.amazonaws.com", "sagemaker.amazonaws.com"],
                                },
                                "Action": "sts:AssumeRole",
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": ["events.amazonaws.com"],
                                },
                                "Action": "sts:AssumeRole",
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "AWS": user['User']['Arn']
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ],
                    },
                ),
                Description="Football Project Role",
            )

            role_arn = response["Role"]["Arn"]

            self._attach_policies(role_name, policy_name, region)

            self.logger.info(f'Role "{role_name}" successfully created with ARN "{role_arn}".')
            return role_arn
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            self.logger.warning(f'Role "{role_name}" already exists.')
            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response["Role"]["Arn"]
            return role_arn
        except Exception as e:
            self.logger.error(f"An error occurred while creating the role: {str(e)}", exc_info=True)
            raise

    def _attach_policies(self, role_name, policy_name, region):
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=role_name,
        )

        ecr_policy_arn = self._create_custom_ecr_policy(policy_name, region)

        self.iam_client.attach_role_policy(
            PolicyArn=ecr_policy_arn,
            RoleName=role_name
        )

        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
            RoleName=role_name,
        )
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess",
            RoleName=role_name,
        )

    def _create_custom_ecr_policy(self, policy_name, region):
        custom_ecr_policy_name = 'CustomElasticContainerPolicy'
        try:
            custom_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:GetAuthorizationToken"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ecr:CreateRepository",
                            "ecr:DeleteRepository",
                            "ecr:TagResource",
                            "ecr:DescribeRepositories",
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:PutLifecyclePolicy",
                            "ecr:BatchGetImage",
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:PutImage",
                            "ecr:InitiateLayerUpload",
                            "ecr:UploadLayerPart",
                            "ecr:CompleteLayerUpload",
                        ],
                        "Resource": [
                            f"arn:aws:ecr:{region}:{self.account_id}:repository/sagemaker-processing-container",
                            f"arn:aws:ecr:{region}:{self.account_id}:repository/xgb-clf-training-container"
                        ]
                    }
                ]
            }

            self.iam_client.create_policy(
                PolicyName=custom_ecr_policy_name,
                PolicyDocument=json.dumps(custom_policy_document)
            )

            self.logger.debug(f'Policy "{custom_ecr_policy_name}" create.')
            time.sleep(5)

            return f"arn:aws:iam::{self.account_id}:policy/{custom_ecr_policy_name}"
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            policy_arn = f"arn:aws:iam::{self.account_id}:policy/{custom_ecr_policy_name}"
            self.logger.debug(f'Policy "{policy_name}" already exists.')
            return policy_arn

    def create_lambda_execution_role(self, lambda_role_name, bucket, region, pipeline_name, model_package_group):
        """
        Create the IAM role for Lambda with necessary permissions to interact with S3, DynamoDB, and SageMaker.
        """
        try:
            # Create the Lambda execution role
            role_response = self.iam_client.create_role(
                RoleName=lambda_role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "lambda.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "events.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "sagemaker.amazonaws.com"
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
                ),
                Description="Lambda Execution Role for S3, SageMaker, and DynamoDB Access",
            )
            time.sleep(5)

            lambda_role_arn = role_response["Role"]["Arn"]
            self.logger.info(f'Created role "{lambda_role_name}" with ARN "{lambda_role_arn}".')

            # Attach the custom policy for Lambda to interact with S3, DynamoDB, and SageMaker
            custom_policy_arn = self._create_custom_lambda_policy(bucket, region, pipeline_name, model_package_group, lambda_role_name)
            self.iam_client.attach_role_policy(
                PolicyArn=custom_policy_arn,
                RoleName=lambda_role_name
            )
            time.sleep(5)

            return lambda_role_arn

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            role_response = self.iam_client.get_role(RoleName=lambda_role_name)
            lambda_role_arn = role_response["Role"]["Arn"]
            self.logger.info(f'Role "{lambda_role_name}" already exists with ARN "{lambda_role_arn}".')
            return lambda_role_arn

        except Exception as e:
            self.logger.error(f"Failed to create Lambda role: {str(e)}")
            raise

    def _create_custom_lambda_policy(self, bucket, region, pipeline_name, model_package_group, lambda_role_name):
        """
        Create a custom IAM policy to allow Lambda to access S3, SageMaker, and manage Lambda functions.
        """
        custom_lambda_policy_name = 'LambdaCustomPolicy'

        try:
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "lambda:AddPermission",
                            "lambda:InvokeFunction",
                            "lambda:CreateFunction",
                            "lambda:UpdateFunctionCode",
                            "lambda:UpdateFunctionConfiguration"
                        ],
                        "Resource": [
                            "arn:aws:lambda:eu-north-1:284415450706:function:run_pipeline_fn",
                            "arn:aws:lambda:eu-north-1:284415450706:function:deployment_fn"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:CreateModel",
                            "sagemaker:CreateEndpointConfig",
                            "sagemaker:CreateEndpoint",
                            "sagemaker:UpdateEndpoint",
                            "sagemaker:ListModelPackages"
                        ],
                        "Resource": [
                            f"arn:aws:sagemaker:{region}:{self.account_id}:model/*",
                            f"arn:aws:sagemaker:{region}:{self.account_id}:endpoint/*",
                            f"arn:aws:sagemaker:{region}:{self.account_id}:endpoint-config/*",
                            f"arn:aws:sagemaker:{region}:{self.account_id}:model-package/{model_package_group}/*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": "sagemaker:ListEndpoints",
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": "sagemaker:StartPipelineExecution",
                        "Resource": f"arn:aws:sagemaker:{region}:{self.account_id}:pipeline/{pipeline_name}"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket",
                            "s3:GetObjectTagging",
                            "s3:PutObject"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{bucket}",
                            f"arn:aws:s3:::{bucket}/*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": f"arn:aws:iam::284415450706:role/{lambda_role_name}"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:CreateTable",
                            "dynamodb:DeleteItem"
                        ],
                        "Resource": f"arn:aws:dynamodb:{region}:{self.account_id}:table/S3FileUploadStatus"
                    },
                    {
                        "Effect": "Allow",
                        "Action": "logs:CreateLogGroup",
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogStream",
                            "logs:PutLogEvents"
                        ],
                        "Resource": "arn:aws:logs:*:*:*"
                    }
                ]
            }

            policy_response = self.iam_client.create_policy(
                PolicyName=custom_lambda_policy_name,
                PolicyDocument=json.dumps(policy_document)
            )

            self.logger.debug(f'Created custom policy "{custom_lambda_policy_name}".')
            time.sleep(5)

            return policy_response["Policy"]["Arn"]

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            policy_arn = f"arn:aws:iam::{self.account_id}:policy/{custom_lambda_policy_name}"
            self.logger.debug(f'Policy "{custom_lambda_policy_name}" already exists.')
            return policy_arn

        except Exception as e:
            self.logger.error(f"Failed to create custom Lambda policy: {str(e)}")
            raise
