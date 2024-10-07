import json
import logging

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
            self.logger.warning(f'Role "{role_name}" already exists. Updating assume role policy.')

            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": f"arn:aws:iam::{self.account_id}:user/{self.user_name}"
                        },
                        "Action": "sts:AssumeRole"
                    },
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": ["lambda.amazonaws.com", "events.amazonaws.com", "ec2.amazonaws.com", "sagemaker.amazonaws.com"],
                        },
                        "Action": "sts:AssumeRole",
                    },
                ]
            }

            self.iam_client.update_assume_role_policy(RoleName=role_name, PolicyDocument=json.dumps(trust_policy))

            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response["Role"]["Arn"]

            self._attach_policies(role_name, policy_name, region)
            self.logger.info(f'Role "{role_name}" already exists. Policies updated.')
            return role_arn
        except Exception as e:
            self.logger.error(f"An error occurred while creating the role: {str(e)}", exc_info=True)
            raise

    def create_lambda_role(self):
        import json
        lambda_role_name = "lambda-deployment-role"
        lambda_role_arn = None

        try:
            response = self.iam_client.create_role(
                RoleName=lambda_role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": ["lambda.amazonaws.com", "events.amazonaws.com"],
                                },
                                "Action": "sts:AssumeRole",
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "AWS": f"arn:aws:iam::{self.account_id}:user/{self.user_name}"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ],
                    },
                ),
                Description="Lambda Endpoint Deployment",
            )

            lambda_role_arn = response["Role"]["Arn"]

            self.iam_client.attach_role_policy(
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                RoleName=lambda_role_name,
            )

            self.iam_client.attach_role_policy(
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                RoleName=lambda_role_name,
            )

            print(f'Role "{lambda_role_name}" created with ARN "{lambda_role_arn}".')
            return lambda_role_arn
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            response = boto3.client("iam").get_role(RoleName=lambda_role_name)
            lambda_role_arn = response["Role"]["Arn"]
            print(f'Role "{lambda_role_name}" already exists with ARN "{lambda_role_arn}".')
            return lambda_role_arn

    def _attach_policies(self, role_name, policy_name, region):
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            RoleName=role_name,
        )
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=role_name,
        )

        ecr_policy = self._create_custom_ecr_policy(policy_name, region)

        self.iam_client.attach_role_policy(
            PolicyArn=ecr_policy['Arn'],
            RoleName=role_name
        )

        lambda_policy = self._create_custom_lambda_policy(role_name, region)

        self.iam_client.attach_role_policy(
            PolicyArn=lambda_policy['Arn'],
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

    def _create_custom_lambda_policy(self, role_name, policy_name):
        custom_lambda_policy_name = 'LambdaCustomPolicy'

        try:
            policy = {
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
                        "Resource": "arn:aws:lambda:eu-north-1:284415450706:function:deployment_fn"
                    }
                ]
            }

            response = self.iam_client.create_policy(
                PolicyName=custom_lambda_policy_name,
                PolicyDocument=json.dumps(policy)
            )

            policy = self.iam_client.attach_role_policy(
                PolicyArn=response['Policy']['Arn'],
                RoleName=role_name,
            )

            return policy
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            policy_arn = f"arn:aws:iam::{self.account_id}:policy/{custom_lambda_policy_name}"
            policy = self.iam_client.get_policy(PolicyArn=policy_arn)
            self.logger.debug(f'Policy "{policy_name}" already exists.')
            return policy['Policy']

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

            policy = self.iam_client.create_policy(
                PolicyName=custom_ecr_policy_name,
                PolicyDocument=json.dumps(custom_policy_document)
            )
            return policy

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            policy_arn = f"arn:aws:iam::{self.account_id}:policy/{custom_ecr_policy_name}"
            policy = self.iam_client.get_policy(PolicyArn=policy_arn)
            self.logger.debug(f'Policy "{policy_name}" already exists.')
            return policy['Policy']
