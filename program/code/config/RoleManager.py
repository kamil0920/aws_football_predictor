import json
import logging

import boto3


class RoleManager:
    def __init__(self, account_id, user_name, role_name, policy_name, region):
        self.account_id = account_id
        self.user_name = user_name
        self.role_name = role_name
        self.policy_name = policy_name
        self.region = region
        self.session = boto3.Session()
        self.iam_client = self.session.client('iam')
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_role_and_policy(self):
        """
        Orchestrates the creation of policies and role updates.
        """

        try:
            self.logger.info(f"Attempting to create role '{self.role_name}' in account '{self.account_id}' for user '{self.user_name}'.")

            user = self.iam_client.get_user(
                UserName=self.user_name,
            )

            response = self.iam_client.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": ["lambda.amazonaws.com", "events.amazonaws.com", "ec2.amazonaws.com", "sagemaker.amazonaws.com"],
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

            self._attach_policies()

            self.logger.info(f'Role "{self.role_name}" successfully created with ARN "{role_arn}".')
            return role_arn
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            self.logger.warning(f'Role "{self.role_name}" already exists. Updating assume role policy.')

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

            self.iam_client.update_assume_role_policy(RoleName=self.role_name, PolicyDocument=json.dumps(trust_policy))

            response = self.iam_client.get_role(RoleName=self.role_name)
            role_arn = response["Role"]["Arn"]

            self._attach_policies()
            self.logger.info(f'Role "{self.role_name}" already exists. Policies updated.')
            return role_arn
        except Exception as e:
            self.logger.error(f"An error occurred while creating the role: {str(e)}", exc_info=True)
            raise

    def _attach_policies(self):
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            RoleName=self.role_name,
        )
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=self.role_name,
        )

        policy = self._create_custom_ecr_policy()

        self.iam_client.attach_role_policy(
            PolicyArn=policy['Arn'],
            RoleName=self.role_name
        )
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
            RoleName=self.role_name,
        )
        self.iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess",
            RoleName=self.role_name,
        )

    def _create_custom_ecr_policy(self):
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
                            "ecr:PutImage"
                        ],
                        "Resource": [
                            f"arn:aws:ecr:{self.region}:{self.account_id}:repository/sagemaker-processing-container",
                            f"arn:aws:ecr:{self.region}:{self.account_id}:repository/xgb-clf-training-container"
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
            self.logger.debug(f'Policy "{self.policy_name}" already exists.')
            return policy['Policy']
