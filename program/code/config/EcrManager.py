import json
import logging
from botocore.exceptions import ClientError


class EcrManager:
    def __init__(self, ecr_client):
        self.ecr_client = ecr_client
        self.logger = logging.getLogger(__name__)

    def create_repository(self, repository_name):
        """
        Creates an ECR repository if it does not exist.
        """
        try:
            response = self.ecr_client.describe_repositories(
                repositoryNames=[repository_name]
            )
            repository = response['repositories'][0]
            self.logger.info(f"Repository already exists: {repository['repositoryUri']}")
            return repository
        except self.ecr_client.exceptions.RepositoryNotFoundException:
            try:
                response = self.ecr_client.create_repository(
                    repositoryName=repository_name,
                    tags=[{'Key': 'Project', 'Value': 'home-win-match-predictor'}]
                )
                repository = response['repository']
                self.logger.info(f"Repository created successfully: {repository['repositoryUri']}")
                return repository
            except ClientError as e:
                self.logger.error(f"Error creating repository: {e}")
                raise
        except ClientError as e:
            self.logger.error(f"Error describing repository: {e}")
            raise

    def put_lifecycle_policy(self, repository_name, lifecycle_policy):
        """
        Attaches a lifecycle policy to an ECR repository.
        """
        try:
            self.ecr_client.put_lifecycle_policy(
                repositoryName=repository_name,
                lifecyclePolicyText=json.dumps(lifecycle_policy)
            )
            self.logger.info(f"Lifecycle policy attached to repository {repository_name} successfully.")
        except ClientError as e:
            self.logger.error(f"Error putting lifecycle policy: {e}")
            raise