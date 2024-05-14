import os
import shutil
import subprocess
from pathlib import Path
import boto3


class SageMakerContainerBuilder:
    def __init__(self, code_folder, image_name, local_mode=True):
        self.code_folder = Path(code_folder)
        self.image_name = image_name
        self.local_mode = local_mode
        self.training_path = self.code_folder / "containers" / "training"

    def setup_directories(self):
        self.training_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.code_folder / "train.py",
            self.training_path / "train.py",
        )

    def create_requirements(self):
        requirements = """sagemaker-training
xgboost
pandas
numpy
scikit-learn"""

        with open(self.training_path / 'requirements.txt', 'w') as f:
            f.write(requirements)

    def create_dockerfile(self):
        dockerfile_contents = """FROM python:3.10-slim
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    python3 \
    build-essential \
    libssl-dev

COPY requirements.txt .
RUN pip install --user --upgrade pip
RUN pip3 install -r requirements.txt

COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py"""

        with open(self.training_path / 'Dockerfile', 'w') as f:
            f.write(dockerfile_contents)

    def build_image(self):
        build_command = f"docker build --platform='linux/amd64' -t {self.image_name} {self.training_path}" if not self.local_mode else f"docker build -t {self.image_name} {self.training_path}"
        subprocess.run(build_command, shell=True, check=True)

    def push_to_ecr(self):
        if not self.local_mode:
            client = boto3.client("sts")
            account_id = client.get_caller_identity().get("Account")
            region = boto3.session.Session().region_name
            uri_suffix = "amazonaws.com.cn" if region in ["cn-north-1", "cn-northwest-1"] else "amazonaws.com"
            repository_uri = f"{account_id}.dkr.ecr.{region}.{uri_suffix}/{self.image_name}:latest"

            # Create ECR repository if it does not exist
            ecr_client = boto3.client('ecr')
            try:
                ecr_client.describe_repositories(repositoryNames=[self.image_name])
            except ecr_client.exceptions.RepositoryNotFoundException:
                ecr_client.create_repository(repositoryName=self.image_name)

            # Login to ECR
            login_password = subprocess.getoutput(f"aws ecr get-login-password --region {region}")
            subprocess.run(
                f"echo {login_password} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.{uri_suffix}",
                shell=True, check=True)

            # Tag and push the image
            subprocess.run(f"docker tag {self.image_name} {repository_uri}", shell=True, check=True)
            subprocess.run(f"docker push {repository_uri}", shell=True, check=True)

    def get_image_uri(self):
        client = boto3.client("sts")
        account_id = client.get_caller_identity().get("Account")
        region = boto3.session.Session().region_name
        uri_suffix = "amazonaws.com.cn" if region in ["cn-north-1", "cn-northwest-1"] else "amazonaws.com"
        return f"{account_id}.dkr.ecr.{region}.{uri_suffix}/{self.image_name}:latest"

    def build_and_push(self):
        self.setup_directories()
        self.create_requirements()
        self.create_dockerfile()
        self.build_image()
        if not self.local_mode:
            self.push_to_ecr()
        return self.get_image_uri()
