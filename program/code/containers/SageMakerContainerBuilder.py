import os
import shutil
import subprocess
from pathlib import Path
import boto3
import platform
import os


class SageMakerContainerBuilder:
    def __init__(self, code_folder, image_name, local_mode=True):
        self.code_folder = Path(code_folder)
        self.image_name = image_name
        self.local_mode = local_mode
        self.training_path = self.code_folder / "containers" / "training"

    def setup_directories(self):
        self.training_path.mkdir(parents=True, exist_ok=True)

    def create_requirements(self):
        requirements = """sagemaker-training
pandas==2.0.2
numpy
xgboost-cpu==2.1.1
scikit-learn==1.3.2
scipy
sagemaker
smdebug"""
        with open(self.training_path / 'requirements.txt', 'w') as f:
            f.write(requirements)

    def create_dockerfile(self):
        dockerfile_contents = """
FROM python:3.10-slim
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    python3 \
    build-essential \
    libssl-dev

COPY requirements.txt .
RUN pip install --user --upgrade pip
RUN pip3 install -r requirements.txt

COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py
                             """
        with open(self.training_path / 'Dockerfile', 'w') as f:
            f.write(dockerfile_contents)

    def build_image(self):
        platform_arg = "--platform='linux/amd64'" if platform.system() != "Windows" else ""
        build_command = f"docker build {platform_arg} -t {self.image_name} {self.training_path}" if not self.local_mode else f"docker build -t {self.image_name} {self.training_path}"

        try:
            result = subprocess.run(build_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
            print(result.stdout)
            print("Docker image built successfully.")
        except subprocess.CalledProcessError as e:
            print("An error occurred while building the Docker image.")
            print("Command:", e.cmd)
            print("Return code:", e.returncode)
            print("Output:", e.output)
            print("Error message:", e.stderr)

    def push_to_ecr(self):
        if not self.local_mode:
            client = boto3.client("sts")
            account_id = os.environ["ACCOUNT_ID"]
            region = os.environ["AWS_REGION"]
            print(f'account_id: {account_id}')
            print(f'region: {region}')
            uri_suffix = "amazonaws.com.cn" if region in ["cn-north-1", "cn-northwest-1"] else "amazonaws.com"
            repository_uri = f"{account_id}.dkr.ecr.{region}.{uri_suffix}/{self.image_name}:latest"

            ecr_client = boto3.client('ecr')
            try:
                ecr_client.describe_repositories(repositoryNames=[self.image_name])
            except ecr_client.exceptions.RepositoryNotFoundException:
                ecr_client.create_repository(repositoryName=self.image_name)

            login_password = subprocess.getoutput(f"aws ecr get-login-password --region {region}")
            subprocess.run(
                f"echo {login_password} | aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com",
                shell=True, check=True, capture_output=True, text=True, encoding='utf-8'
            )

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
