import os
import platform
import shutil
import subprocess
from pathlib import Path
import boto3
from tqdm import tqdm


class SageMakerInferenceContainerBuilder:
    def __init__(self, code_folder, image_name, local_mode=True):
        self.code_folder = Path(code_folder)
        self.image_name = image_name
        self.local_mode = local_mode
        self.inference_path = self.code_folder / "containers" / "inference"

    def setup_directories(self):
        self.inference_path.mkdir(parents=True, exist_ok=True)

    def create_requirements(self):
        requirements = """sagemaker-inference
numpy
pandas
scikit-learn==1.2.1
xgboost==1.5.0
fastapi
uvicorn
flask"""

        with open(self.inference_path / 'requirements.txt', 'w') as f:
            f.write(requirements)

    def create_dockerfile(self):
        dockerfile_contents = """FROM python:3.10-slim

COPY inference.py /opt/
COPY requirements.txt /opt/
COPY run_server.sh /opt/serve

RUN mkdir -p /pipeline
RUN mkdir -p /classes

RUN chmod +x /opt/serve
RUN chmod +x /classes

WORKDIR /opt

RUN pip install -r requirements.txt

ENV PATH="/opt/:${PATH}"
"""

        with open(self.inference_path / 'Dockerfile', 'w') as f:
            f.write(dockerfile_contents)

    def build_image(self):
        platform_arg = "--platform='linux/amd64'" if platform.system() != "Windows" else ""
        build_command = f"docker build {platform_arg} -t {self.image_name} {self.inference_path}" if not self.local_mode else f"docker build -t {self.image_name} {self.inference_path}"

        try:
            subprocess.run(build_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")  # Print standard error (stderr)
            raise

    def push_to_ecr(self):
        if not self.local_mode:
            print("Pushing Docker image to ECR...")
            client = boto3.client("sts")
            account_id = client.get_caller_identity().get("Account")
            region = boto3.session.Session().region_name
            repository_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{self.image_name}:latest"

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
            with tqdm(total=100, desc="Pushing image") as pbar:
                push_command = f"docker push {repository_uri}"
                process = subprocess.Popen(push_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                while process.poll() is None:
                    pbar.update(10)
                process.communicate()
                pbar.update(100 - pbar.n)
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, push_command)
            print("Docker image pushed to ECR successfully.")
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