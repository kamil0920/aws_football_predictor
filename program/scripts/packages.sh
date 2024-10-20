
#!/bin/bash
# This script upgrades the packages on a SageMaker 
# Studio Kernel Application.

set -eux

pip install -q --upgrade pip
pip install -q --upgrade awscli boto3
pip install -q --upgrade scikit-learn==1.3.2
pip install -q --upgrade PyYAML==6.0
pip install -q --upgrade sagemaker
pip install -q --upgrade flask
pip install -q --upgrade ipytest
