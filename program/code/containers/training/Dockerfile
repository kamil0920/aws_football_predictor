
FROM python:3.10-slim
RUN apt-get -y update && apt-get install -y --no-install-recommends     python3     build-essential     libssl-dev

COPY requirements.txt.txt .
RUN pip install --user --upgrade pip
RUN pip3 install -r requirements.txt.txt

COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py
                             