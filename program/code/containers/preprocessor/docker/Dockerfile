
FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --user --upgrade pip
RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
