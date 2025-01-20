# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY deps/requirements_api.txt requirements.txt
COPY pyproject.toml .
COPY src/dtu_mlops_project/api.py .
COPY src/dtu_mlops_project/imagenet-simple-labels.json ./src/dtu_mlops_project/

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
