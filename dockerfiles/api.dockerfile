# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
COPY pyproject.toml .
COPY src/dtu_mlops_project/ .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src/dtu_mlops_project/api:app", "--host", "0.0.0.0", "--port", "8000"]
