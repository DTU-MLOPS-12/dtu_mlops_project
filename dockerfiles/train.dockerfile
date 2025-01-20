# Export credentials (test locally)
# export GCP_SA_KEY_DVC=$(cat gcloud_credentials_base64.txt)

# Build with secret
# docker build --secret id=GCP_SA_KEY_DVC -f dockerfiles/train.dockerfile -t train:latest .

# Debug
# docker run -it --entrypoint bash train:latest

FROM python:3.12-slim

# Add ARG for build-time variables
ARG WANDB_API_KEY

# Set as environment variable
ENV WANDB_API_KEY=$WANDB_API_KEY

# Add validation step
RUN if [ -z "$WANDB_API_KEY" ]; then \
    echo "WANDB_API_KEY is not set" && exit 1; \
    else echo "WANDB_API_KEY is set"; \
    fi
    
# Install system dependencies including Google Cloud SDK
RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential \
    gcc \
    curl \
    gnupg \
    lsb-release && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt update && \
    apt install -y google-cloud-sdk && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN mkdir -p /data/processed
RUN mkdir /output

# Add gcloud auth step using GitHub secret and copy dataset
RUN --mount=type=secret,id=GCP_SA_KEY_DVC,mode=0444 \
    cat /run/secrets/GCP_SA_KEY_DVC | tr -d '\n' | base64 -d > /tmp/gcloud-credentials.json && \
    chmod 400 /tmp/gcloud-credentials.json && \
    gcloud auth activate-service-account --key-file=/tmp/gcloud-credentials.json && \
    gsutil -m cp -r gs://mlops_grp_12_data_bucket_public/data/processed/timm-imagenet-1k-wds-subset /data/processed/ && \
    rm -f /tmp/gcloud-credentials.json

ENTRYPOINT ["python", "-u", "src/dtu_mlops_project/train.py"]
