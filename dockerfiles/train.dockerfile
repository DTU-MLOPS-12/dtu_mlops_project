# Export credentials (test locally)
# export GCP_SA_KEY_DVC=$(cat gcloud_credentials_base64.txt)

# Build with secret
# docker build --secret id=GCP_SA_KEY_DVC -f dockerfiles/train.dockerfile -t train:latest .

# Debug
# docker run -it --entrypoint bash train:latest

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Add ARG for build-time variables
ARG WANDB_API_KEY

# Set as environment variable
ENV WANDB_API_KEY=$WANDB_API_KEY

# Install system dependencies including Google Cloud SDK and Python/pip
RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential \
    gcc \
    curl \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3.12-venv \
    python-is-python3 \
    python3-full && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt update && \
    apt install -y google-cloud-sdk && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY configs/ configs/

WORKDIR /
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
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
