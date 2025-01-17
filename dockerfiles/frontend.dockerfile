# Base image
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deps/requirements_frontend.txt requirements.txt
COPY pyproject.toml .
COPY src/dtu_mlops_project/frontend.py .

COPY requirements_dev.txt /app/requirements_dev.txt
COPY frontend.py /app/frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run", "frontend.py", "--server.port", "8000"]