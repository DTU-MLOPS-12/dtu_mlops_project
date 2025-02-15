import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "dtu_mlops_project"
PYTHON_VERSION = "3.12"

# Setup commands


@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={
            PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands


@task
def preprocess_data(
    ctx: Context,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py --raw-dir {raw_dir} --processed-dir {processed_dir}",
            echo=True, pty=not WINDOWS)


@task
def train(ctx: Context,
          lr: float = 1e-3,
          batch_size: int = 32,
          epochs: int = 10) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --lr {lr} --batch-size {batch_size} --epochs {epochs}",
            echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context,
             model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate model."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py --model-checkpoint {model_checkpoint}", echo=True, pty=not WINDOWS)


@task
def visualize(ctx: Context,
              model_checkpoint: str = "models/model.pth", 
              figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py --model-checkpoint {model_checkpoint} --figure-name {figure_name}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run("docker build -t train:latest . -f dockerfiles/train.dockerfile",
            echo=True, pty=not WINDOWS)
    ctx.run("docker build -t api:latest . -f dockerfiles/api.dockerfile",
            echo=True, pty=not WINDOWS)


@task
def runserver(ctx: Context, port: int = 8000) -> None:
    """
    Runs and serves the backend API.

    :param ctx: (Invoke context)
    :param port: socket port to run the backend on (default: 8000)
    """
    ctx.run(f"uvicorn --reload --port {port} dtu_mlops_project.api:app")
