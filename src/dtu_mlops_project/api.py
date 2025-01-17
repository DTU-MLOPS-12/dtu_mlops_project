import functools
from contextlib import asynccontextmanager
from http import HTTPStatus
import json
import typing

import argparse
import fastapi
import torch
import timm
import wandb
from loguru import logger
from PIL import Image

# Allow loading weights using pickle
torch.serialization.add_safe_globals([argparse.Namespace])

MODEL_NAME = "dtu_mlops_project_grp_12/mlops_project/model"


@functools.cache
def get_labels() -> dict:
    """
    Loads and caches the labels for the corresponding class indices.
    """
    with open("src/dtu_mlops_project/imagenet-simple-labels.json") as f:
        labels = json.load(f)
    return labels


@functools.cache
def get_dummy_model() -> typing.Callable:
    """
    Initializes a 'dummy' model ('mobilenetv4') for temporary use.

    :returns: A callable which can run the initialized model.
    """
    model_name = "mobilenetv4_conv_small.e2400_r224_in1k"
    mobilenetv4_model = timm.create_model(model_name, pretrained=True)
    mobilenetv4_model = mobilenetv4_model.eval()

    data_config = timm.data.resolve_model_data_config(mobilenetv4_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    def _run_model(image, k: int = 5) -> tuple:
        """
        Runs the model on an image object with and computes the top 'k'
        probabilities and their associated class indices.

        :param image: The image to run the model on.
        :param k: The number of classes to return.
        :returns: a tuple of top 'k' class indices and their probabilities.
        """
        output = mobilenetv4_model(transforms(image).unsqueeze(0))

        probabilities, class_indices = torch.topk(output.softmax(dim=1) * 100, k=k)
        return probabilities, class_indices

    return _run_model


@functools.cache
def download_wandb_model(version: str = "latest") -> dict:
    """
    Downloads the model from W&B with a given version.

    :param version: the version of the model to use.
    :returns: A dictionary representing the model checkpoint.
    """
    run = wandb.init()
    model_artifact = run.use_artifact(f"{MODEL_NAME}:{version}", type="model")
    model_artifact.download()

    return torch.load(model_artifact.file())


@functools.cache
def get_wandb_model(version: str = "latest"):
    """
    Downloads and initializes the model from W&B with a
    given version.

    :param version: the version of the model to use.
    :returns: a loaded and initialized model.
    """
    wandb_model_checkpoint = download_wandb_model(version)
    num_classes = wandb_model_checkpoint["args"].num_classes

    model_name = "mobilenetv4_conv_small.e2400_r224_in1k"
    mobilenetv4_model = timm.create_model(model_name, num_classes=num_classes)

    mobilenetv4_model.load_state_dict(wandb_model_checkpoint["state_dict"])
    mobilenetv4_model = mobilenetv4_model.eval()

    data_config = timm.data.resolve_model_data_config(mobilenetv4_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    def _run_model(image, k: int = min(5, num_classes)) -> tuple:
        """
        Runs the model on an image object with and computes the top 'k'
        probabilities and their associated class indices.

        :param image: The image to run the model on.
        :param k: The number of classes to return.
        :returns: a tuple of top 'k' class indices and their probabilities.
        """
        output = mobilenetv4_model(transforms(image).unsqueeze(0))

        probabilities, class_indices = torch.topk(output.softmax(dim=1) * 100, k=k)
        return probabilities, class_indices

    return _run_model


dummy_model = get_dummy_model()
wandb_model = get_wandb_model()

IMAGE_MIME_TYPES = ("image/jpeg", "image/png", "image/svg", "image/svg+xml")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logger.info("API is up and running...")
    yield
    logger.info("Shutting down API")


# The FastAPI app object
app = fastapi.FastAPI(lifespan=lifespan)

# A base response for indicating OK requests.
HTTP_200_OK = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}


@app.get("/")
def home():
    """
    Root end-point (for health-check purposes)
    """
    logger.debug("Received a health-check request.")
    return HTTP_200_OK


@app.get("/about/")
def about():
    """
    A small 'about' section
    """
    return HTTP_200_OK | {
        "model_name": f"{MODEL_NAME}",
        "repository_url": "https://github.com/DTU-MLOPS-12/dtu_mlops_project",
    }


def check_image(image_file: fastapi.UploadFile):
    """
    Checks an image file for the correct MIME type.

    :param image_file: image file to check.
    """
    if image_file.content_type not in IMAGE_MIME_TYPES:
        logger.error(
            "Received an image file of with the wrong MIME type - "
            f"expected: {IMAGE_MIME_TYPES}, "
            f"got: {image_file.content_type}"
        )
        raise fastapi.HTTPException(
            status_code=400,
            detail=(
                f"Got an invalid file type: {image_file.content_type}. Acceptable file types are: {IMAGE_MIME_TYPES}"
            ),
        )


def preprocess_image(image_file: fastapi.UploadFile) -> Image:
    """
    Converts a FastAPI UploadFile to a PIL image and puts it
    in RGB mode.

    :param image_file: the image file to preprocess
    :returns: an opened processed PIL image
    """
    image = Image.open(image_file.file)
    if image.mode != "RGB":
        logger.debug("Image is not in RGB mode. Performing conversion")
        image = image.convert(mode="RGB")

    return image


def compute_results(probs: torch.Tensor, classes: torch.Tensor) -> dict:
    """
    Converts the probability and class indices tensors to a
    human-readable table.

    :param probs: probability tensor
    :param classes: class indices tensor
    :returns: a table relating class names to probabilities
    """
    labels = get_labels()
    return {labels[clz.item()]: prob.item() for prob, clz in zip(probs[0], classes[0])}


@app.post("/api/predict/")
def api_predict(image_file: fastapi.UploadFile):
    """
    API endpoint that receives an image file an runs the real
    model through it.
    """
    check_image(image_file)
    image = preprocess_image(image_file)

    probs, classes = wandb_model(image)
    logger.debug(f"Model computed probabilities: '{probs}' with corresponding class indices: '{classes}'")

    results = compute_results(probs, classes)
    logger.debug(f"Final results: '{results}'")

    return HTTP_200_OK | {
        "probabilities": results,
    }


@app.post("/api/predict/dummy/")
def api_predict_dummy(image_file: fastapi.UploadFile):
    """
    API endpoint that receives an image file an runs the dummy
    model through it.
    """
    check_image(image_file)
    image = preprocess_image(image_file)

    probs, classes = dummy_model(image)
    logger.debug(f"Dummy model computed probabilities: '{probs}' with corresponding class indices: '{classes}'")

    results = compute_results(probs, classes)
    logger.debug(f"Final results: '{results}'")

    return HTTP_200_OK | {
        "probabilities": results,
    }
