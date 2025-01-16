import functools
from contextlib import asynccontextmanager
from http import HTTPStatus
import json
import typing

import fastapi
import torch
import timm
from loguru import logger
from PIL import Image


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


dummy_model = get_dummy_model()

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
        "model_name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "base_model_url": "",
        "repository_url": "https://github.com/HackTheOxidation/dtu_mlops_project",
        "dataset_name": "",
        "dataset_url": "",
    }


@app.post("/api/predict/")
def api_predict(image_file: fastapi.UploadFile):
    """
    API endpoint that receives an image file an runs the dummy
    model through it.
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

    image = Image.open(image_file.file)
    if image.mode != "RGB":
        logger.debug("Image is not in RGB mode. Performing conversion")
        image = image.convert(mode="RGB")

    probs, classes = dummy_model(image)
    logger.debug(f"Dummy model computed probabilities: '{probs}' with corresponding class indices: '{classes}'")

    labels = get_labels()

    results = {labels[clz.item()]: prob.item() for prob, clz in zip(probs[0], classes[0])}
    logger.debug(f"Final results: '{results}'")

    return HTTP_200_OK | {
        "probabilities": results,
    }
