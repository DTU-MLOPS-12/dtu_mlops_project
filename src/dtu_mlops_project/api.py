from contextlib import asynccontextmanager

import fastapi
from loguru import logger
from http import HTTPStatus
from PIL import Image

from . import model

dummy_model = model.get_dummy_model()

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logger.info("API is up and running...")
    yield
    logger.info("Shutting down API")


# The FastAPI app object
app = fastapi.FastAPI(lifespan=lifespan)

# A base response for indicating OK requests.
HTTP_200_OK = {
    'message': HTTPStatus.OK.phrase,
    'status-code': HTTPStatus.OK
}


@app.get('/')
def home():
    """
    Root end-point (for health-check purposes)
    """
    logger.debug("Received a health-check request.")
    return HTTP_200_OK


@app.get('/about/')
def about():
    """
    A small 'about' section
    """
    return HTTP_200_OK | {
        'model_name': 'mobilenetv4_conv_small.e2400_r224_in1k',
        'base_model_url': '',
        'repository_url': 'https://github.com/HackTheOxidation/dtu_mlops_project',
        'dataset_name': '',
        'dataset_url': ''
    }


@app.post("/api/predict/")
def api_predict(image_file: fastapi.UploadFile | None = None):
    """
    API endpoint that receives an image file an runs the dummy
    model through it.
    """
    if not image_file:
        logger.error("Received a request without an image file/with an invalid file")
        raise fastapi.HTTPException(
            status_code=400,
            detail='Expected a valid image file'
        )

    image = Image.open(image_file.file)
    if image.mode != 'RGB':
        logger.debug("Image is not in RGB mode. Performing conversion")
        image = image.convert(mode='RGB')

    probs, classes = dummy_model(image)
    logger.debug(f"Dummy model computed probabilities: '{probs}' "
                 f"with corresponding class indices: '{classes}'")

    return HTTP_200_OK | {
        'probabilities': probs,
        'classification_indices': classes,
    }
