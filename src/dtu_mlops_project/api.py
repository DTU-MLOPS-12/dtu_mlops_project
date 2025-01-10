from contextlib import asynccontextmanager
import fastapi
from http import HTTPStatus
from PIL import Image

from . import model


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("API is up and running...")
    yield
    print("Shutting down API")


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
    return HTTP_200_OK


@app.get('/about')
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


@app.post("/api/predict")
def api_predict(image_file: fastapi.UploadFile | None = None):
    """
    API endpoint that receives an image file an runs the dummy
    model through it.
    """
    if not image_file:
        return {
            'message': HTTPStatus.BAD_REQUEST.phrase,
            'status': HTTPStatus.BAD_REQUEST,
        }

    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert(mode='RGB')

    probs, classes = model.dummy_model(image)

    return HTTP_200_OK | {
        'probabilities': probs,
        'classification_indices': classes,
    }

