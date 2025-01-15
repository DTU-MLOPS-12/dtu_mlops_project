import functools
import random
import shutil
from locust import HttpUser, between, task
from urllib.request import urlopen
from tempfile import NamedTemporaryFile
from PIL import Image


@functools.cache
def _get_test_image() -> str:
    """
    Retrieves the test image from a URL, saves it in a temporary
    file and return the name of if.

    :returns: the name of the test image file.
    """
    test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    with urlopen(test_image_url) as image_data_response:
        with NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as tmp_image_file:
            shutil.copyfileobj(image_data_response, tmp_image_file)
    return tmp_image_file.name


class ApiUser(HttpUser):
    """
    A Locust user for testing the performance of the API backend service.
    """

    wait_time = between(1, 3)

    @task
    def get_root(self) -> None:
        """
        Task simulating a user visiting the root endpoint of the API.
        """
        self.client.get("/")

    @task
    def get_about(self) -> None:
        """
        Task simulating a user visiting the '/about' endpoint of the API.
        """
        self.client.get("/about/")

    @task
    def post_api_predict(self) -> None:
        """
        Task simulating a user requesting inference for an image.
        """
        with open(_get_test_image(), 'rb') as image_file:
            self.client.post("/api/predict/", files={'image_file': image_file})
