import unittest
import shutil
import tempfile
from urllib.request import urlopen
from PIL import Image
import httpx
from fastapi.testclient import TestClient
from dtu_mlops_project.api import app


class TestIntegrationAPI(unittest.TestCase):
    def setUp(self):
        self.test_image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

    def test_api_is_alive(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(200, response.status_code)

    def test_api_about_page(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(200, response.status_code)

    def test_api_predict_no_image_gives_bad_request(self):
        with TestClient(app) as client:
            response = client.post("/api/predict/")
            self.assertEqual(422, response.status_code)

    def test_api_predict_other_file_gives_bad_request(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            with TestClient(app) as client:
                files = {'image_file': open(tmp_file.name, 'rb')}
                response = client.post("/api/predict/", files=files)
                self.assertEqual(400, response.status_code)

    def test_api_predict_valid_image_is_ok(self):
        with urlopen(self.test_image_url) as image_data_response:
            with tempfile.NamedTemporaryFile(mode="wb",
                                             delete=False,
                                             suffix=".png") as tmp_image_file:
                shutil.copyfileobj(image_data_response, tmp_image_file)

        files = {'image_file': open(tmp_image_file.name, 'rb')}

        with TestClient(app) as client:
            response = client.post("/api/predict/", files=files)
            print(response.text)
            self.assertEqual(200, response.status_code)
            self.assertEqual("OK", response.json()['message'])

    def tearDown(self):
        pass
