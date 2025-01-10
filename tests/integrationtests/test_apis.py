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
            response = client.post("/api/predict")
            self.assertEqual(400, response.status_code)

    def test_api_predict_valid_image_is_ok(self):
        with urlopen(self.test_image_url) as image_data_response:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_image_file:
                shutil.copyfileobj(image_data_response, tmp_image_file)

        files = {'upload-file': open(tmp_image_file.name, 'rb')}

        with TestClient(app) as client:
            response = client.post("/api/predict/", files=files)
            self.assertEqual(200, response.status_code)

    def tearDown(self):
        pass
