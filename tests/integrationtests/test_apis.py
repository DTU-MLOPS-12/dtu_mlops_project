import unittest
from fastapi.testclient import TestClient
from dtu_mlops_project.api import app


class TestIntegrationAPI(unittest.TestCase):
    def setUp(self):
        pass

    def test_api_is_alive(self):
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(200, response.status_code)

    def tearDown(self):
        pass
