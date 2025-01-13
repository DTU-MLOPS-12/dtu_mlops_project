import fastapi
from http import HTTPStatus

app = fastapi.FastAPI()


@app.get("/")
def home():
    """
    Root end-point (for health-check purposes)
    """
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}
