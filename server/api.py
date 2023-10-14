from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.models import OpenAPI
from fastapi.openapi.utils import get_openapi

from PIL import Image

import mimetypes

#delete after implementing nn
import random


# This is the function that evaluates the image, feed your function with PIL image object
def get_prediction(image) :
    res = random.randint(0, 1)                      ### YOUR INFERENCE FUNCTION HERE ###
    if res == 1 :
        res = get_sub_prediction(image)
    return res

def get_sub_prediction(image):
    return random.randint(1, 4)

#===============================API========================================

allowed_types = ['image/jpeg', 'image/jpg', 'image/png']

app=FastAPI()

# This route accepts image from a request
# It responds with a class id
#   0 - Correct image
#   1 - Car is not on a brake stand
#   2 - Photo is taken from a screen
#   3 - Photo is taken from a screen and has been photoshoped
#   4 - Photo has been photoshoped
@app.post("/upload/")
async def upload_file(file: UploadFile) :
    mime, _ = mimetypes.guess_type(file.filename)

    if mime not in allowed_types :
        raise HTTPException(status_code=415, detail="Invalid file type. Only JPEG, JPG and PNG images are allowed")

    image = Image.open(file.file)
    res = get_prediction(image)
    return {"result": res}

#==========================DOCUMENTATION====================================
# just start the server and go to '/docs' for documentation

@app.get("/openapi.json", response_model=OpenAPI, include_in_schema=False)
async def get_open_api_endpoint() :
    return get_openapi(
        title="AutoEYE API",
        version="1.0.0",
        description="public api for AutoEYE",
        routes=app.routes
    )

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() :
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AutoEYE API")

@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html() :
    return get_redoc_html(openapi_url="/openapi.json", title="FastAPI ReDoc")
