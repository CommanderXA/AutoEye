import logging
import os

from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.models import OpenAPI
from fastapi.openapi.utils import get_openapi

import mimetypes

import torch
from torchvision.transforms import v2

from PIL import Image
from hydra import initialize, compose

from nn.autoeye.config import Config
from nn.autoeye.model import AutoEye

# configuration
initialize(version_base=None, config_path="./nn/conf")
cfg = compose(config_name="config")
log = logging.getLogger(__name__)
Config.setup(cfg, log, train=False)

# get model
model = AutoEye()
model.eval()

if cfg.hyper.pretrained and os.path.exists(f"{Config.model_path[:-3]}_best.pt"):
    checkpoint = torch.load(f"{Config.model_path[:-3]}_best.pt")
    model.load(checkpoint)
    Config.set_trained_epochs(checkpoint["epochs"])

logging.info(
    f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
)


# This is the function that evaluates the image, feed your function with PIL image object
transforms = v2.Compose(
    [
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_prediction(image):
    image = transforms(image).unsqueeze(0)
    logits = model(image)
    logits = torch.argmax(logits, 1).item()
    return logits


# ===============================API========================================

allowed_types = ["image/jpeg", "image/jpg", "image/png"]

app = FastAPI()


# This route accepts image from a request
# It responds with a class id
#   0 - Correct image
#   1 - Car is not on a brake stand
#   2 - Photo is taken from a screen
#   3 - Photo is taken from a screen and has been photoshoped
#   4 - Photo has been photoshoped
@app.post("/upload/")
async def upload_file(file: UploadFile):
    mime, _ = mimetypes.guess_type(file.filename)

    if mime not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Invalid file type. Only JPEG, JPG and PNG images are allowed",
        )

    image = Image.open(file.file).convert("RGB")
    res = get_prediction(image)
    return {"result": res}


# ==========================DOCUMENTATION====================================
# just start the server and go to '/docs' for documentation


@app.get("/openapi.json", response_model=OpenAPI, include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title="AutoEYE API",
        version="1.0.0",
        description="public api for AutoEYE",
        routes=app.routes,
    )


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AutoEYE API")


@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    return get_redoc_html(openapi_url="/openapi.json", title="FastAPI ReDoc")
