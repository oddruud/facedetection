from face_detect import *
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import (
    FileResponse,
    Response,
    StreamingResponse
)
import numpy as np
import io
from PIL import Image
from typing import Annotated

app = FastAPI()
active_face = None
@app.post(
    "/find_face"
)
async def find_face_endpoint(
    image: Annotated[bytes, File()],
):
    imageStream = io.BytesIO(image)
    imageFile = Image.open(imageStream).convert('RGB')
    person_id, bounding_box = detect_face(imageFile)
    active_face = {"person": person_id, "bounding_box": {"x": bounding_box["x"], "y": bounding_box["y"], "width": bounding_box["width"], "height": bounding_box["height"]}}
    return active_face


@app.get(
    "/get_active_face"
)
async def get_active_face_endpoint():
    return active_face