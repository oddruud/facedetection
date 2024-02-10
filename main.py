from face_detect import *
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import (
    FileResponse,
    Response,
    StreamingResponse
)

import io
from PIL import Image
from typing import Annotated

app = FastAPI()

@app.post(
    "/find_face"
)
async def find_face_endpoint(
    image: Annotated[bytes, File()],
):
    imageStream = io.BytesIO(image)
    imageFile = Image.open(imageStream).convert('RGB')
    frame_filename = "frame.jpg"
    imageFile.save(frame_filename)
    result = detect_face(frame_filename)
    
    return {"person": result}

