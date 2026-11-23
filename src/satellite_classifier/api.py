from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms

from satellite_classifier.predict import (
    CLASS_NAMES,
    load_model,
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the API starts.
    model, device = load_model()
    app.state.model = model
    app.state.device = device

    yield


app = FastAPI(
    title="EuroSAT Satellite Classifier",
    description="Classifies RGB satellite images into ten land-cover classes.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a JPEG, PNG or WebP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        # Verify that the uploaded content is a valid image.
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()

        # Reopen after verification to perform preprocessing.
        with Image.open(BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    model = request.app.state.model
    device = request.app.state.device

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

        probability_values = probabilities.squeeze(0).cpu().tolist()

    class_probabilities = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(CLASS_NAMES, probability_values)
    }

    return PredictionResponse(
        predicted_class=CLASS_NAMES[predicted_index.item()],
        confidence=round(confidence.item(), 4),
        probabilities=class_probabilities,
    )
