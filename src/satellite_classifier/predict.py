from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from satellite_classifier.model import SatelliteCNN

CHECKPOINT_PATH = Path("models/best_model.pth")

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    device = get_device()

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = SatelliteCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def predict_image(image: Image.Image):
    model, device = load_model()

    image = image.convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

    predicted_class = CLASS_NAMES[predicted_index.item()]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":
    image_path = Path("data/EuroSAT_RGB/Forest")

    sample_image = next(image_path.glob("*.jpg"))
    with Image.open(sample_image) as image:
        result = predict_image(image)

    print(result)
