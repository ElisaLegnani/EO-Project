import torch
from PIL import Image

from satellite_classifier import predict


class DummyModel(torch.nn.Module):
    """A small fake model for testing predictions."""

    def forward(self, images):
        return torch.tensor(
            [[5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            device=images.device,
        )


def test_prediction_output(monkeypatch):
    # Replace checkpoint loading with our fake model
    monkeypatch.setattr(
        predict,
        "load_model",
        lambda: (DummyModel(), torch.device("cpu")),
    )

    image = Image.new("RGB", (64, 64))

    result = predict.predict_image(image)

    assert "predicted_class" in result
    assert "confidence" in result
    assert result["predicted_class"] == predict.CLASS_NAMES[0]
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1
