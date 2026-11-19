import torch

from satellite_classifier.model import SatelliteCNN


def test_model_output_shape():
    model = SatelliteCNN(num_classes=10)
    images = torch.randn(8, 3, 64, 64)
    outputs = model(images)

    assert outputs.shape == (8, 10)


def test_model_outputs_finite_values():
    model = SatelliteCNN(num_classes=10)
    images = torch.randn(8, 3, 64, 64)
    outputs = model(images)

    assert torch.isfinite(outputs).all()
