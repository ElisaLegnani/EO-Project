import torch
from fastapi.testclient import TestClient

import satellite_classifier.api as api


def test_health_endpoint(monkeypatch):
    # Avoid loading the real model checkpoint during testing
    monkeypatch.setattr(
        api,
        "load_model",
        lambda: (object(), torch.device("cpu")),
    )

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
