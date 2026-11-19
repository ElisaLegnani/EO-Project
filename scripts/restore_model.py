import torch
from satellite_classifier.model import SatelliteCNN

checkpoint = torch.load(
    "models/best_model.pth",
    map_location="cpu",
)

model = SatelliteCNN(num_classes=10)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Checkpoint loaded successfully")
print("Classes:", checkpoint["class_names"])
print("Best epoch:", checkpoint["epoch"])
print("Validation accuracy:", checkpoint["val_accuracy"])
