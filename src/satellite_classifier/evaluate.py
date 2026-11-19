from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from satellite_classifier.data import create_dataloaders
from satellite_classifier.model import SatelliteCNN

DATA_DIR = Path("data/EuroSAT_RGB")
CHECKPOINT_PATH = Path("models/best_model.pth")
ARTIFACTS_DIR = Path("artifacts")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def evaluate():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader, class_names = create_dataloaders(DATA_DIR)

    model = SatelliteCNN().to(device)
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nPer-class evaluation:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("EuroSAT Test Set Confusion Matrix")
    fig.tight_layout()

    output_path = ARTIFACTS_DIR / "confusion_matrix_test.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    evaluate()
