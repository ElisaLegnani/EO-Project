import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay

from satellite_classifier.config import TrainConfig
from satellite_classifier.data import create_dataloaders
from satellite_classifier.model import SatelliteCNN


def set_seed(seed: int):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(config: TrainConfig):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(config.device)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += batch_size

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += batch_size

            all_predictions.extend(predictions.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        all_predictions,
        all_labels,
    )


def save_training_curves(history, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss

    axes[0].plot(
        history["train_loss"],
        label="Training",
    )

    axes[0].plot(
        history["val_loss"],
        label="Validation",
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy

    axes[1].plot(
        history["train_accuracy"],
        label="Training",
    )

    axes[1].plot(
        history["val_accuracy"],
        label="Validation",
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()

    output_path = output_dir / "training_curves.png"

    fig.savefig(output_path)

    plt.close(fig)


def save_confusion_matrix(
    true_labels,
    predictions,
    class_names,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predictions,
        display_labels=class_names,
        xticks_rotation=45,
        ax=ax,
    )

    ax.set_title("Validation Confusion Matrix")

    fig.tight_layout()

    output_path = output_dir / "confusion_matrix.png"

    fig.savefig(output_path)

    plt.close(fig)


def main():
    config = TrainConfig()

    set_seed(config.seed)

    device = get_device(config)

    print("Using device:", device)

    # Load datasets

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = create_dataloaders(
        config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    # Create model

    model = SatelliteCNN(num_classes=config.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    # Create output directories

    config.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = config.checkpoint_dir / "best_model.pth"

    # Configure MLflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run():
        # Log configuration parameters

        mlflow.log_params(
            {
                "seed": config.seed,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "num_workers": config.num_workers,
                "num_classes": config.num_classes,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "model": "SatelliteCNN",
                "device": str(device),
            }
        )

        best_val_accuracy = 0.0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # Training loop

        for epoch in range(config.epochs):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            (
                val_loss,
                val_accuracy,
                _,
                _,
            ) = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_accuracy)

            # Log metrics to MLflow

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

            print(
                f"Epoch [{epoch + 1}/{config.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_accuracy:.4f}"
            )

            # Save the best checkpoint

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "config": config.model_dump(mode="json"),
                        "epoch": epoch + 1,
                        "val_accuracy": val_accuracy,
                    },
                    checkpoint_path,
                )

        print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")

        # Log the best checkpoint

        mlflow.log_artifact(
            str(checkpoint_path),
            artifact_path="model",
        )

        # Generate training curves

        save_training_curves(
            history,
            config.artifacts_dir,
        )

        # Reload best checkpoint for validation predictions

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        _, _, predictions, true_labels = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        save_confusion_matrix(
            true_labels,
            predictions,
            class_names,
            config.artifacts_dir,
        )

        # Log generated artifacts

        mlflow.log_artifacts(
            str(config.artifacts_dir),
            artifact_path="evaluation",
        )


if __name__ == "__main__":
    main()
