from pathlib import Path
import matplotlib.pyplot as plt
from satellite_classifier.data import create_dataloaders

DATA_DIR = Path("data/EuroSAT_RGB")


def main():

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = create_dataloaders(DATA_DIR)

    # Display dataset sizes

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    # Retrieve one training batch

    images, labels = next(iter(train_loader))

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Classes:", class_names)
    print("First 10 labels:", labels[:10].tolist())

    # Visualize the first 10 images

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 7),
    )

    for ax, image, label in zip(
        axes.flat,
        images[:10],
        labels[:10],
    ):
        # Convert from [C, H, W] to [H, W, C]

        image = image.permute(1, 2, 0)

        # Convert the tensor to a NumPy array

        image = image.numpy()

        # Convert the integer label into a class name

        class_name = class_names[label.item()]

        ax.imshow(image)
        ax.set_title(class_name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
