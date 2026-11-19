from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/EuroSAT_RGB")


def main():
    class_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for ax, class_dir in zip(axes.flat, class_dirs):
        image_path = next(class_dir.glob("*.jpg"))

        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(class_dir.name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
