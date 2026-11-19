from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/EuroSAT_RGB")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    class_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())

    total_images = 0

    print("\nEuroSAT Dataset Inspection")
    print("-" * 50)

    for class_dir in class_dirs:
        images = [
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        total_images += len(images)

        print(f"{class_dir.name:25s}: {len(images)} images")

        if images:
            with Image.open(images[0]) as image:
                print(f"  Image size: {image.size}, Mode: {image.mode}")

    print("-" * 50)
    print(f"Total images: {total_images}")
    print(f"Total classes: {len(class_dirs)}")


if __name__ == "__main__":
    main()
