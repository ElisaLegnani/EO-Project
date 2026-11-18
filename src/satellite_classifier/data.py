from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

BATCH_SIZE = 32
IMAGE_SIZE = 64
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def get_transforms():
    """Return the transformations applied to satellite images."""

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def create_datasets(data_dir: Path, seed: int = SEED):
    """Load EuroSAT and create stratified dataset splits."""

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    transform = get_transforms()

    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
    )

    if len(full_dataset.classes) != 10:
        raise ValueError(
            f"Expected 10 classes, found {len(full_dataset.classes)}"
        )

    class_indices = {}

    for index, (_, label) in enumerate(full_dataset.samples):
        class_indices.setdefault(label, []).append(index)

    train_indices = []
    val_indices = []
    test_indices = []

    generator = torch.Generator().manual_seed(seed)

    for indices in class_indices.values():

        if len(indices) < 3:
            raise ValueError(
                "Each class needs at least 3 images for splitting."
            )

        indices = torch.tensor(indices)

        shuffled_indices = indices[
            torch.randperm(
                len(indices),
                generator=generator,
            )
        ].tolist()

        total = len(shuffled_indices)

        train_size = max(1, int(TRAIN_RATIO * total))
        val_size = max(1, int(VAL_RATIO * total))
        test_size = total - train_size - val_size

        if test_size < 1:
            train_size -= 1
            test_size = 1

        train_indices.extend(
            shuffled_indices[:train_size]
        )

        val_indices.extend(
            shuffled_indices[
                train_size:train_size + val_size
            ]
        )

        test_indices.extend(
            shuffled_indices[
                train_size + val_size:
            ]
        )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        full_dataset.classes,
    )

def create_dataloaders(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
):
    """Create DataLoaders for training, validation and testing."""

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_names,
    ) = create_datasets(data_dir, seed=seed)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    )
