from pathlib import Path

from PIL import Image

from satellite_classifier.data import create_dataloaders


def create_fake_dataset(data_dir: Path):
    class_names = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]

    for class_name in class_names:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True)

        for index in range(10):
            image = Image.new(
                "RGB",
                (64, 64),
                color=(index * 10, 100, 150),
            )

            image.save(class_dir / f"image_{index}.png")


def test_dataloaders_return_expected_shapes(tmp_path):
    data_dir = tmp_path / "EuroSAT_RGB"

    create_fake_dataset(data_dir)

    (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = create_dataloaders(data_dir)

    images, labels = next(iter(train_loader))

    # Check image dimensions

    assert images.ndim == 4

    assert images.shape[1:] == (3, 64, 64)

    # Check label dimensions

    assert labels.ndim == 1

    # Check number of classes

    assert len(class_names) == 10

    # Check that all datasets contain images

    assert len(train_loader.dataset) > 0

    assert len(val_loader.dataset) > 0

    assert len(test_loader.dataset) > 0

    # Check that all samples are accounted for

    total_samples = (
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    )

    assert total_samples == 100


def test_dataset_splits_do_not_overlap(tmp_path):
    data_dir = tmp_path / "EuroSAT_RGB"

    create_fake_dataset(data_dir)

    (
        train_loader,
        val_loader,
        test_loader,
        _,
    ) = create_dataloaders(data_dir)

    train_indices = set(train_loader.dataset.indices)

    val_indices = set(val_loader.dataset.indices)

    test_indices = set(test_loader.dataset.indices)

    assert train_indices.isdisjoint(val_indices)

    assert train_indices.isdisjoint(test_indices)

    assert val_indices.isdisjoint(test_indices)
