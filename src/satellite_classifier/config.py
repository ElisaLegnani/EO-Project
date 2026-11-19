from pathlib import Path
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Configuration for satellite classifier training."""

    data_dir: Path = Path("data/EuroSAT_RGB")

    seed: int = 42

    batch_size: int = Field(default=32, gt=0)

    learning_rate: float = Field(
        default=0.001,
        gt=0,
    )

    epochs: int = Field(default=10, gt=0)

    num_workers: int = Field(default=0, ge=0)

    num_classes: int = Field(default=10, gt=0)

    checkpoint_dir: Path = Path("models")

    artifacts_dir: Path = Path("artifacts")

    experiment_name: str = "eurosat-classification"

    device: str = "auto"
