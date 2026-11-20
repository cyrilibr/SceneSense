"""Shared settings for the five experiments."""
from dataclasses import dataclass
from pathlib import Path

CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
SEED = 42
IMAGE_SIZE = 160
EXPERIMENTS = (
    "cnn_baseline",
    "cnn_augmentation",
    "cnn_dropout",
    "mobilenet_frozen",
    "mobilenet_finetune",
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/intel-image-classification")
    output_dir: Path = Path("results")
    size: int = IMAGE_SIZE
    batch_size: int = 32
    seed: int = SEED
    validation_fraction: float = 0.10
    epochs_cnn: int = 25
    epochs_transfer: int = 15
    epochs_finetune: int = 10
    patience: int = 4
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "splits.csv"
