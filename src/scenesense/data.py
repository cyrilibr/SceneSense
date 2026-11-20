"""Image indexing and batch loading without leaking evaluation images into training."""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from scenesense.config import CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _find_class_folder(data_dir: Path, split: str) -> Path:
    """Accept both the original double-nested folders and the flat version."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / split / split,
        data_dir / split,
        data_dir,
    ]
    for folder in candidates:
        if folder.is_dir() and all((folder / name).is_dir() for name in CLASSES):
            return folder
    raise FileNotFoundError(
        f"Cannot find the six class folders under {data_dir}/{split}. "
        "Extract the Intel Kaggle dataset so seg_train and seg_test are present."
    )


def build_manifest(data_dir: Path, output_csv: Path, seed: int = 42,
                   validation_fraction: float = 0.10) -> pd.DataFrame:
    """Make a fixed, class-stratified split; keep the original test split intact."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be strictly between 0 and 0.5")
    data_dir, output_csv = Path(data_dir).resolve(), Path(output_csv)
    train_dir = _find_class_folder(data_dir, "seg_train")
    test_dir = _find_class_folder(data_dir, "seg_test")
    rng = np.random.default_rng(seed)
    records = []
    seen = set()

    for label, cls in enumerate(CLASSES):
        train_paths = sorted(p for p in (train_dir / cls).iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        test_paths = sorted(p for p in (test_dir / cls).iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if len(train_paths) < 2 or len(test_paths) < 1:
            raise ValueError(f"Not enough images for class {cls}: "
                             f"train={len(train_paths)}, test={len(test_paths)}")
        indices = rng.permutation(len(train_paths))
        n_val = max(1, int(round(len(train_paths) * validation_fraction)))
        val_ids = set(indices[:n_val].tolist())
        for i, path in enumerate(train_paths):
            relative = path.relative_to(data_dir).as_posix()
            if relative in seen:
                raise ValueError(f"Duplicate image in split: {relative}")
            seen.add(relative)
            records.append((relative, label, "validation" if i in val_ids else "train"))
        for path in test_paths:
            relative = path.relative_to(data_dir).as_posix()
            if relative in seen:
                raise ValueError(f"An image path appears in multiple splits: {relative}")
            seen.add(relative)
            records.append((relative, label, "test"))

    frame = pd.DataFrame(records, columns=["path", "label", "split"])
    frame = frame.sort_values(["split", "label", "path"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def read_manifest(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    frame = pd.read_csv(csv_path)
    if not {"path", "label", "split"}.issubset(frame.columns):
        raise ValueError("The split manifest must contain path, label and split columns")
    if frame.empty or frame.path.duplicated().any():
        raise ValueError("Empty manifest or duplicated file paths")
    if not set(frame["split"]) <= {"train", "validation", "test"}:
        raise ValueError("Unexpected split name in manifest")
    if not set(frame["label"].astype(int)) <= set(range(len(CLASSES))):
        raise ValueError("Unexpected class ID in manifest")
    missing = [p for p in frame.path if not (data_dir / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images: {missing[0]}")
    return frame


class ImageBatches(keras.utils.PyDataset):
    """Images are 0..255 float32; each model owns its correct preprocessing layer."""

    def __init__(self, frame: pd.DataFrame, data_dir: Path, image_size: int = 160,
                 batch_size: int = 32, shuffle: bool = False, seed: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        if frame.empty:
            raise ValueError("Cannot train or evaluate an empty image split")
        self.frame = frame.reset_index(drop=True)
        self.data_dir = Path(data_dir).resolve()
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(frame))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        ids = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.frame.iloc[ids]
        images = np.empty((len(batch), self.image_size, self.image_size, 3), dtype=np.float32)
        labels = batch["label"].to_numpy(dtype=np.int32)
        for i, name in enumerate(batch["path"]):
            with Image.open(self.data_dir / name) as img:
                rgb = ImageOps.exif_transpose(img).convert("RGB")
                rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                images[i] = np.asarray(rgb, dtype=np.float32)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def make_batches(frame, data_dir, split, settings):
    subset = frame.loc[frame.split == split]
    return ImageBatches(subset, data_dir, settings.size, settings.batch_size,
                        shuffle=(split == "train"), seed=settings.seed)
