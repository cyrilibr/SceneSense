import numpy as np
import pandas as pd
import pytest

from scenesense.config import CLASSES
from scenesense.data import ImageBatches, build_manifest, read_manifest


def test_stratified_splits_are_stable(small_dataset, tmp_path):
    first = build_manifest(small_dataset, tmp_path / "split_a.csv", seed=42)
    second = build_manifest(small_dataset, tmp_path / "split_b.csv", seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 42
    assert set(first.split) == {"train", "validation", "test"}
    assert (first.split == "train").sum() == 24
    assert (first.split == "validation").sum() == 6
    assert (first.split == "test").sum() == 12
    assert len(first.path.unique()) == len(first)
    for label in range(len(CLASSES)):
        assert len(first[(first.label == label) & (first.split == "test")]) == 2
    pd.testing.assert_frame_equal(first, read_manifest(small_dataset, tmp_path / "split_a.csv"))


def test_batch_shapes_and_pixel_scale(small_dataset, tmp_path):
    frame = build_manifest(small_dataset, tmp_path / "manifest.csv")
    seq = ImageBatches(frame[frame.split == "train"], small_dataset, image_size=64,
                       batch_size=5, shuffle=False)
    images, labels = seq[0]
    assert images.shape == (5, 64, 64, 3)
    assert images.dtype == np.float32
    assert labels.dtype == np.int32
    assert 0 <= images.min() <= images.max() <= 255
    assert sum(len(seq[i][1]) for i in range(len(seq))) == 24
    with pytest.raises(IndexError):
        seq[len(seq)]


def test_read_detects_missing_image(small_dataset, tmp_path):
    frame = build_manifest(small_dataset, tmp_path / "original.csv")
    frame.loc[0, "path"] = "missing.png"
    frame.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(FileNotFoundError):
        read_manifest(small_dataset, tmp_path / "bad.csv")


def test_invalid_validation_fraction(small_dataset, tmp_path):
    with pytest.raises(ValueError):
        build_manifest(small_dataset, tmp_path / "invalid.csv", validation_fraction=0)
