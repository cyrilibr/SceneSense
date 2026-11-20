from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenesense.config import CLASSES


@pytest.fixture()
def small_dataset(tmp_path):
    root = tmp_path / "intel"
    rng = np.random.default_rng(10)
    for split, n in (("seg_train", 5), ("seg_test", 2)):
        for label, name in enumerate(CLASSES):
            folder = root / split / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 200, size=(42, 38, 3), dtype=np.uint8)
                arr[..., label % 3] = np.clip(arr[..., label % 3] + 30, 0, 255)
                Image.fromarray(arr).save(folder / f"image_{i}.png")
    return root
