"""Download the public Intel Image Classification dataset from Kaggle.

Run after `pip install kagglehub`, with internet access.
"""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/intel-image-classification"))
    args = parser.parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install the dataset helper first: pip install kagglehub") from exc
    source = Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    found = []
    for part in ("seg_train", "seg_test"):
        matches = [p for p in source.rglob(part) if p.is_dir() and p.parent.name != part]
        matches = [p for p in matches if any((p / cls).is_dir() for cls in ("buildings", "forest")) or
                   (p / part).is_dir()]
        if not matches:
            raise FileNotFoundError(f"Could not find {part} within the downloaded dataset: {source}")
        origin = min(matches, key=lambda p: len(p.parts))
        destination = target / part
        shutil.copytree(origin, destination, dirs_exist_ok=True)
        found.append(destination)
        print(f"Copied {part}: {destination}")
    print("Dataset ready. Run: python -m scenesense.cli prepare")


if __name__ == "__main__":
    main()
