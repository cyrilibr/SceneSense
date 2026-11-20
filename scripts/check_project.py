"""Check notebook validity and the presence of all expected deliverables."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "docs/rapport.md",
    ROOT / "docs/questions_oral.md", ROOT / "slides/SceneSense_presentation.pptx",
]
for path in required:
    if not path.is_file():
        raise SystemExit(f"Missing deliverable: {path}")
    print(f"OK {path.relative_to(ROOT)}")
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nbformat.validate(nbformat.read(path, as_version=4))
    print(f"OK {path.relative_to(ROOT)}")
print("Project file checks passed")
