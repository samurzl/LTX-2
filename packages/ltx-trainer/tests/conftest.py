import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

for relative_path in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
    "packages/ltx-trainer/scripts",
):
    sys.path.insert(0, str(REPO_ROOT / relative_path))
