import json
import os
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


def write_json(path: Path, obj: object) -> None:
    """Write `path` atomically, so a killed run never leaves a truncated result that blocks recomputation."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
