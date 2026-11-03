import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


def write_json(path: Path, obj: object) -> None:
    """Write `path` atomically, so a killed run never leaves a truncated result that blocks recomputation."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)
