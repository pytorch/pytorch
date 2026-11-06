import json
import os
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


def write_json(path: Path, obj: object) -> None:
    """Write `path` atomically, so a killed run never leaves a truncated result that blocks recomputation."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        # mkstemp creates the file 0o600.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
