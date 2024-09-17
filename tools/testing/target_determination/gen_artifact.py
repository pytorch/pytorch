from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def gen_ci_artifact(included: list[Any], excluded: list[Any]) -> None:
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)
