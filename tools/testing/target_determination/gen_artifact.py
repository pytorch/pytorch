import json
import os
from pathlib import Path
from typing import Any, List


REPO_ROOT = Path(__file__).absolute().parents[3]


def gen_ci_artifact(included: List[Any], excluded: List[Any]) -> None:
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)
