import json
import os
import pathlib
from typing import Any, List


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4 - 1]


def gen_ci_artifact(included: List[Any], excluded: List[Any]) -> None:
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)
