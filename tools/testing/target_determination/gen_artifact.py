import json
import pathlib
from typing import Any, List
import os

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def gen_ci_artifact(included: List[Any], excluded: List[Any]) -> None:
    with open(REPO_ROOT / f"test/test-reports/td_exclusions-{os.urandom(10).hex()}.json", "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)
