import pathlib
import json

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def gen_ci_artifact(excluded_tests):
    with open(REPO_ROOT / "test/test-reports/td_exclusions.json", "w") as f:
        json.dump(excluded_tests, f)
