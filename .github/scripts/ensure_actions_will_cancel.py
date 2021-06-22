#!/usr/bin/env python3

import argparse
import sys
import yaml

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def concurrency_key(filename: Path) -> str:
    workflow_name = filename.with_suffix("").name.replace("_", "-")
    return f"{workflow_name}-${{{{ github.event.pull_request.number || github.sha }}}}"


def should_check(filename: Path) -> bool:
    with open(filename, "r") as f:
        content = f.read()

    data = yaml.safe_load(content)
    on = data.get("on", data.get(True, {}))
    return "pull_request" in on


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensure all relevant GitHub actions jobs will be cancelled based on a concurrency key"
    )
    args = parser.parse_args()

    files = list(WORKFLOWS.glob("*.yml"))

    errors_found = False
    files = [f for f in files if should_check(f)]
    for filename in files:
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        expected = {
            "group": concurrency_key(filename),
            "cancel-in-progress": True,
        }
        if data.get("concurrency", None) != expected:
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            errors_found = True

    if errors_found:
        sys.exit(1)
