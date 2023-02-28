#!/usr/bin/env python3

import argparse
import sys
import yaml

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EXPECTED_GROUP = "${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}" \
    "-${{ github.event_name == 'workflow_dispatch' }}"


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
    names = set()
    for filename in files:
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        name = data.get("name")
        if name is not None and name in names:
            print("ERROR: duplicate workflow name:", name, file=sys.stderr)
            errors_found = True
        names.add(name)
        actual = data.get("concurrency", {})
        if not actual.get("group", "").startswith(EXPECTED_GROUP):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency group should start with {EXPECTED_GROUP} but found {actual.get('group', None)}",
                file=sys.stderr,
            )
            errors_found = True
        if not actual.get("cancel-in-progress", False):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency cancel-in-progress should be True but found {actual.get('cancel-in-progress', None)}",
                file=sys.stderr,
            )

    if errors_found:
        sys.exit(1)
