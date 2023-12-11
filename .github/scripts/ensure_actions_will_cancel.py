#!/usr/bin/env python3

import sys

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EXPECTED_GROUP_PREFIX = (
    "${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}"
)
EXPECTED_GROUP = (
    EXPECTED_GROUP_PREFIX + "-${{ github.event_name == 'workflow_dispatch' }}"
)


def should_check(filename: Path) -> bool:
    with open(filename) as f:
        content = f.read()

    data = yaml.safe_load(content)
    on = data.get("on", data.get(True, {}))
    return "pull_request" in on


if __name__ == "__main__":
    errors_found = False
    files = [f for f in WORKFLOWS.glob("*.yml") if should_check(f)]
    names = set()
    for filename in files:
        with open(filename) as f:
            data = yaml.safe_load(f)

        name = data.get("name")
        if name is not None and name in names:
            print("ERROR: duplicate workflow name:", name, file=sys.stderr)
            errors_found = True
        names.add(name)
        actual = data.get("concurrency", {})
        if filename.name == "create_release.yml":
            if not actual.get("group", "").startswith(EXPECTED_GROUP_PREFIX):
                print(
                    f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                    file=sys.stderr,
                )
                print(
                    f"concurrency group should start with {EXPECTED_GROUP_PREFIX} but found {actual.get('group', None)}",
                    file=sys.stderr,
                )
                errors_found = True
        elif not actual.get("group", "").startswith(EXPECTED_GROUP):
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
