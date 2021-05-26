#!/usr/bin/env python3

import argparse
import sys
import yaml
import io
import re

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def concurrency_key(filename):
    workflow_name = filename.with_suffix("").name.replace("_", "-")
    return f"{workflow_name}-${{{{ github.event.pull_request.number || github.sha }}}}"


SKIP = {
    "update_disabled_tests.yml",
    "stale_pull_requests.yml",
    "update_s3_htmls.yml",
    "push_nightly_docker_ghcr.yml",
}


def should_skip(filename):
    if filename.name in SKIP:
        return True

    with open(filename, "r") as f:
        content = f.read()
        if "@generated DO NOT EDIT MANUALLY" in content:
            return True

    data = yaml.safe_load(content)
    on = str(data.get("on", data.get(True, None)))
    if "'pull_request'" not in on and "'workflow_dispatch'" not in on:
        return True

    return False


def insert_concurrency_keys(filename):
    # Import here since this isn't installed in CI where this script runs
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.width = 1000
    data = yaml.load(filename)

    on = str(data["on"])
    if "'pull_request'" not in on and "'workflow_dispatch'" not in on:
        print(f"Skipping non-PR workflow '{filename}'")
        return

    data["concurrency"] = {
        "group": concurrency_key(filename),
        "cancel-in-progress": True,
    }

    buf = io.StringIO()
    yaml.dump(data, buf)

    # ruamel.YAML doesn't put in a newline so do that to make it look nicer
    content = buf.getvalue()
    content = re.sub(
        r"\nconcurrency:\n",
        r"\n\nconcurrency:\n",
        content,
        flags=re.MULTILINE,
    )

    with open(filename, "w") as f:
        f.write(content)
    print(f"Wrote {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensure all relevant GitHub actions jobs will be cancelled based on a concurrency key"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        required=False,
        help="insert concurrency keys instead of checking",
    )
    args = parser.parse_args()

    files = WORKFLOWS.glob("*.yml")

    errors_found = False
    files = [f for f in files if not should_skip(f)]
    for filename in files:
        if args.write:
            # Go and try to insert the relevant entry for the concurrency group
            insert_concurrency_keys(filename)
        else:
            with open(filename, "r") as f:
                data = yaml.safe_load(f)

            expected = {
                "group": concurrency_key(filename),
                "cancel-in-progress": True,
            }
            if "concurrency" not in data.keys():
                print(
                    f"'concurrency' not found in '{filename.relative_to(REPO_ROOT)}'",
                    file=sys.stderr,
                )
                errors_found = True
            elif data["concurrency"] != expected:
                print(
                    f"'concurrency' incorrect in '{filename.relative_to(REPO_ROOT)}'",
                    file=sys.stderr,
                )
                errors_found = True

    if errors_found:
        sys.exit(1)
