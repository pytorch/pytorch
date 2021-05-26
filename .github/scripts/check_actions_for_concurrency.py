#!/usr/bin/env python3

import argparse
import sys
import yaml
import io
import re

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def concurrency_key(filename, job_name):
    workflow_name = filename.with_suffix("").name.replace("_", "-")
    return f"{workflow_name}-{job_name}-${{{{ github.event.pull_request.number || github.sha }}}}"


def insert_concurrency_keys(filename):
    # Import here since this isn't installed in CI where this script runs
    from ruamel.yaml import YAML

    with open(filename, "r") as f:
        if "@generated DO NOT EDIT MANUALLY" in f.read():
            print(f"Skipping generated file '{filename}'")

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.width = 1000
    data = yaml.load(filename)

    for job_name, data in data["jobs"].items():
        data["concurrency"] = {
            "group": concurrency_key(filename, job_name),
            "cancel-in-progress": True,
        }

    buf = io.StringIO()
    yaml.dump(data, buf)
    # ruamel.YAML sucks and puts a newline before this entry instead of after, so
    # fix that up manually
    content = buf.getvalue()
    content = re.sub(
        r"\n\n( +)concurrency:\n(.*)\n(.*)\n",
        r"\n\1concurrency:\n\2\n\3\n\n",
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
    for filename in files:
        if args.write:
            # Go and try to insert the relevant entry for the concurrency group
            insert_concurrency_keys(filename)
        else:
            with open(filename, "r") as f:
                data = yaml.safe_load(f)

            for job_name, data in data.get("jobs", {}).items():
                expected = {
                    "group": concurrency_key(filename, job_name),
                    "cancel-in-progress": True,
                }
                if "concurrency" not in data.keys():
                    print(
                        f"'concurrency' key not found for '{job_name}' in '{filename.relative_to(REPO_ROOT)}'",
                        file=sys.stderr,
                    )
                    errors_found = True
                elif data["concurrency"] != expected:
                    print(
                        f"'concurrency' key incorrect for '{job_name}' in '{filename.relative_to(REPO_ROOT)}'",
                        file=sys.stderr,
                    )
                    errors_found = True

    if errors_found:
        sys.exit(1)
