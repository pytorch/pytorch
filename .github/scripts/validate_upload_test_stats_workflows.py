#!/usr/bin/env python3
"""Check that every workflow upload-test-stats.yml listens for actually exists.

`on: workflow_run: workflows:` matches a workflow's `name:`, not its filename, so
an entry that names nothing silently never fires and that workflow's test results
are never ingested -- with no error anywhere. `mac-mps` sat in the list while the
workflow was named `Mac MPS`, so Mac MPS results went unrecorded.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
UPLOADER = WORKFLOWS / "upload-test-stats.yml"


def workflow_names() -> dict[str, str]:
    out = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        m = re.search(r"^name:\s*(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            out[m.group(1).strip().strip("'\"")] = path.name
    return out


def main() -> int:
    # `on` parses as the boolean True in YAML 1.1, which is what PyYAML implements.
    triggers = yaml.safe_load(UPLOADER.read_text(encoding="utf-8"))[True]
    listed = triggers["workflow_run"]["workflows"]
    missing = [n for n in listed if n not in workflow_names()]
    if missing:
        print(
            f"{UPLOADER.name} listens for workflows that do not exist: "
            f"{', '.join(missing)}\n"
            "Entries must match a workflow's `name:` field, not its filename.",
            file=sys.stderr,
        )
        return 1
    print(f"upload-test-stats.yml: all {len(listed)} workflows exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
