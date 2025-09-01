#!/usr/bin/env python3
"""
TODO
"""

from __future__ import annotations

import argparse
import json
import os.path
from enum import Enum
from typing import NamedTuple

import ruamel.yaml  # type: ignore[import]


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="github actions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    for fn in args.filenames:
        with open(fn) as f:
            contents = f.read()

        yaml = ruamel.yaml.YAML()  # type: ignore[attr-defined]
        try:
            r = yaml.load(contents)
        except Exception as err:
            msg = LintMessage(
                path=None,
                line=None,
                char=None,
                code="GHA",
                severity=LintSeverity.ERROR,
                name="YAML load failure",
                original=None,
                replacement=None,
                description=f"Failed due to {err.__class__.__name__}:\n{err}",
            )

            print(json.dumps(msg._asdict()), flush=True)
            continue

        for job_name, job in r.get("jobs", {}).items():
            # This filter is flexible, the idea is to avoid catching all of
            # the random label jobs that don't need secrets
            # Check for build, test, and binary jobs that should have secrets: inherit
            # Binary jobs are included as they often need access to secrets for publishing
            uses = os.path.basename(job.get("uses", ""))
            if ("build" in uses or "test" in uses or "binary" in uses):
                if job.get("secrets") != "inherit":
                    desc = "missing 'secrets: inherit' field"
                    if job.get("secrets") is not None:
                        desc = "has 'secrets' field which is not standard form 'secrets: inherit'"
                    msg = LintMessage(
                        path=fn,
                        line=job.lc.line,
                        char=None,
                        code="GHA",
                        severity=LintSeverity.ERROR,
                        name="missing secrets: inherit",
                        original=None,
                        replacement=None,
                        description=(f"GitHub actions job '{job_name}' {desc}"),
                    )

                    print(json.dumps(msg._asdict()), flush=True)
