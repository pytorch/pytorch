"""Checks for consistency of jobs between different GitHub workflows.

Any job with a specific `sync-tag` must match all other jobs with the same `sync-tag`.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from yaml import dump, load


if TYPE_CHECKING:
    from collections.abc import Iterable


# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]


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


def glob_yamls(path: Path) -> Iterable[Path]:
    return itertools.chain(path.glob("**/*.yml"), path.glob("**/*.yaml"))


def load_yaml(path: Path) -> Any:
    with open(path) as f:
        return load(f, Loader)


def is_workflow(yaml: Any) -> bool:
    return yaml.get("jobs") is not None


def print_lint_message(path: Path, job: dict[str, Any], sync_tag: str) -> None:
    job_id = next(iter(job.keys()))
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if f"{job_id}:" in line:
            line_number = i + 1

    lint_message = LintMessage(
        path=str(path),
        line=line_number,
        char=None,
        code="WORKFLOWSYNC",
        severity=LintSeverity.ERROR,
        name="workflow-inconsistency",
        original=None,
        replacement=None,
        description=f"Job doesn't match other jobs with sync-tag: '{sync_tag}'",
    )
    print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="workflow consistency linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    # Go through the provided files, aggregating jobs with the same sync tag
    tag_to_jobs = defaultdict(list)
    for path in args.filenames:
        workflow = load_yaml(Path(path))
        jobs = workflow["jobs"]
        for job_id, job in jobs.items():
            try:
                sync_tag = job["with"]["sync-tag"]
            except KeyError:
                continue

            # remove the "if" field, which we allow to be different between jobs
            # (since you might have different triggering conditions on pull vs.
            # trunk, say.)
            if "if" in job:
                del job["if"]

            # same is true for ['with']['test-matrix']
            if "test-matrix" in job.get("with", {}):
                del job["with"]["test-matrix"]

            tag_to_jobs[sync_tag].append((path, {job_id: job}))

    # For each sync tag, check that all the jobs have the same code.
    for sync_tag, path_and_jobs in tag_to_jobs.items():
        baseline_path, baseline_dict = path_and_jobs.pop()
        baseline_str = dump(baseline_dict)

        printed_baseline = False

        for path, job_dict in path_and_jobs:
            job_str = dump(job_dict)
            if baseline_str != job_str:
                print_lint_message(path, job_dict, sync_tag)

                if not printed_baseline:
                    print_lint_message(baseline_path, baseline_dict, sync_tag)
                    printed_baseline = True
