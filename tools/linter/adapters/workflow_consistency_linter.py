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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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


def print_lint_message(
    path: Path,
    job: dict[str, Any],
    sync_tag: str,
    baseline_path: Path,
    baseline_job_id: str,
) -> None:
    job_id = next(iter(job.keys()))
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if f"{job_id}:" in line:
            line_number = i + 1

    lint_message = LintMessage(
        path=str(path),
        # pyrefly: ignore [unbound-name]
        line=line_number,
        char=None,
        code="WORKFLOWSYNC",
        severity=LintSeverity.ERROR,
        name="workflow-inconsistency",
        original=None,
        replacement=None,
        description=f"Job doesn't match other job {baseline_job_id} in file {baseline_path} with sync-tag: '{sync_tag}'",
    )
    print(json.dumps(lint_message._asdict()), flush=True)


def get_jobs_with_sync_tag(
    job: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    sync_tag = job.get("with", {}).get("sync-tag")
    if sync_tag is None:
        return None

    # remove the "if" field, which we allow to be different between jobs
    # (since you might have different triggering conditions on pull vs.
    # trunk, say.)
    if "if" in job:
        del job["if"]

    # same is true for ['with']['test-matrix']
    if "test-matrix" in job.get("with", {}):
        del job["with"]["test-matrix"]
    # and ['with']['tests-to-include'], since dispatch filters differ
    if "tests-to-include" in job.get("with", {}):
        del job["with"]["tests-to-include"]

    # normalize needs: remove helper job-filter so comparisons ignore it
    needs = job.get("needs")
    if needs:
        needs_list = [needs] if isinstance(needs, str) else list(needs)
        needs_list = [n for n in needs_list if n != "job-filter"]
        if not needs_list:
            job.pop("needs", None)
        elif len(needs_list) == 1:
            job["needs"] = needs_list[0]
        else:
            job["needs"] = needs_list

    return (sync_tag, job_id, job)


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

    # Go through all files, aggregating jobs with the same sync tag
    tag_to_jobs = defaultdict(list)
    for path in REPO_ROOT.glob(".github/workflows/*"):
        if not path.is_file() or path.suffix not in {".yml", ".yaml"}:
            continue
        workflow = load_yaml(path)
        if not is_workflow(workflow):
            continue
        clean_path = path.relative_to(REPO_ROOT)
        jobs = workflow.get("jobs", {})
        for job_id, job in jobs.items():
            res = get_jobs_with_sync_tag(job)
            if res is None:
                continue
            sync_tag, job_id, job_dict = res
            tag_to_jobs[sync_tag].append((clean_path, job_id, job_dict))

    # Check the files passed as arguments
    for path in args.filenames:
        workflow = load_yaml(Path(path))
        jobs = workflow["jobs"]
        for job_id, job in jobs.items():
            res = get_jobs_with_sync_tag(job)
            if res is None:
                continue
            sync_tag, job_id, job_dict = res
            job_str = dump(job_dict)

            # For each sync tag, check that all the jobs have the same code.
            for baseline_path, baseline_job_id, baseline_dict in tag_to_jobs[sync_tag]:
                baseline_str = dump(baseline_dict)

                if job_id != baseline_job_id or job_str != baseline_str:
                    print_lint_message(
                        path, job_dict, sync_tag, baseline_path, baseline_job_id
                    )
