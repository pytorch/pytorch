"""Test run report format (see README.md). Standard library only."""

from __future__ import annotations

import json
import os
import signal
from typing import Any, NamedTuple


# Bumped on breaking changes.
SCHEMA_VERSION = "0.1"
REPORT_SUFFIX = ".jsonl"


class TestId(NamedTuple):
    """How tests.tests identifies a test."""

    file: str
    suite: str
    case_name: str
    language: str


def identity(nodeid: str) -> TestId:
    """Launched file, innermost class and parametrized name of a node id. Classes
    that test_jit.py imports from jit/ keep test_jit.py as their file."""
    path, _, rest = nodeid.partition("::")
    head, bracket, params = rest.partition("[")
    parts = head.split("::")
    suite = parts[-2] if len(parts) > 1 else ""
    return TestId(path, suite, parts[-1] + bracket + params, "python")


def fallback_declared_case_name(nodeid: str) -> str:
    """The case name without pytest parameters."""
    return identity(nodeid).case_name.partition("[")[0]


def report_path(directory: str, report_uuid: str) -> str:
    """``<directory>/<directory name>-<uuid>.jsonl``."""
    name = os.path.basename(os.path.normpath(directory))
    return os.path.join(directory, f"{name}-{report_uuid}{REPORT_SUFFIX}")


def _job_id() -> int:
    # The workflow's job-id lookup can fail.
    job = os.environ.get("JOB_ID", "")
    return int(job) if job.isdigit() else 0


def report_record(report_uuid: str, environment: Any) -> dict[str, Any]:
    """``environment`` is an ``environment.Environment``."""
    return {
        "type": "report",
        "schema_version": SCHEMA_VERSION,
        "repo": os.environ.get("GITHUB_REPOSITORY", ""),
        "github_workflow_job_id": _job_id(),
        "report_uuid": report_uuid,
        "environment": environment.identity(),
        "flags": dict(sorted(environment.flags.items())),
        "properties": environment.properties,
    }


def run_record(
    nodeid: str,
    rerun_number: int,
    outcome: str,
    started: float,
    ended: float,
    outcome_summary: str = "",
    declared_case_name: str | None = None,
) -> dict[str, Any]:
    test = identity(nodeid)
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "file": test.file,
        "suite": test.suite,
        "case_name": test.case_name,
        "language": test.language,
        "declared_case_name": declared_case_name or fallback_declared_case_name(nodeid),
        "rerun_number": rerun_number,
        "outcome": outcome,
        "outcome_summary": outcome_summary,
        "started_at": int(started * 1000),
        "ended_at": int(ended * 1000),
        "properties": {},
    }


def exit_summary(exit_code: int) -> str:
    """The outcome summary of a run whose process exited with ``exit_code``."""
    signals = {s.value: s.name for s in signal.Signals}
    name = f" ({signals[-exit_code]})" if -exit_code in signals else ""
    return f"the test process exited with code {exit_code}{name}"


def _json_safe(value: Any) -> Any:
    # Escape lone surrogates, which UTF-8 can't encode.
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode()
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def line(record: dict[str, Any]) -> str:
    return json.dumps(_json_safe(record), separators=(",", ":")) + "\n"
