"""Test run report format (see README.md). Standard library only."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, NamedTuple


# Bumped on breaking changes.
SCHEMA_VERSION = "1.0"
REPORT_SUFFIX = ".report.jsonl"


class TestId(NamedTuple):
    """How tests.tests identifies a test."""

    file: str
    suite: str
    case_name: str
    language: str
    declared_case_name: str


def gtest_identities(binary: str, listing: dict[str, Any]) -> dict[str, TestId]:
    """Each test in a gtest binary's JSON test list, by its full name
    ``<suite>.<test>``, which is also its pytest-cpp item's name.

    gtest adds an instantiation's ``Prefix/`` and a typed suite's ``/<type>`` to
    the suite, and a value-parameterized test's ``/<param>`` to the test name; the
    declared name drops that ``/<param>``."""
    file = f"cpp/{Path(binary).stem}"
    identities = {}
    for suite in listing["testsuites"]:
        for test in suite["testsuite"]:
            suite_name, name = suite["name"], test["name"]
            declared = name.partition("/")[0] if "value_param" in test else name
            test_id = TestId(file, suite_name, name, "cpp", declared)
            identities[f"{suite_name}.{name}"] = test_id
    return identities


def report_path(prefix: str, report_uuid: str) -> str:
    """``<prefix>-<uuid>.report.jsonl``."""
    return f"{prefix}-{report_uuid}{REPORT_SUFFIX}"


def _job_id() -> int:
    # get_workflow_job_id.py leaves JOB_ID empty when its GitHub API call fails,
    # and it's unset outside CI.
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
    test: TestId,
    rerun_number: int,
    outcome: str,
    started: float,
    ended: float,
    outcome_summary: str = "",
) -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        **test._asdict(),
        "rerun_number": rerun_number,
        "outcome": outcome,
        "outcome_summary": outcome_summary,
        "started_at": int(started * 1000),
        "ended_at": int(ended * 1000),
        "properties": {},
    }


def line(record: dict[str, Any]) -> str:
    # ensure_ascii would write a lone surrogate as a \ud800 escape, which ClickHouse
    # rejects as invalid JSON, so write UTF-8 and replace lone surrogates with "?".
    text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8", "replace").decode() + "\n"
