"""Test run report format (see README.md). Standard library only."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, NamedTuple


# Bumped on breaking changes.
SCHEMA_VERSION = "0.1"
REPORT_SUFFIX = ".report.jsonl"


class TestId(NamedTuple):
    """How tests.tests identifies a test."""

    file: str
    suite: str
    case_name: str
    language: str
    declared_case_name: str


def gtest_identity(binary: str, name: str) -> TestId:
    """A pytest-cpp gtest's identity, from its binary and gtest name.

    pytest-cpp names a gtest ``<suite>.<test>`` from --gtest_list_tests. gtest
    suite and test names are C++ identifiers, so neither contains ".", and the
    split is exact. gtest adds an instantiation's ``Prefix/`` and a typed suite's
    ``/N`` to the suite, and a value-parameterized test's ``/<param>`` to the test
    name; the declared name drops that ``/<param>``."""
    suite, _, case_name = name.partition(".")
    declared = case_name.partition("/")[0]
    return TestId(f"cpp/{Path(binary).stem}", suite, case_name, "cpp", declared)


def gtest_def_files(listing: Path, root: Path) -> dict[str, str]:
    """``<suite>.<test>`` to the file under ``root`` that defines it, from the JSON
    that ``--gtest_list_tests`` writes under GTEST_OUTPUT; {} without a listing.

    gtest records each test's file as the compiler was given it, an absolute path
    under the build's checkout, which needn't be ``root``. The file is the longest
    suffix of that path that exists under ``root``, or "" if none does, as for the
    sources ROCm builds compile from hipified copies."""
    try:
        suites = json.loads(listing.read_text(encoding="utf-8"))["testsuites"]
    except (OSError, ValueError, KeyError):
        return {}
    resolved: dict[str, str] = {}

    def def_file(path: str) -> str:
        if path not in resolved:
            # Without the root or drive, so every suffix is relative to root.
            parts = [p for p in re.split(r"[\\/]", path) if p and not p.endswith(":")]
            suffixes = ("/".join(parts[i:]) for i in range(len(parts)))
            resolved[path] = next((s for s in suffixes if (root / s).is_file()), "")
        return resolved[path]

    return {
        f"{suite['name']}.{test['name']}": def_file(test.get("file", ""))
        for suite in suites
        for test in suite["testsuite"]
    }


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
    def_file: str = "",
) -> dict[str, Any]:
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        **test._asdict(),
        "def_file": def_file,
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
