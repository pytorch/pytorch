"""
Emit adhoc-failure rows for tests that started but never produced a junit-xml
entry — e.g. when the GHA test step was killed by `timeout-minutes`, by
`gh run cancel`, by an OOM-killer, or by any other SIGTERM/SIGKILL that hit
pytest before `pytest_sessionfinish` ran.

The pytest plugin `StepcurrentPlugin` (test/conftest.py) persists the
in-flight nodeid on every `pytest_runtest_protocol` hook to
`.pytest_cache/v/cache/stepcurrent/<key>/lastrun`. It also writes
`made_failing_xml=true` to the same directory on `pytest_sessionfinish` when
exitstatus != 0 — meaning pytest finished cleanly enough to emit failure xml.

This script walks every `stepcurrent` subdir, reads `lastrun` and
`made_failing_xml`, then emits one adhoc-failure JSON for each nodeid that:

1. Was the last test to start running (per `lastrun`).
2. Has NOT been flagged `made_failing_xml=true` (segfault path already
   covered by `run_test.py::run_test_retries`).
3. Does NOT appear as a `<testcase>` in any junit-xml under
   `test/test-reports/**/*.xml` (i.e. pytest never wrote a row for it).

Designed to be invoked from a GHA composite-action step gated `if: failure()`
right before the existing `upload-test-artifacts` zip+upload step, so it
fires for ANY test-step failure mode — not just SIGTERM — without competing
with the in-process `parse_xml_and_upload_json()` path or the
`FAILED CONSISTENTLY` path in `run_test.py`.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.testing.upload_artifacts import upload_adhoc_failure_json


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STEPCURRENT_DIR = REPO_ROOT / ".pytest_cache" / "v" / "cache" / "stepcurrent"
TEST_REPORTS_DIR = REPO_ROOT / "test" / "test-reports"


def _read_pytest_cache_value(path: Path) -> str | None:
    """pytest.Cache writes JSON files; null/missing → None, string → unquoted."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _collect_completed_testcases() -> set[tuple[str, str]]:
    """Build a set of (classname, name) tuples from every junit-xml on disk.

    junit-xml `<testcase>` elements carry `classname` (may be empty) + `name`.
    These match the (className, testName) split used by
    `upload_adhoc_failure_json` so dedup is cheap.
    """
    completed: set[tuple[str, str]] = set()
    if not TEST_REPORTS_DIR.is_dir():
        return completed
    for xml_file in glob.glob(f"{TEST_REPORTS_DIR}/**/*.xml", recursive=True):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            # A partially-written XML is treated as no data — the adhoc helper
            # will fill the gap if the relevant testcase is missing.
            continue
        for testcase in tree.iter("testcase"):
            classname = testcase.get("classname") or ""
            name = testcase.get("name") or ""
            if name:
                completed.add((classname, name))
    return completed


def _nodeid_to_emit_args(nodeid: str) -> tuple[str, str] | None:
    """Map a pytest nodeid to (invoking_file, current_failure_arg).

    A pytest nodeid is `<path>.py::<class>::<name>` or `<path>.py::<name>`.
    `upload_adhoc_failure_json` expects:
      - invoking_file: path without `.py` (e.g. `dynamo/test_foo`)
      - current_failure: the `::`-delimited tail (className::testName or just
        testName), so its own splitter recovers the same classname/name.
    """
    parts = nodeid.split("::")
    if len(parts) < 2:
        return None
    path_part = parts[0]
    if not path_part.endswith(".py"):
        return None
    invoking_file = path_part[:-3]
    # Re-join everything after the file part; `upload_adhoc_failure_json`
    # takes the last two `::`-separated segments.
    current_failure = "::".join(parts[1:])
    return invoking_file, current_failure


def _classname_testname(current_failure: str) -> tuple[str, str]:
    parts = current_failure.split("::")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "", current_failure


def main() -> int:
    if not STEPCURRENT_DIR.is_dir():
        print(f"No stepcurrent cache at {STEPCURRENT_DIR}; nothing to emit.")
        return 0

    completed = _collect_completed_testcases()
    emitted = 0

    for entry in sorted(STEPCURRENT_DIR.iterdir()):
        if not entry.is_dir():
            continue

        if _read_pytest_cache_value(entry / "made_failing_xml") is True:
            # pytest reached sessionfinish with exit != 0; failure xml was
            # already written and the FAILED CONSISTENTLY path may have
            # already emitted an adhoc row. Skip either way.
            continue

        lastrun = _read_pytest_cache_value(entry / "lastrun")
        if not isinstance(lastrun, str) or not lastrun:
            continue

        emit_args = _nodeid_to_emit_args(lastrun)
        if emit_args is None:
            continue
        invoking_file, current_failure = emit_args

        classname, testname = _classname_testname(current_failure)
        if (classname, testname) in completed:
            # junit-xml already has a row for this test (it ran to completion
            # and a later test was killed). Don't duplicate.
            continue

        reason = (
            "Test step terminated before pytest could write junit-xml "
            "(SIGTERM/SIGKILL — e.g. timeout-minutes, run cancellation, "
            f"or OOM). Last test seen by stepcurrent: {lastrun}"
        )
        suffix = f"{classname}_{testname}".replace("/", "_").replace(" ", "_")
        try:
            upload_adhoc_failure_json(
                invoking_file,
                current_failure,
                reason=reason,
                s3_key_suffix=f"incomplete_{suffix}",
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to emit adhoc row for {lastrun}: {e}")
            continue
        emitted += 1
        print(f"Emitted adhoc-incomplete row for {lastrun}")

    print(f"upload_adhoc_incomplete_tests: emitted {emitted} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
