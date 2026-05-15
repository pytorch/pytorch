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
3. Has no matching `<testcase>` (keyed on the xunit2 `file` and `name`
   attributes) in any `test/test-reports/**/*.xml` already on disk.

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

from tools.testing.upload_artifacts import (
    parse_pytest_nodeid,
    upload_adhoc_failure_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STEPCURRENT_DIR = REPO_ROOT / ".pytest_cache" / "v" / "cache" / "stepcurrent"
TEST_REPORTS_DIR = REPO_ROOT / "test" / "test-reports"


def _read_pytest_cache_value(path: Path) -> object | None:
    """pytest.Cache writes JSON files; missing/unparseable → None."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _collect_completed_testcases() -> set[tuple[str, str]]:
    """Build a set of (file_attr, name_attr) tuples from every junit-xml on disk.

    Keyed on the xunit2 `<testcase file="..." name="...">` attributes — those
    match the rootdir-relative file path and the full test name (including
    any parametrize suffix), which is what `parse_pytest_nodeid` returns for
    the in-flight nodeid. xunit2's `classname` uses a dotted module path so we
    deliberately avoid matching on it.
    """
    completed: set[tuple[str, str]] = set()
    if not TEST_REPORTS_DIR.is_dir():
        return completed
    for xml_file in glob.glob(f"{TEST_REPORTS_DIR}/**/*.xml", recursive=True):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            # A partially-written XML is treated as no data — the adhoc helper
            # will still fire if the relevant testcase is missing elsewhere.
            continue
        for testcase in tree.iter("testcase"):
            file_attr = testcase.get("file") or ""
            name_attr = testcase.get("name") or ""
            if file_attr and name_attr:
                completed.add((file_attr, name_attr))
    return completed


def _sanitize_for_s3_key(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(":", "_")
    )


def main() -> int:
    if not STEPCURRENT_DIR.is_dir():
        print(f"No stepcurrent cache at {STEPCURRENT_DIR}; nothing to emit.")
        return 0

    if not os.environ.get("JOB_ID") or not os.environ.get("GITHUB_RUN_ID"):
        # `upload_adhoc_failure_json` would early-return anyway, but log it
        # clearly so the CI step output flags the misconfiguration.
        print(
            "JOB_ID or GITHUB_RUN_ID not set in env; cannot emit adhoc rows. "
            "The composite action must pass `job-id` and set JOB_ID/GITHUB_RUN_ID."
        )
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

        parsed = parse_pytest_nodeid(lastrun)
        if parsed is None:
            print(f"Skipping unparseable nodeid: {lastrun!r}")
            continue
        test_file_path, invoking_file, classname, testname = parsed

        if (test_file_path, testname) in completed:
            # junit-xml already has a row for this test (it ran to completion
            # and a later test was killed). Don't duplicate.
            continue

        reason = (
            "Test step terminated before pytest could write junit-xml "
            "(SIGTERM/SIGKILL — e.g. timeout-minutes, run cancellation, "
            f"or OOM). Last test seen by stepcurrent: {lastrun}"
        )
        suffix = _sanitize_for_s3_key(f"incomplete_{classname}_{testname}")
        try:
            upload_adhoc_failure_json(
                invoking_file,
                lastrun,
                reason=reason,
                s3_key_suffix=suffix,
                classname=classname,
                testname=testname,
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
