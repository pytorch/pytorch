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
3. Has no matching `<testcase>` (keyed on xunit2 `(file, classname, name)`
   attributes, with classname compared by last segment to bridge xunit2's
   dotted module-qualified form vs the bare class segment in the nodeid)
   in any `test/test-reports/**/*.xml` already on disk.

Designed to be invoked from a GHA composite-action step gated explicitly on
the caller's `test-conclusion == 'failure' || 'cancelled'`, right before the
existing `upload-test-artifacts` zip+upload step, so it fires for ANY
test-step failure mode — not just SIGTERM — without competing with the
in-process `parse_xml_and_upload_json()` path or the `FAILED CONSISTENTLY`
path in `run_test.py`.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
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

# stepcurrent_key shapes from run_test.py:508-530:
#   <test_file>                         (cpp test, line 508)
#   <test_file>_<hex16>                 (cpp variant, line 522)
#   <test_file>_<shard>_<hex16>         (regular, line 530)
# `test_file` itself may contain underscores. Extract by stripping the
# `_<shard>?_<hex16>` suffix if present.
_STEPCURRENT_KEY_RE = re.compile(r"^(.+?)(?:_\d+)?_[a-f0-9]{16}$")


def _read_pytest_cache_value(path: Path) -> object | None:
    """pytest.Cache writes JSON files; missing/unparseable → None."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _invoking_file_from_stepcurrent_key(key: str) -> str:
    """Recover the run_test.py `test_file` (invocation unit) from the cache key."""
    m = _STEPCURRENT_KEY_RE.match(key)
    return m.group(1) if m else key


def _xunit_classname_last_segment(classname: str) -> str:
    """xunit2 emits `package.module.TestClass` for class tests, `package.module`
    for function tests. The "last segment" is `TestClass` (class tests) or
    the full path (function tests; we treat as no-class)."""
    if "." not in classname:
        return ""  # function-test xunit2 classname is just the module path
    return classname.rsplit(".", 1)[-1]


def _collect_completed_testcases() -> set[tuple[str, str, str]]:
    """Build `(file, last_class_segment, name)` from every junit-xml on disk.

    Match key bridges xunit2's dotted module-qualified `classname` against
    the bare class segment that `parse_pytest_nodeid` returns.
    """
    completed: set[tuple[str, str, str]] = set()
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
            classname = testcase.get("classname") or ""
            name_attr = testcase.get("name") or ""
            if file_attr and name_attr:
                completed.add(
                    (file_attr, _xunit_classname_last_segment(classname), name_attr)
                )
    return completed


def _sanitize_for_s3_key(s: str) -> str:
    # S3 keys allow most chars but we keep the existing path-safe shape and
    # avoid `:` / `/` / `[` / `]` / whitespace which can collide between
    # different parametrize values (`a:b` vs `a::b` vs `a/b`). The nodeid
    # hash appended in main() preserves distinguishability.
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
        test_file_path, _, classname, testname = parsed

        # Use the run_test.py invocation unit as `invoking_file` so this row
        # groups with other rows from the same test_file in downstream
        # consumers (HUD, autorevert). The pytest nodeid's path goes into
        # `file_attr` so the row still records the true source location.
        invoking_file = _invoking_file_from_stepcurrent_key(entry.name)

        if (test_file_path, classname, testname) in completed:
            # junit-xml already has a row for this test (it ran to completion
            # and a later test was killed). Don't duplicate.
            continue

        reason = (
            "Test step terminated before pytest could write junit-xml "
            "(SIGTERM/SIGKILL — e.g. timeout-minutes, run cancellation, "
            f"or OOM). Last test seen by stepcurrent: {lastrun}"
        )
        # Deterministic, collision-resistant s3 key: sanitized human-readable
        # prefix plus an 8-char hash of the full nodeid so parametrize values
        # like `a:b` vs `a::b` vs `a/b` don't collapse onto each other.
        prefix = _sanitize_for_s3_key(f"incomplete_{classname}_{testname}")
        digest = hashlib.sha256(lastrun.encode("utf-8")).hexdigest()[:8]
        suffix = f"{prefix}_{digest}"
        try:
            upload_adhoc_failure_json(
                invoking_file,
                lastrun,
                reason=reason,
                s3_key_suffix=suffix,
                classname=classname,
                testname=testname,
                file_attr=test_file_path,
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
