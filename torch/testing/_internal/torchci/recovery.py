"""Records the test a process was running when it died, for whoever started it:
run_test.py (``finish``) or a ``run_tests --subprocess`` parent
(``record_dead_subprocess``)."""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from typing import Any

from torch.testing._internal.torchci import environment, report


def effective_exit_code(returned: int, elapsed: float, timeout: float | None) -> int:
    """124 for a process that timed out, else ``returned``. When pytest exits in
    the SIGINT grace period after a timeout, retry_shell returns its exit code
    instead: 2, or -2 if the SIGINT killed it."""
    if timeout is not None and elapsed >= timeout and returned in (2, -signal.SIGINT):
        return 124
    return returned


def _dead_run(
    test: report.TestId, rerun_number: int, started: float, exit_code: int
) -> dict[str, Any]:
    # 124 is retry_shell's timeout.
    outcome = "timed_out" if exit_code == 124 else "crashed"
    summary = report.exit_summary(exit_code)
    return report.run_record(test, rerun_number, outcome, started, time.time(), summary)


def finish(path: str, inflight: dict[str, Any] | None, exit_code: int) -> None:
    """Appends the run the writer published as in flight (plugin.py), unless the
    process wrote it before dying. A torn last line is ended first so it stays one
    unparsable line."""
    if not inflight:
        return
    test = report.TestId(**inflight["test"])
    run = _dead_run(test, inflight["rerun_number"], inflight["started_at"], exit_code)
    with open(path, "ab+") as f:
        f.seek(0)
        content = f.read()
        for line in reversed(content.splitlines()):
            try:
                last = json.loads(line)
            except ValueError:
                continue
            if last.get("type") == "run":
                keys = ("file", "suite", "case_name", "rerun_number")
                if all(last.get(key) == run[key] for key in keys):
                    return
                break
        if content and not content.endswith(b"\n"):
            f.write(b"\n")
        f.write(report.line(run).encode())


def record_dead_subprocess(
    prefix: str, test: report.TestId, exit_code: int, started: float
) -> None:
    """Writes a report of its own for a ``--subprocess`` child that crashed or timed
    out."""
    if not (exit_code < 0 or exit_code == 124):
        return
    run = _dead_run(test, 0, started, exit_code)
    report_uuid = str(uuid.uuid4())
    path = report.report_path(prefix, report_uuid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.line(report.report_record(report_uuid, environment.capture())))
        f.write(report.line(run))
