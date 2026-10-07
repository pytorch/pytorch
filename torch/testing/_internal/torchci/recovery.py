"""Records the test a process was running when it died, for whoever started it:
run_test.py (``finish``) or a ``run_tests --subprocess`` parent
(``record_dead_subprocess``)."""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from torch.testing._internal.torchci import environment, report


def effective_exit_code(returned: int, elapsed: float, timeout: float | None) -> int:
    """124 for a process that timed out, else ``returned``. When pytest exits in
    the SIGINT grace period after a timeout, retry_shell returns its exit code
    instead: 2, or -2 if the SIGINT killed it."""
    if timeout is not None and elapsed >= timeout and returned in (2, -signal.SIGINT):
        return 124
    return returned


def _dead_run(inflight: dict[str, Any], exit_code: int) -> dict[str, Any]:
    # 124 is retry_shell's timeout.
    outcome = "timed_out" if exit_code == 124 else "crashed"
    return report.run_record(
        inflight["nodeid"],
        inflight["rerun_number"],
        outcome,
        inflight["started_at"],
        time.time(),
        outcome_summary=report.exit_summary(exit_code),
        declared_case_name=inflight.get("declared_case_name"),
    )


def finish(path: str, inflight: dict[str, Any] | None, exit_code: int) -> None:
    """Appends the run the writer published as in flight (plugin.py), unless the
    process wrote it before dying. A torn last line is ended first so it stays one
    unparsable line."""
    if not inflight:
        return
    run = _dead_run(inflight, exit_code)
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


def _rootdir_relative(path: str) -> str:
    # pytest's rootdir is the nearest directory with pytest.ini, the repo root.
    absolute = Path(path).resolve()
    for parent in absolute.parents:
        if (parent / "pytest.ini").is_file():
            return absolute.relative_to(parent).as_posix()
    return path


def record_dead_subprocess(
    directory: str, nodeid: str, exit_code: int, started: float
) -> None:
    """Writes a report of its own for a ``--subprocess`` child that crashed or timed
    out. ``nodeid`` is cwd-relative; the child's lines use the rootdir-relative one."""
    if not (exit_code < 0 or exit_code == 124):
        return
    path, sep, rest = nodeid.partition("::")
    nodeid = _rootdir_relative(path) + sep + rest
    inflight = {"nodeid": nodeid, "rerun_number": 0, "started_at": started}
    run = _dead_run(inflight, exit_code)
    report_uuid = str(uuid.uuid4())
    os.makedirs(directory, exist_ok=True)
    with open(report.report_path(directory, report_uuid), "w", encoding="utf-8") as f:
        f.write(report.line(report.report_record(report_uuid, environment.capture())))
        f.write(report.line(run))
