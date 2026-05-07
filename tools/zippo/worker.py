"""Persistent worker process for parallel pytest collection.

Each worker pays the cost of importing `torch` and `pytest` exactly once at
startup, then loops over file paths posted to a queue, invoking pytest's
collection API on each file and posting the resulting node IDs (or any error)
back to the coordinator. After processing `max_tasks` files, the worker exits
cleanly so the coordinator can spawn a fresh process and bound memory growth.
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from multiprocessing.queues import Queue


class _CollectReportCapture:
    """Plugin hook that records failed collection reports.

    Collection errors (e.g. ImportError at import time) do not raise out of
    `perform_collect` — they are surfaced via `pytest_collectreport` and stored
    in the session's report cache. We catch them here so the worker can
    surface them as failures back to the coordinator.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []

    def pytest_collectreport(self, report) -> None:  # type: ignore[no-untyped-def]
        if report.failed:
            longrepr = getattr(report, "longrepr", None)
            msg = "collection failed"
            if longrepr:
                # pytest's longrepr last line is typically `E   ExcType: msg`.
                # Strip the leading `E   ` for readability.
                last = str(longrepr).splitlines()[-1].lstrip()
                if last.startswith("E "):
                    last = last[1:].lstrip()
                msg = last or msg
            self.failures.append(msg)


def _collect_one(path: str) -> list[tuple[str, tuple[str, ...]]]:
    """Run pytest collection on a single file and return (node_id, markers) pairs.

    Builds a fresh pytest Config per file: pluginmanager / cache state / conftest
    bindings are all rootdir+args-scoped, so reusing a Config across files is
    not safe. The win this design captures is the `import torch` amortization.

    The session lifecycle here mirrors `_pytest.main.wrap_session`: configure,
    sessionstart, perform_collect, sessionfinish, unconfigure. Skipping
    `_do_configure`/`sessionstart` produces empty `session.items` because some
    initial-collection wiring runs in those hooks.

    Markers come from `item.iter_markers()`: function + class + module-level
    pytestmark, inherited along the class hierarchy. The worker reports them
    verbatim; any routing policy (defaulting unmarked tests, etc.) lives in
    the caller (`tools.zippo.cli`) so this stays policy-free.

    Raises RuntimeError if any collection report failed (e.g. ImportError on
    module load); the caller turns this into an "err" message.
    """
    from _pytest.config import _prepareconfig, ExitCode
    from _pytest.main import Session

    # Note: do NOT disable the `terminal` plugin. pytest's stock options like
    # `-rEfX` and `--tb=native` are registered by it, so projects that put
    # those in pytest.ini (e.g. PyTorch) will fail with "unrecognized
    # arguments" if it is missing. We silence pytest's output via stdout/err
    # redirection below instead.
    # Default `prepend` import mode is intentional: PyTorch's test tree relies
    # on it (each file's dir on sys.path so `from common_utils import X` etc.
    # work). The cost is that files with duplicate basenames across dirs
    # (test_checkpoint.py x2, test_utils.py x5, ...) can collide on the
    # second collect within a worker's lifetime and surface as collection
    # failures. The alternative (importlib mode) breaks far more files
    # because it doesn't put dirs on sys.path.
    capture = _CollectReportCapture()
    pytest_args = [path, "--collect-only", "-q", "-p", "no:cacheprovider"]
    config = _prepareconfig(args=pytest_args, plugins=[capture])
    try:
        session = Session.from_config(config)
        session.exitstatus = ExitCode.OK
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            config._do_configure()
            try:
                config.hook.pytest_sessionstart(session=session)
                session.perform_collect(args=[path], genitems=True)
            finally:
                config.hook.pytest_sessionfinish(
                    session=session, exitstatus=session.exitstatus
                )
        if capture.failures:
            raise RuntimeError("; ".join(capture.failures))
        return [
            (item.nodeid, tuple(m.name for m in item.iter_markers()))
            for item in session.items
        ]
    finally:
        # Always unconfigure. Skipping it looks like a 2x speedup in small
        # microbenchmarks (it does ~5s of GC on test_meta.py's 83k items),
        # but at scale (1000+ files * 65 workers) the unfreed Configs and
        # plugin state accumulate per worker and the run grinds to a halt.
        config._ensure_unconfigure()


def worker_main(
    task_q: Queue,
    result_q: Queue,
    max_tasks: int,
) -> None:
    """Worker entry point.

    Sends three message kinds via `result_q`:
        ("start", pid, path)            - heartbeat before collection begins
        ("ok",    pid, path, entries)   - successful collection; entries is a
                                          list of (node_id, marker_names) tuples
        ("err",   pid, path, msg)       - collection raised; worker keeps going
    """
    # NOTE: do NOT set CUDA_VISIBLE_DEVICES="" here. PyTorch's
    # `instantiate_device_type_tests` reads `torch.cuda.is_available()` at
    # collection time and only emits CUDA test variants if it's True. Forcing
    # CPU-only here silently undercounts tests by hundreds per file.

    # Permanently silence the worker's stdout/stderr at the file-descriptor
    # level. The coordinator owns the user-facing terminal, so anything any
    # plugin or C extension writes here would otherwise interleave with the
    # progress line. fd-level redirect (vs `contextlib.redirect_stderr`) is
    # required because pytest plugins cache `sys.stderr` during config init,
    # and the warnings module writes through that cached reference.
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    # Pay heavy import costs once per worker lifetime. torch is best-effort:
    # if it is not importable we still want the tool to work.
    #
    # When zippo runs from the pytorch repo (e.g. `python -m tools.zippo` from
    # /home/.../pytorch), Python sets sys.path[0] to that absolute directory,
    # which causes `import torch` to find the source tree's `torch/` package
    # before the editable install's compiled artifacts and raise. Drop the
    # repo root from sys.path for the import, then put it back so subsequent
    # `tools.*` imports keep resolving.
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _shadowing = {_repo_root, "", "."}
    _saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if p not in _shadowing]
    try:
        try:
            import torch  # noqa: F401
        except BaseException:
            pass
        import pytest  # noqa: F401
    finally:
        sys.path[:] = _saved_path

    pid = os.getpid()
    tasks_done = 0
    while True:
        path = task_q.get()
        if path is None:
            return
        result_q.put(("start", pid, path))
        try:
            entries = _collect_one(path)
            result_q.put(("ok", pid, path, entries))
        except RuntimeError as e:
            # Collection-report failures surface as RuntimeError from
            # `_collect_one` — its message is already the cleaned-up pytest
            # error, so don't double-wrap it.
            result_q.put(("err", pid, path, str(e)))
        except BaseException as e:
            # Anything else (SystemExit from pytest internals, unexpected
            # errors) — keep the worker alive and report.
            tb = traceback.format_exception_only(type(e), e)
            msg = "".join(tb).strip() or f"{type(e).__name__}: {e}"
            result_q.put(("err", pid, path, msg))

        tasks_done += 1
        if max_tasks and tasks_done >= max_tasks:
            return
