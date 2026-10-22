# mypy: allow-untyped-defs
"""Compile-time TunableOp support for ROCm hipBLASLt GEMMs.

TunableOp keys solutions on an exact shape signature, which models lowered with
dynamic shapes rarely hit at runtime; `tune` mode widens the recorded coverage and
`reuse` mode borrows and validates a neighbour's solution, so execution only ever
needs an exact match. Both benchmark in the dedicated worker owned by this module,
because hipBLASLt tuning can abort the process outright or hang.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, NamedTuple

import torch

from . import config


log = logging.getLogger(__name__)


def is_tune_mode() -> bool:
    """True while building the results file: tune GEMMs and record solutions."""
    return config.aten_tunableop_mode == "tune" and torch.cuda.is_available()


def is_reuse_mode() -> bool:
    """True while consuming the results file: borrow and validate, never tune."""
    return config.aten_tunableop_mode == "reuse" and torch.cuda.is_available()


def is_enabled() -> bool:
    return is_tune_mode() or is_reuse_mode()


class BenchmarkJob:
    """One GEMM benchmark, executed in the TunableOp worker.

    Picklable so it can cross the worker pipe: a BenchmarkRequest carries only tensor
    metadata, and the worker materializes its own tensors. TunableOp's flags are
    process-global, which is why the worker must stay dedicated (see get_worker).
    """

    def __init__(self, bmreq: Any, tuning: bool, nearest_match: bool = False) -> None:
        self.bmreq = bmreq
        self.tuning = tuning
        self.nearest_match = nearest_match
        # Stamped by submit() with the submitting process's logical device.
        self.device = 0

    def __call__(self) -> float:
        tunable = torch.cuda.tunable
        # The worker starts on device 0 of the inherited visibility mask, which
        # is not necessarily the device being compiled for. TunableOp's results
        # file is per device ordinal, so getting this wrong both benchmarks the
        # wrong GPU and writes to the wrong file.
        torch.cuda.set_device(self.device)
        tunable.enable(True)
        tunable.tuning_enable(self.tuning)
        tunable._nearest_match_enable(self.nearest_match)
        # TunableOp cannot tune under graph capture.
        self.bmreq.benchmark_with_cudagraphs = False
        return self.bmreq.benchmark()


class Outcome(NamedTuple):
    """Result of a worker submission.

    `status` is "ok", "timeout" (worker hung and was killed), "crash" (worker died,
    typically a GPU fault) or "error" (the job raised). `time` is meaningful only
    when status is "ok".
    """

    time: float | None
    status: str


_worker: Any = None
_worker_key: tuple[int, str] | None = None
_atexit_registered = False
# One worker, one pipe. Extern kernels are benchmarked sequentially today
# (select_algorithm routes them to benchmark_in_current_process, not the thread
# pool), but nothing structural enforces that, and interleaved put/get would
# either corrupt the pickle stream or hand a caller another job's timing.
# TuningProcessPool guards its processes with a queue for the same reason.
_worker_lock = threading.Lock()


def get_worker(key: tuple[int, str]) -> Any:
    """The long-lived process that runs every TunableOp benchmark for this compile.

    Deliberately dedicated rather than Inductor's shared TuningProcessPool:
    TunableOp's flags are process-global, so a worker that also benchmarks triton
    candidates would tune unrelated GEMMs and append them to the results file. One
    worker per compile rather than one per GEMM, since startup dominates otherwise.

    Keyed on (device, results filename). The results file is per device ordinal, so
    a worker left pinned to an earlier device would benchmark the wrong GPU; and it
    reads that file once at startup, so a second compile that points
    PYTORCH_TUNABLEOP_FILENAME somewhere else would otherwise keep reading and
    appending to the first compile's file.
    """
    global _worker, _worker_key, _atexit_registered
    from .autotune_process import TuningProcess

    if _worker is not None and _worker_key != key:
        shutdown_worker()
    if _worker is None:
        # device=None on purpose. TuningProcess.start() *replaces* the inherited
        # CUDA_VISIBLE_DEVICES with str(device), so handing it a logical ordinal
        # points the worker at a different physical GPU -- with
        # CUDA_VISIBLE_DEVICES=4,7 and logical device 1, the parent uses physical
        # 7 and the worker would take physical 1, outside the job's allocation.
        # Inductor's own pool avoids this by passing the visibility tokens
        # themselves (autotune_process.get_device_list). Leaving the mask alone is
        # simpler and keeps UUID/MIG tokens intact, since the subprocess inherits
        # os.environ; BenchmarkJob then selects the logical device on arrival.
        _worker = TuningProcess(device=None)
        _worker_key = key
        if not _atexit_registered:
            atexit.register(shutdown_worker)
            _atexit_registered = True
    return _worker


def shutdown_worker() -> None:
    global _worker, _worker_key
    if _worker is not None:
        try:
            _worker.shutdown(wait=False)
        except Exception:
            log.debug("TunableOp worker shutdown failed", exc_info=True)
        _worker = None
    _worker_key = None


def submit(job: BenchmarkJob, timeout: float | None = None) -> Outcome:
    """Run one job in the worker, bounding how long we wait for it.

    A hang or crash leaves the worker dead; the next submission restarts it and the
    restarted process re-reads the results file, so recorded solutions survive.
    Faults are reported rather than raised: a compile must continue even when a GEMM
    cannot be benchmarked.
    """
    if timeout is None:
        timeout = config.aten_tunableop_timeout
    try:
        # Inside the try with everything else: the contract above is that this
        # function reports faults rather than raising, and current_device() is
        # as able to fail as the benchmark is.
        device = torch.cuda.current_device()
        # The worker inherits this process's visibility mask rather than having
        # it rewritten, so the same logical ordinal names the same GPU there.
        job.device = device
        with _worker_lock:
            worker = get_worker((device, torch.cuda.tunable.get_filename()))
            worker.put(job)
            return Outcome(worker.get(timeout=timeout), "ok")
    except TimeoutError:
        return Outcome(None, "timeout")
    except EOFError:
        return Outcome(None, "crash")
    except Exception as e:  # noqa: BLE001 - a failed benchmark must not fail the compile
        log.debug("TunableOp benchmark failed: %s", e)
        return Outcome(None, "error")
