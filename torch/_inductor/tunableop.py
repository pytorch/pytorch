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
import os
import threading
from typing import Any, Literal, NamedTuple

import torch

from . import config


log = logging.getLogger(__name__)


# Outcome of a worker submission: "ok", "timeout" (worker hung and was killed),
# "crash" (worker died, typically a GPU fault) or "error" (the job raised).
WorkerStatus = Literal["ok", "timeout", "crash", "error"]

_valid_modes = ("off", "tune", "reuse")
_mode_warned = False


def _mode() -> str:
    """The configured mode, warning once on values that silently mean "off".

    config.aten_tunableop_mode accepts any string from INDUCTOR_ATEN_TUNABLEOP_MODE,
    so a typo like "Tune" or "on" would otherwise disable both passes silently.
    """
    global _mode_warned
    mode = config.aten_tunableop_mode
    if mode not in _valid_modes and not _mode_warned:
        _mode_warned = True
        log.warning(
            "Unknown INDUCTOR_ATEN_TUNABLEOP_MODE=%r (expected one of %s); "
            'treating as "off"',
            mode,
            ", ".join(_valid_modes),
        )
    return mode


def is_tune_mode() -> bool:
    """True while building the results file: tune GEMMs and record solutions."""
    return _mode() == "tune" and torch.cuda.is_available()


def is_reuse_mode() -> bool:
    """True while consuming the results file: borrow and validate, never tune."""
    return _mode() == "reuse" and torch.cuda.is_available()


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
        # Stamped by submit() with the resolved results filename (see
        # resolve_filename): the worker re-enters it via set_filename so it
        # reads and appends to the same file the parent's key names.
        self.filename = ""

    def __call__(self) -> float:
        tunable = torch.cuda.tunable
        # The worker starts on device 0 of the inherited visibility mask, which
        # is not necessarily the device being compiled for. TunableOp's results
        # file is per device ordinal, so getting this wrong both benchmarks the
        # wrong GPU and writes to the wrong file.
        torch.cuda.set_device(self.device)
        # The worker is a fresh process whose results manager initializes lazily
        # from the env/default file -- not the parent's file. Re-enter the
        # filename resolved in submit() before anything touches the manager, or
        # the worker reads and appends to the wrong file.
        if self.filename:
            tunable.set_filename(self.filename, insert_device_ordinal=False)
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
    status: WorkerStatus


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

    Keyed on (device, resolved results filename). The results file is per
    device ordinal, so a worker left pinned to an earlier device would
    benchmark the wrong GPU; and it reads that file once at startup, so a
    second compile that points PYTORCH_TUNABLEOP_FILENAME somewhere else would
    otherwise keep reading and appending to the first compile's file. The
    filename comes from submit() via resolve_filename() -- never "" -- because
    get_filename() is empty until the manager initializes.
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


def _insert_device_ordinal(filename: str, device: int) -> str:
    """Mirror of TuningContext::SetFilename(filename, insert_device_ordinal=true)."""
    ordinal = str(device)
    if "%d" in filename:
        return filename.replace("%d", ordinal, 1)
    dot = filename.rfind(".")
    if dot != -1:
        return filename[:dot] + ordinal + filename[dot:]
    return filename + ordinal


def resolve_filename(device: int) -> str:
    """The results filename the worker must use, resolved like the C++ manager.

    get_filename() returns "" until set_filename() is called or the results
    manager initializes, which mirrors the C++ default rule (GetTuningResultsManager:
    PYTORCH_TUNABLEOP_FILENAME env, else "tunableop_results.csv", then the device
    ordinal inserted). Resolving it here -- rather than letting the worker lazily
    re-derive it -- is what makes the worker key meaningful: without this, the key
    stays "" across set_filename calls and env changes, so no restart happens and
    the worker keeps appending to the first file.

    The ordinal must be inserted here too: the worker re-enters the name with
    insert_device_ordinal=False, so skipping it makes the worker write
    "tunableop.csv" while the runtime manager reads "tunableop0.csv".
    """
    filename = torch.cuda.tunable.get_filename()
    if not filename:
        filename = _insert_device_ordinal(
            os.environ.get("PYTORCH_TUNABLEOP_FILENAME", "tunableop_results.csv"),
            device,
        )
    return filename


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
        # Stamp the resolved filename so __call__ re-enters it via set_filename:
        # the worker is a fresh process and would otherwise lazily initialize
        # its manager from the env/default file, not the parent's file.
        job.filename = resolve_filename(device)
        with _worker_lock:
            worker = get_worker((device, job.filename))
            worker.put(job)
            return Outcome(worker.get(timeout=timeout), "ok")
    except TimeoutError:
        return Outcome(None, "timeout")
    except EOFError:
        return Outcome(None, "crash")
    except Exception as e:  # a failed benchmark must not fail the compile
        log.debug("TunableOp benchmark failed: %s", e)
        # A catchable exception can leave a sticky fault behind (e.g. an illegal
        # address corrupting the GPU context): the worker stays alive, so every
        # later GEMM would fail the same way. Restart it so the next submission
        # gets a clean process; recorded solutions survive via the results file.
        shutdown_worker()
        return Outcome(None, "error")
