"""Compile-time TunableOp support for GEMMs.

Covers both backends TunableOp supports -- hipBLASLt on ROCm and cuBLASLt on
CUDA -- which is why the gate is torch.cuda.is_available() rather than a ROCm
check, and why the propagated settings include a cuBLASLt-only knob.

TunableOp keys solutions on an exact shape signature, which models lowered with
dynamic shapes rarely hit at runtime; `tune` mode widens the recorded coverage and
`reuse` mode borrows and validates a neighbour's solution, so execution only ever
needs an exact match. Both benchmark in the dedicated worker owned by this module,
because hipBLASLt tuning in particular can abort the process outright or hang.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, Literal, NamedTuple, TYPE_CHECKING

import torch

from . import config


if TYPE_CHECKING:
    from .autotune_process import TuningProcess


log = logging.getLogger(__name__)


WorkerStatus = Literal["ok", "timeout", "crash", "error"]

_valid_modes = ("off", "tune", "reuse")
_mode_warned = False


def _mode() -> str:
    """The configured mode, warning once on values that silently mean "off".

    config.cuda.aten_tunableop_mode accepts any string from
    TORCHINDUCTOR_ATEN_TUNABLEOP_MODE,
    so a typo like "Tune" or "on" would otherwise disable both passes silently.
    """
    global _mode_warned
    # Normalized here, not at the env read: this covers a value set directly on
    # the config too, and "Tune" would otherwise disable both passes.
    mode = config.cuda.aten_tunableop_mode.lower()
    if mode not in _valid_modes and not _mode_warned:
        _mode_warned = True
        log.warning(
            "Unknown TORCHINDUCTOR_ATEN_TUNABLEOP_MODE=%r (expected one of %s); "
            'treating as "off"',
            mode,
            ", ".join(_valid_modes),
        )
    return mode


# PYTORCH_TUNABLEOP_TUNING=0 and PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 are read
# through static env checks that win over the setters (Tunable.cpp), and the
# worker inherits the parent's environment, so tune mode would benchmark every
# GEMM, report success, and persist nothing.
_BLOCKING_ENV = (
    ("PYTORCH_TUNABLEOP_TUNING", "0"),
    ("PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1"),
)
_env_warned = False


def _warn_if_env_blocks_recording() -> None:
    global _env_warned
    if _env_warned:
        return
    blocking = [name for name, bad in _BLOCKING_ENV if os.environ.get(name) == bad]
    if blocking:
        _env_warned = True
        log.warning(
            "TunableOp tune mode will record nothing: %s overrides the tuning "
            "flags the worker sets, so no result is written. Unset it to tune.",
            " and ".join(blocking),
        )


def is_tune_mode() -> bool:
    """True while building the results file: tune GEMMs and record solutions."""
    if _mode() != "tune" or not torch.cuda.is_available():
        return False
    _warn_if_env_blocks_recording()
    return True


def is_reuse_mode() -> bool:
    """True while consuming the results file: borrow and validate, never tune."""
    return _mode() == "reuse" and torch.cuda.is_available()


def is_enabled() -> bool:
    return is_tune_mode() or is_reuse_mode()


class BenchmarkJob:
    """One GEMM benchmark, executed in the TunableOp worker.

    Picklable so it can cross the worker pipe: a BenchmarkRequest carries only tensor
    metadata, and the worker materializes its own tensors. TunableOp's flags are
    process-global, which is why the worker must stay dedicated (see
    _get_worker_locked).
    """

    def __init__(
        self,
        bmreq: Any,
        tuning: bool,
        nearest_match: bool = False,
        dyn_dims_mask: tuple[bool, bool, bool, bool] | None = None,
    ) -> None:
        self.bmreq = bmreq
        self.tuning = tuning
        self.nearest_match = nearest_match
        # (M, N, K, BATCH). Thread-local in the parent, so unlike the knobs in
        # settings it cannot be snapshotted -- the caller passes it explicitly
        # and __call__ re-enters it around the benchmark.
        self.dyn_dims_mask = dyn_dims_mask

    def __call__(self, device: int, filename: str, settings: dict[str, Any]) -> float:
        tunable = torch.cuda.tunable
        # The worker starts on device 0 of the inherited mask, not the job's.
        torch.cuda.set_device(device)
        # Before anything touches the manager: it initializes lazily from the
        # env default, not the parent's file.
        if filename:
            tunable.set_filename(filename, insert_device_ordinal=False)
        tunable.enable(True)
        tunable.tuning_enable(self.tuning)
        tunable._nearest_match_enable(self.nearest_match)
        for name, value in settings.items():
            getattr(tunable, name)(value)
        # benchmark_run_fn captures if either is set, and tuning cannot run
        # under capture.
        self.bmreq.benchmark_with_cudagraphs = False
        self.bmreq.config_cudagraph_benchmarking = False
        # Thread-local in the setting process, so it does not cross the pipe.
        mask = self.dyn_dims_mask
        try:
            if mask is not None and any(mask):
                with tunable.dynamic_dims_mask(
                    M=mask[0], N=mask[1], K=mask[2], BATCH=mask[3]
                ):
                    return self.bmreq.benchmark()
            return self.bmreq.benchmark()
        finally:
            # The worker outlives this job, so without this it sits idle
            # holding what the benchmark allocated. The context stays either way.
            try:
                torch.cuda.empty_cache()
            except Exception:
                log.debug("TunableOp worker empty_cache failed", exc_info=True)


class Outcome(NamedTuple):
    """Result of a worker submission.

    `status` is "ok", "timeout" (worker hung and was killed), "crash" (worker died,
    typically a GPU fault) or "error" (the job raised). `time` is meaningful only
    when status is "ok".
    """

    time: float | None
    status: WorkerStatus


_worker: TuningProcess | None = None
_worker_key: tuple[int, str] | None = None
_atexit_registered = False
# How long a graceful shutdown waits for the child before resorting to SIGKILL.
_SHUTDOWN_TIMEOUT_S = 10.0
# One worker, one pipe. Extern kernels are benchmarked sequentially today
# (select_algorithm routes them to benchmark_in_current_process, not the thread
# pool), but nothing structural enforces that, and interleaved put/get would
# either corrupt the pickle stream or hand a caller another job's timing.
# TuningProcessPool guards its processes with a queue for the same reason.
_worker_lock = threading.Lock()


def _get_worker_locked(key: tuple[int, str]) -> TuningProcess:
    """The long-lived process that runs every TunableOp benchmark for this compile.

    Assumes _worker_lock is held.

    Dedicated rather than the shared TuningProcessPool because TunableOp's flags
    are process-global: a shared worker would tune unrelated GEMMs into the
    results file. Reused across GEMMs because startup dominates, and torn down
    only on a key change or at exit, so an idle child holds its context.

    Keyed on (device, resolved results filename): a restart is what releases the
    old GPU's context and the results map the worker loaded at startup.
    """
    global _worker, _worker_key, _atexit_registered
    from .autotune_process import TuningProcess

    if _worker is not None and _worker_key != key:
        _shutdown_locked()
    if _worker is None:
        # device=None on purpose. TuningProcess.start() *replaces* the inherited
        # CUDA_VISIBLE_DEVICES with str(device), so a logical ordinal would point
        # the worker at a different physical GPU: with CUDA_VISIBLE_DEVICES=4,7
        # and logical 1, the parent uses physical 7 and the worker takes physical
        # 1. Leaving the mask alone keeps UUID/MIG tokens intact; BenchmarkJob
        # selects the logical device on arrival.
        _worker = TuningProcess(device=None)
        _worker_key = key
        if not _atexit_registered:
            atexit.register(shutdown_worker)
            _atexit_registered = True
    return _worker


def shutdown_worker() -> None:
    """Shut down the worker. Callers holding _worker_lock use _shutdown_locked."""
    with _worker_lock:
        _shutdown_locked()


def _reap(worker: TuningProcess) -> None:
    """Wait briefly on an already-signalled child so it does not linger.

    TuningProcess.kill() sends SIGKILL and closes the pipes but never calls
    process.wait(), so without this the child stays a zombie until GC collects
    the Popen. Bounded, because this runs under _worker_lock and from atexit.
    """
    try:
        worker.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
    except Exception:
        log.debug("TunableOp worker did not exit within %ss", _SHUTDOWN_TIMEOUT_S)


def _shutdown_locked(graceful: bool = True) -> None:
    """Assumes _worker_lock is held (_get_worker_locked calls it on key change).

    graceful=False kills outright, for a suspected GPU fault: once the
    reference is dropped there is no handle left to kill the child with.
    Either way it is signalled, waited on with a bound, and its pipes closed
    before the reference goes.
    """
    global _worker, _worker_key
    if _worker is not None:
        try:
            if graceful:
                _worker.shutdown(wait=False)
                try:
                    if _worker.alive():
                        _worker.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    log.debug(
                        "TunableOp worker did not exit in %ss; killing",
                        _SHUTDOWN_TIMEOUT_S,
                    )
                    _worker.kill()
                    _reap(_worker)
                else:
                    _worker.close()
            else:
                # kill() signals only when alive() is true, and alive() is gated
                # on running, which close() already cleared on the EOF path. A
                # child that closed its pipes and then wedged in device teardown
                # would never be signalled, and would hold its context and
                # allocator memory on the GPU with no handle left to reach it.
                process = _worker.process
                if process is not None and process.poll() is None:
                    process.kill()
                _worker.kill()
                _reap(_worker)
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


# Parameters of the search, stamped onto each job: (parent getter, worker
# setter) pairs. These are the knobs a caller can set through the API, which
# does not cross the pipe. Env-sourced values reach the worker by inheritance
# instead -- which is not always benign, see _warn_if_env_blocks_recording.
#
# Which operation to run is decided per job from the Inductor mode, and is
# carried as BenchmarkJob fields rather than stamped here. Absent for that
# reason:
#   * record_untuned selects the offline-tuning workflow, which is an
#     alternative to tuning here rather than a modifier of it. Stamping it
#     would leave a tune-mode worker tuning every GEMM and then persisting
#     none of them, since AddAndMaybePersist writes only when tuning is on and
#     record-untuned is off.
#   * the dynamic-dims mask is per call, not per process, so the job carries it
#     (dyn_dims_mask).
# Numerical-check tolerances are absent for a duller reason: no getter exists.
_TUNABLE_SETTINGS: tuple[tuple[str, str], ...] = (
    ("get_max_tuning_duration", "set_max_tuning_duration"),
    ("get_max_tuning_iterations", "set_max_tuning_iterations"),
    ("get_rotating_buffer_size", "set_rotating_buffer_size"),
    ("get_cublaslt_requested_algo_count", "set_cublaslt_requested_algo_count"),
    # Decides what a non-tuning lookup can find, so the worker must search the
    # same catalogue the parent would.
    ("wildcard_fallback_is_enabled", "wildcard_fallback_enable"),
)


def _rotating_buffer_mb(value: int) -> int:
    """Convert get_rotating_buffer_size()'s bytes to the MB the setter wants.

    Unset, the getter reports an L2 size, and a plain conversion would turn
    "use L2" into a rounded-down constant -- 0, meaning no rotation at all,
    below 1 MiB. Any negative value restores "use L2". Compared against the
    current device because that is the one GetRotatingBufferSize read, which
    is not necessarily the job's device.
    """
    current = torch.cuda.current_device()
    if value == torch.cuda.get_device_properties(current).L2_cache_size:
        return -1
    return value // (1024 * 1024)


# Getters whose units differ from the matching setter's.
_SETTING_CONVERSIONS: dict[str, Any] = {
    "set_rotating_buffer_size": _rotating_buffer_mb,
}


def snapshot_settings() -> dict[str, Any]:
    """Read the parent's tuning knobs for stamping onto the job.

    The worker is long-lived and keyed only on (device, filename), so it never
    sees settings changed after it starts; stamping per job keeps it in sync.
    Best-effort per knob: a getter failure skips it, leaving whatever the
    previous job set, rather than failing the submission.
    """
    tunable = torch.cuda.tunable
    settings: dict[str, Any] = {}
    for getter, setter in _TUNABLE_SETTINGS:
        try:
            value = getattr(tunable, getter)()
            convert = _SETTING_CONVERSIONS.get(setter)
            settings[setter] = convert(value) if convert else value
        except Exception:
            log.debug("TunableOp setting snapshot failed for %s", getter)
    return settings


def resolve_and_pin_filename(device: int) -> str:
    """The results file, committed to so the parent cannot later pick another.

    Assumes _worker_lock is held. Until the manager initializes it is free to
    choose its own ordinal on first use, so a worker told anything else would
    write where the runtime never reads -- and a process compiling for two
    devices would strand one graph's results. Setting it removes the guess:
    GetTuningResultsManager skips its own derivation once filename_ is set.
    """
    tunable = torch.cuda.tunable
    filename = tunable.get_filename()
    if not filename:
        filename = _insert_device_ordinal(
            os.environ.get("PYTORCH_TUNABLEOP_FILENAME", "tunableop_results.csv"),
            device,
        )
        tunable.set_filename(filename, insert_device_ordinal=False)
    return filename


def _job_device(job: BenchmarkJob) -> int:
    """The device the job's tensors live on, which is the one to tune on.

    Not current_device(): Inductor scopes V.graph.set_current_device, which is
    graph bookkeeping, and compile_fx's CUDA guard covers post-grad passes
    rather than lowering, so a graph targeting cuda:1 can be autotuned while
    the process current device is 0. The child builds its tensors from these
    metas, so this is the GPU it really tunes on, and the ordinal an
    uninitialized results manager will name its file for. Mirrors
    BenchmarkRequest._get_benchmark_device, including its fall back to the
    current device when nothing carries one.
    """
    bmreq = job.bmreq
    metas = [getattr(bmreq, "output_tensor_meta", None)]
    metas += list(getattr(bmreq, "input_tensor_meta", None) or [])
    for meta in metas:
        index = getattr(getattr(meta, "device", None), "index", None)
        if index is not None:
            return index
    return torch.cuda.current_device()


class _StampedJob:
    """What crosses the pipe: a job plus the state submit() resolved for it.

    Keeping it here rather than on BenchmarkJob means submit() does not mutate
    its argument, and a job carries only what its caller chose.
    """

    def __init__(
        self, job: Any, device: int, filename: str, settings: dict[str, Any]
    ) -> None:
        self.job = job
        self.device = device
        self.filename = filename
        self.settings = settings

    def __call__(self) -> float:
        return self.job(self.device, self.filename, self.settings)


def submit(job: BenchmarkJob, timeout: float | None = None) -> Outcome:
    """Run one job in the worker, bounding how long we wait for it.

    A hang or crash leaves the worker dead; the next submission restarts it and the
    restarted process re-reads the results file, so recorded solutions survive.
    Faults are reported rather than raised: a compile must continue even when a GEMM
    cannot be benchmarked.
    """
    if timeout is None:
        timeout = config.cuda.aten_tunableop_timeout
    try:
        # Inside the try: this reports faults rather than raising, and reading
        # the device can fail just as the benchmark can.
        device = _job_device(job)
        settings = snapshot_settings()
    except Exception as e:  # resolve before the worker: no restart
        # Failed before the worker was involved (no device, no filename): report
        # without touching it. Shutting down here would kill a healthy worker
        # another thread is using, and -- worse -- write the None sentinel into
        # the pipe mid-stream after _worker_lock is released.
        log.debug("TunableOp benchmark setup failed: %s", e)
        return Outcome(None, "error")
    try:
        with _worker_lock:
            # Under the lock: two threads must not pin different filenames.
            filename = resolve_and_pin_filename(device)
            stamped = _StampedJob(job, device, filename, settings)
            worker = _get_worker_locked((device, filename))
            try:
                worker.put(stamped)
                return Outcome(worker.get(timeout=timeout), "ok")
            except (TimeoutError, EOFError) as e:
                # running is cleared only by a real pipe fault; a job that
                # itself raises TimeoutError leaves it set and must be filed as
                # an error so it still gets the sticky-fault restart.
                if not worker.running:
                    status: WorkerStatus = (
                        "timeout" if isinstance(e, TimeoutError) else "crash"
                    )
                    # get() killed or closed it but never waited; reap it.
                    _shutdown_locked(graceful=False)
                    return Outcome(None, status)
                log.debug("TunableOp job raised %r; treating as an error", e)
                _shutdown_locked(graceful=False)
                return Outcome(None, "error")
            except Exception as e:  # a failed benchmark must not fail the compile
                log.debug("TunableOp benchmark failed: %s", e)
                # A caught exception can leave a sticky GPU fault behind, so
                # every later GEMM would fail the same way. Kill rather than
                # ask: a child wedged in device teardown never acts on the
                # sentinel. Recorded solutions survive via the results file.
                _shutdown_locked(graceful=False)
                return Outcome(None, "error")
    except Exception as e:  # lock/restart machinery itself failed
        log.debug("TunableOp worker machinery failed: %s", e)
        return Outcome(None, "error")
