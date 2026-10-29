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
from torch.utils._ordered_set import OrderedSet

from . import config


if TYPE_CHECKING:
    from .autotune_process import TuningProcess


log = logging.getLogger(__name__)


WorkerStatus = Literal["ok", "timeout", "crash", "error"]

_valid_modes = ("off", "tune", "reuse")
_mode_warned = False


def _mode() -> str:
    """The configured mode, warning once on values that silently mean "off".

    TORCHINDUCTOR_AUTOTUNE_TUNABLEOP_MODE is taken as given, so an unknown value
    like "on" would otherwise disable both passes with no signal.
    """
    global _mode_warned
    configured = config.cuda.autotune_tunableop_mode
    # Normalized here, not at the env read, so a value set on the config is
    # covered too.
    mode = configured.lower()
    if mode not in _valid_modes and not _mode_warned:
        _mode_warned = True
        log.warning(
            "Unknown TORCHINDUCTOR_AUTOTUNE_TUNABLEOP_MODE=%r (expected one of %s); "
            'treating as "off"',
            configured,
            ", ".join(_valid_modes),
        )
    return mode


# Read through static env checks that win over the setters (Tunable.cpp), and
# the worker inherits the parent's environment, so neither pass can undo them.
# Tune: either one stops AddAndMaybePersist writing a row.
_TUNE_BLOCKING_ENV = (
    ("PYTORCH_TUNABLEOP_TUNING", "0"),
    ("PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1"),
)
# Reuse: TUNING=0 is harmless, it never tunes. These each break it --
# RECORD_UNTUNED skips ReadFile, NEAREST_MATCH_ENABLED=0 stops borrowing, and
# RECORD_BORROWED_ENABLED=0 keeps the validated winner out of the results file.
_REUSE_BLOCKING_ENV = (
    ("PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1"),
    ("PYTORCH_TUNABLEOP_NEAREST_MATCH_ENABLED", "0"),
    ("PYTORCH_TUNABLEOP_RECORD_BORROWED_ENABLED", "0"),
)
_env_warned: OrderedSet[str] = OrderedSet()
_warned_faults: OrderedSet[str] = OrderedSet()


def _warn_fault_once(status: WorkerStatus, detail: str) -> None:
    """A fault repeats for every GEMM; warn once per kind, then drop to debug."""
    kind = f"{status}:{detail}"
    report = log.debug if kind in _warned_faults else log.warning
    _warned_faults.add(kind)
    report("TunableOp benchmark %s: %s", status, detail, exc_info=True)


def _warn_if_env_blocks(
    blocking_env: tuple[tuple[str, str], ...], consequence: str
) -> None:
    blocking = [name for name, bad in blocking_env if os.environ.get(name) == bad]
    # Keyed per mode as well as per variable: one mode's warning must not
    # silence the other's, and a variable that blocks both says something
    # different about each.
    kind = f"{consequence}:{','.join(blocking)}"
    if blocking and kind not in _env_warned:
        _env_warned.add(kind)
        log.warning(
            "TunableOp %s: %s overrides the flags the worker sets. Unset it.",
            consequence,
            " and ".join(blocking),
        )


def is_tune_mode() -> bool:
    """True while building the results file: tune GEMMs and record solutions."""
    if _mode() != "tune" or not torch.cuda.is_available():
        return False
    _warn_if_env_blocks(_TUNE_BLOCKING_ENV, "tune mode will record nothing")
    return True


def is_reuse_mode() -> bool:
    """True while consuming the results file: borrow and validate, never tune."""
    if _mode() != "reuse" or not torch.cuda.is_available():
        return False
    _warn_if_env_blocks(_REUSE_BLOCKING_ENV, "reuse mode will not borrow or record")
    return True


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
# One worker, one pipe: interleaved put/get would corrupt the pickle stream or
# hand a caller another job's timing. TuningProcessPool uses a queue likewise.
_worker_lock = threading.Lock()


def _get_worker_locked(key: tuple[int, str]) -> TuningProcess:
    """The process that runs every TunableOp benchmark. Module-global, so it
    spans compiles -- nothing marks their boundaries.

    Assumes _worker_lock is held.

    Dedicated rather than the shared TuningProcessPool because TunableOp's flags
    are process-global: a shared worker would tune unrelated GEMMs into the
    results file. Reused across GEMMs because startup dominates.

    Torn down on a key change, on any fault in submit(), and at exit. Keyed on
    (device, resolved results filename) so a restart releases the old GPU's
    context and the results map the child loaded at startup.
    """
    global _worker, _worker_key, _atexit_registered
    from .autotune_process import TuningProcess

    if _worker is not None and _worker_key != key:
        _shutdown_locked()
    if _worker is None:
        # device=None on purpose. TuningProcess.start() *replaces* the inherited
        # CUDA_VISIBLE_DEVICES with str(device), so a logical ordinal would point
        # the worker at a different physical GPU: with CUDA_VISIBLE_DEVICES=4,7
        # and logical 1, the parent uses physical 7 and the worker takes 1.
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


def _kill_and_reap(worker: TuningProcess) -> None:
    """SIGKILL and wait: kill() alone skips the signal once running is clear,
    so a child wedged in teardown would stay alive, holding the GPU."""
    process = worker.process
    if process is not None and process.poll() is None:
        process.kill()
    worker.kill()
    _reap(worker)


def _shutdown_locked(graceful: bool = True) -> None:
    """Assumes _worker_lock is held (_get_worker_locked calls it on key change).

    graceful=False kills outright, for a suspected GPU fault: once the
    reference is dropped there is no handle left to kill the child with.
    Either way it is signalled, waited on with a bound, and its pipes closed
    before the reference goes -- including when the sentinel write itself
    fails because the child already closed its read end.
    """
    global _worker, _worker_key
    if _worker is not None:
        try:
            if graceful:
                try:
                    _worker.shutdown(wait=False)
                    if _worker.alive():
                        _worker.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    log.debug("TunableOp worker will not exit; killing")
                    _kill_and_reap(_worker)
                else:
                    _worker.close()
            else:
                _kill_and_reap(_worker)
        except Exception:
            log.debug("TunableOp worker shutdown failed", exc_info=True)
        _worker = None
    _worker_key = None


def _insert_device_ordinal(filename: str, device: int) -> str:
    """Mirror of TuningContext::SetFilename(filename, insert_device_ordinal=true)."""
    if not filename:
        # SetFilename returns early on an empty name, which is how an empty
        # PYTORCH_TUNABLEOP_FILENAME turns persistence off. Appending an
        # ordinal would write results to a file called "0" instead.
        return ""
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
# instead -- which is not always benign, see _warn_if_env_blocks.
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


def _resolve_and_pin_filename_locked(device: int) -> str:
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
        if filename:
            tunable.set_filename(filename, insert_device_ordinal=False)
    return filename


def _job_device(job: BenchmarkJob) -> int:
    """The device the job's tensors live on, which is the one to tune on.

    Not current_device(): Inductor scopes V.graph.set_current_device, which is
    graph bookkeeping, and compile_fx's CUDA guard covers post-grad passes
    rather than lowering, so a graph targeting cuda:1 can be autotuned while
    the process current device is 0. The child builds its tensors from these
    metas, so this is the GPU it really tunes on, and the ordinal
    _resolve_and_pin_filename_locked names an unchosen results file for. Mirrors
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
    Results reach this process only through that file, which the parent's manager
    reads once and lazily, on its first TunableOp GEMM: anything that initialises
    the manager earlier misses rows the worker writes after that.
    Faults are reported rather than raised: a compile must continue even when a GEMM
    cannot be benchmarked.
    """
    if timeout is None:
        timeout = config.cuda.autotune_tunableop_timeout
    try:
        # Inside the try: this reports faults rather than raising, and reading
        # the device can fail just as the benchmark can.
        device = _job_device(job)
        settings = snapshot_settings()
    except Exception:  # resolve before the worker: no restart
        # Report without touching the worker: shutting down here would kill a
        # healthy one another thread is using, and write the None sentinel into
        # the pipe mid-stream once _worker_lock is released.
        _warn_fault_once("error", "could not resolve the device or settings")
        return Outcome(None, "error")
    try:
        with _worker_lock:
            # Under the lock: two threads must not pin different filenames.
            filename = _resolve_and_pin_filename_locked(device)
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
                    _warn_fault_once(status, "the worker was killed or died")
                    # get() killed or closed it but never waited; reap it.
                    _shutdown_locked(graceful=False)
                    return Outcome(None, status)
                _warn_fault_once("error", "the job raised")
                _shutdown_locked(graceful=False)
                return Outcome(None, "error")
            except Exception:  # a failed benchmark must not fail the compile
                _warn_fault_once("error", "the benchmark failed")
                # A caught exception can leave a sticky GPU fault behind, so
                # every later GEMM would fail alike. Kill rather than ask: a
                # child wedged in teardown never acts on the sentinel.
                _shutdown_locked(graceful=False)
                return Outcome(None, "error")
            except BaseException:  # noqa: B036
                # Ordinary failures are handled above, so this is
                # KeyboardInterrupt and friends. The child keeps running and
                # writes its result into the pipe unread, so reusing the worker
                # would hand that result to the next GEMM, and every one after.
                _shutdown_locked(graceful=False)
                raise
    except Exception:  # lock/restart machinery itself failed
        _warn_fault_once("error", "the worker machinery failed")
        return Outcome(None, "error")
