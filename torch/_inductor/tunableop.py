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
import copy
import logging
import math
import os
import threading
import time
from typing import Any, Literal, NamedTuple, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo

from . import config, ir
from .virtualized import V


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
# Reuse: TUNING=0 is harmless, it never tunes. The rest each break it --
# RECORD_UNTUNED skips ReadFile, and the other two drop the borrow or its record.
_REUSE_BLOCKING_ENV = (
    ("PYTORCH_TUNABLEOP_RECORD_UNTUNED", "1"),
    ("PYTORCH_TUNABLEOP_NEAREST_MATCH_ENABLED", "0"),
    ("PYTORCH_TUNABLEOP_RECORD_BORROWED_ENABLED", "0"),
)
_NO_FILE_ENV = (("PYTORCH_TUNABLEOP_FILENAME", ""),)
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
    # silence the other's, and a shared variable means something different in each.
    kind = f"{consequence}:{','.join(blocking)}"
    if blocking and kind not in _env_warned:
        _env_warned.add(kind)
        log.warning(
            "TunableOp %s: %s is set in the environment. Unset it.",
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


def is_tunableop_gemm(choice_name: str) -> bool:
    """Whether an extern choice is a GEMM TunableOp tunes.

    Single source of truth shared by the ladder (maybe_range_pretune) and the
    worker routing (ExternKernelCaller.benchmark): routing an op the ladder
    never tunes would benchmark it in the worker for no coverage, and in tune
    mode any GEMM an unrelated kernel calls internally would pollute the
    results file. Matched by choice name string so a double import of the
    kernel modules cannot break the match.
    """
    return choice_name in _TUNABLEOP_GEMM_CHOICES


# ExternKernelChoice names the ladder and worker routing cover. _int_mm /
# sparse semi-structured variants are not hipBLASLt-tunable.
_TUNABLEOP_GEMM_CHOICES = frozenset(
    OrderedSet(
        [
            "mm",
            "mm_dtype",
            "addmm",
            "addmm_dtype",
            "bmm",
            "bmm_dtype",
            "baddbmm",
            "bias_addmm",
            "_scaled_mm",
            "_scaled_mm_v2",
        ]
    )
)


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
        record_borrowed: bool = False,
        dyn_dims_mask: tuple[bool, bool, bool, bool] | None = None,
        measure: bool = True,
    ) -> None:
        self.bmreq = bmreq
        self.tuning = tuning
        self.nearest_match = nearest_match
        # (M, N, K, BATCH). Thread-local in the parent, so unlike the knobs in
        # settings it cannot be snapshotted -- the caller passes it explicitly
        # and __call__ re-enters it around the benchmark.
        self.dyn_dims_mask = dyn_dims_mask
        self.record_borrowed = record_borrowed
        # False for tune-only jobs (the ladder): the caller discards the time
        # and only needs the tuning side effect, so skip the second benchmark
        # and halve the worker cost per bucket.
        self.measure = measure

    def _warm_then_measure(self) -> float:
        """Warm, then measure.

        The first call is where TunableOp does its compile-time-only work, and
        folding that into the average reports it as the kernel's cost: on a
        3072x1894x242432 addmm it took aten from 3.5 ms to 7.45 ms and lost the
        autotune race to a 4.71 ms triton kernel, while the solution just recorded
        ran in 2.9 ms.
        """
        try:
            measured = self.bmreq.benchmark()
            warm_ok = True
        except Exception:  # a failed warm must still yield a timing
            log.warning(
                "TunableOp warm-up failed; measuring the untuned kernel instead",
                exc_info=True,
            )
            measured = None
            warm_ok = False
        if not self.measure:
            # Tune-only job (the ladder discards times): the warm already
            # performed the tuning side effect; skip the second benchmark.
            # A failed warm re-raises instead of reporting inf: the exception
            # crosses the pipe and submit() restarts the worker, since a warm
            # failure may leave a sticky fault (e.g. an illegal address)
            # behind. Reporting inf would skip that restart and leave the
            # poisoned worker serving every later bucket.
            if measured is None:
                raise RuntimeError("TunableOp tune-only warm-up failed")
            return measured
        # Nearest match is deliberately left on after a successful warm: if the
        # warm call recorded nothing, turning it off would measure the default
        # instead of the borrowed solution. After a FAILED warm, nothing was
        # cached, so turn it off -- otherwise the timed call runs the
        # validation inside the measurement, folding exactly the
        # compile-time-only cost this split exists to exclude.
        torch.cuda.tunable.tuning_enable(False)
        if not warm_ok:
            torch.cuda.tunable._nearest_match_enable(False)
            log.warning(
                "TunableOp warm-up failed and no borrow was validated; "
                "measuring the default kernel instead"
            )
        return self.bmreq.benchmark()

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
        # The write path is gated on it, so without this a validated borrow is
        # thrown away.
        tunable._record_borrowed_enable(self.record_borrowed)
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
                    return self._warm_then_measure()
            return self._warm_then_measure()
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
    """Mirror of TuningContext::SetFilename(filename, insert_device_ordinal=true).

    Copied, not delegated: the C++ takes the ordinal from current_device(), so
    letting it derive the name would mean switching the parent to the job's
    device here and creating a context on it inside the lock -- the work the
    worker exists to keep out of the parent.
    """
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
            if not filename:
                _warn_if_env_blocks(
                    _NO_FILE_ENV,
                    "has no results file, so nothing is recorded or reused",
                )
            stamped = _StampedJob(job, device, filename, settings)
            worker = _get_worker_locked((device, filename))
            try:
                worker.put(stamped)
                value = worker.get(timeout=timeout)
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
            except BaseException:
                # Ordinary failures are handled above, so this is
                # KeyboardInterrupt and friends. The child keeps running and its
                # result sits unread in the pipe: a reused worker would hand that
                # to the next GEMM, and every one after.
                _shutdown_locked(graceful=False)
                raise
    except Exception:  # lock/restart machinery itself failed
        _warn_fault_once("error", "the worker machinery failed")
        return Outcome(None, "error")
    # inf is what a benchmark returns when it declines to run the op, and
    # select_algorithm drops an inf candidate silently rather than reporting it.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        log.warning("TunableOp benchmark returned %r; treating as a fault", value)
        return Outcome(None, "error")
    return Outcome(float(value), "ok")


# Per-process record of laddered GEMMs: (kernel name, traced input/output meta
# signatures, worker scope). A transformer with N identical layers would
# otherwise run the whole ladder N times, and again on every recompile;
# TunableOp turns repeat signatures into lookups, but each submission still
# pickles a job, materializes full-size tensors in the worker and runs the
# benchmark loop. Cleared with the other inductor caches (fresh_cache).
class _PretunedGemms:
    def __init__(self) -> None:
        self.seen: OrderedSet[Any] = OrderedSet()

    def cache_clear(self) -> None:
        self.seen.clear()


try:
    from torch._inductor.utils import clear_on_fresh_cache

    _pretuned_gemms = clear_on_fresh_cache(_PretunedGemms())
except Exception:
    _pretuned_gemms = _PretunedGemms()

# ExternKernelChoice names (winner.choice.name) the ladder covers. Matched by
# string so a double import of the kernel modules cannot break the match.
# _int_mm / sparse semi-structured variants are not hipBLASLt-tunable.
# bias_addmm is excluded: it is only chosen for broadcast (stride-0) bias
# (kernel/mm.py), which the density guard below rejects -- and correctly so,
# since rebuilding its metas densely would tune a layout runtime never asks
# for.
_TUNABLEOP_GEMM_CHOICES = frozenset(
    OrderedSet(
        [
            "mm",
            "mm_dtype",
            "addmm",
            "addmm_dtype",
            "bmm",
            "bmm_dtype",
            "baddbmm",
            "_scaled_mm",
            "_scaled_mm_v2",
        ]
    )
)

# Smallest dim worth its own tuning job. Applied to the dim, not to the symbol:
# for an expression dim the smallest symbol values already yield substantial GEMMs.
_TUNABLEOP_MIN_DYN_DIM = 8

# Cap on a bucket's total footprint -- every operand plus the output, which the
# worker allocates at once. The dim ceiling bounds the dynamic dim, not the
# problem size: a bmm with dynamic batch at dim 16384 and K = N = 4096 fp16 is
# 512 GiB in one operand alone. Each fault can cost a cold worker restart.
_TUNABLEOP_MAX_LADDER_BYTES = 8 * 1024**3

# Ceiling on the SYMBOL, not the dim. A declared range reports what a symbol MAY
# take, not what it does -- every symbol here reports [2, 2**31-1] -- so without a
# cap the ladder runs orders of magnitude past the traced hint. Capping in symbol
# space also leaves an expression dim like 200*s free to reach the size it is
# actually served at, which a dim-space ceiling would truncate. Expression dims
# are bounded instead by the served-dim ceiling below.
_TUNABLEOP_MAX_DYN_SYM = 16384


def maybe_range_pretune(winner) -> None:
    """Tune a dynamic-shape GEMM at a range of sizes of its dynamic dim.

    TunableOp keys a solution on an exact shape signature, so tuning only the
    traced shape leaves runtime missing whenever the dynamic dim differs. Only
    runs in "tune" mode; see torch._inductor.tunableop.

    Coexists with the wildcard path (tunable_dyn_dims_mask): a wildcard entry
    matches every size but records no per-size optimum, so concrete ladder
    entries are still needed for exact hits with measured best kernels. Ladder
    entries are always recorded concrete -- the mask is deliberately not carried
    into the jobs.
    """
    import torch._inductor.select_algorithm as sa

    if not is_tune_mode():
        return
    if not config.cuda.autotune_tunableop_ladder:
        # No ladder entries, but the traced shape is still covered: autotune
        # routes this GEMM's aten candidate through the worker (see
        # ExternKernelCaller.benchmark), which tunes it at the traced shape.
        return
    if not isinstance(winner, sa.ExternKernelCaller):
        return
    # is_tune_mode() proves the host has a GPU, not that this GEMM runs on one.
    # A CPU winner carries an ExternKernelCPUBenchmarkRequest, which the CUDA
    # worker would spend the whole per-GEMM budget on and record nothing for.
    layout_device = getattr(winner.layout, "device", None)
    if layout_device is None or layout_device.type != "cuda":
        return
    # Match on the underlying ExternKernelChoice name. ExternKernelCaller
    # always sets self.choice (and its own name from it). Compared by string
    # so a double import of the kernel modules cannot break the match. Covers
    # the _dtype out-variants and _scaled_mm_v2; _int_mm and the sparse
    # variants are not hipBLASLt-tunable.
    choice_name = winner.choice.name
    if not is_tunableop_gemm(choice_name):
        return
    if getattr(winner, "bmreq", None) is None:
        return

    # Require exactly one free symbol across the operands (one dynamic dim).
    symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    for node in winner.input_nodes:
        for size in node.get_size():
            symbols |= OrderedSet(getattr(size, "free_symbols", ()))
    if len(symbols) != 1:
        return
    s = next(iter(symbols))

    # Never bucket the reduction dim K: a solution is only valid for the exact K
    # it was tuned at. This also rejects attention-style ops where one symbol
    # spans both N and K, where bucketing would change two dims at once.
    nodes = winner.input_nodes
    try:
        # Bias-first ops take [bias, mat1, mat2]; every other GEMM choice takes
        # mat1/mat2 first (scaled variants trail scale/bias tensors after).
        # Resolved from the choice, not the caller name, so _dtype / v2 / bias
        # variants route correctly.
        bias_first = choice_name in ("addmm", "addmm_dtype", "baddbmm")
        a, b = (nodes[1], nodes[2]) if bias_first else (nodes[0], nodes[1])
        a_sz = a.get_size()
        b_sz = b.get_size()
    except (IndexError, AttributeError):
        return
    if len(a_sz) < 2 or len(b_sz) < 2:
        return

    def _has(expr: Any) -> bool:
        return s in getattr(expr, "free_symbols", ())

    dims: OrderedSet[str] = OrderedSet()
    dyn_expr: Any = None  # size expression of the single dynamic dim
    if _has(a_sz[-2]):  # A is [..., M, K]
        dims.add("M")
        dyn_expr = a_sz[-2]
    if _has(b_sz[-1]):  # B is [..., K, N]
        dims.add("N")
        dyn_expr = b_sz[-1]
    if _has(a_sz[-1]) or _has(b_sz[-2]):  # contraction dim K
        dims.add("K")
    for sz in list(a_sz[:-2]) + list(b_sz[:-2]):  # leading batch dims (bmm)
        if _has(sz):
            dims.add("B")
            dyn_expr = sz
    # Exactly one axis. dyn_expr holds whichever axis was seen last, and the
    # ceiling below is computed from it alone, so a symbol spanning two axes
    # caps one and lets the other run away: with M = 1000*s and N = s, an
    # N-derived ceiling of 16384 permits M = 16,384,000 and an MxN output to
    # match. Such buckets also differ from each other in two dims, which is
    # precisely what the nearest matcher refuses to borrow across, so there is
    # nothing to gain by tuning them.
    if "K" in dims or len(dims) != 1 or dyn_expr is None:
        return

    # The ladder walks the SYMBOL, not the dim: for a dim like 200*s, stepping the
    # dim through powers of two would tune sizes it can never take. The domain is
    # the symbol's declared ShapeEnv range, which ignores guards and runtime
    # asserts and so can be looser than the true bound -- the safe direction,
    # since under-covering means a runtime miss.
    sizevars = V.graph.sizevars

    # Skip repeat GEMMs: identical layers in one model (and recompiles) share
    # traced metas, so the first ladder already recorded every bucket. Keyed on
    # the SYMBOLIC sizes and strides (not hints: mm([s,256],[256,512]) and
    # mm([512,256],[256,s]) hint identically at 512 but ladder different dims,
    # as do s and 200*s), the dtypes, and the (device, resolved filename) the
    # worker is keyed on -- a second compile targeting another device or file
    # must not skip. Strides included: mm(A,B) and mm(A,B_colmajor) are both
    # dense with different TunableOp signatures (NN vs NT).
    def _node_sig(n: Any) -> Any:
        try:
            dtype = n.get_dtype() if hasattr(n, "get_dtype") else n.dtype
            size = n.get_size() if hasattr(n, "get_size") else n.size
            stride = n.get_stride() if hasattr(n, "get_stride") else None
            if stride is None:
                try:
                    from torch._inductor.select_algorithm import (
                        get_strides_with_layout_constraints,
                    )

                    stride = sizevars.optimization_hints(
                        get_strides_with_layout_constraints(n)
                    )
                except Exception:
                    stride = ()
            return (
                str(dtype),
                tuple(str(x) for x in size),
                tuple(str(x) for x in stride),
            )
        except Exception:
            return None

    try:
        sigs = [_node_sig(n) for n in winner.input_nodes] + [_node_sig(winner.layout)]
    except Exception:
        sigs = [None]

    def _declared(expr: Any) -> tuple[int | None, int | None]:
        """(lower, upper) from ShapeEnv, each None when unbounded or symbolic."""
        try:
            vr = sizevars.shape_env.bound_sympy(expr)
        except Exception:  # no range for this expression
            return None, None
        lo = hi = None
        try:
            if vr.lower != -int_oo and vr.lower != -sympy.oo:
                lo = int(vr.lower)
        except (TypeError, ValueError):  # symbolic bound
            pass
        try:
            if vr.upper != int_oo and vr.upper != sympy.oo:
                hi = int(vr.upper)
        except (TypeError, ValueError):  # symbolic bound
            pass
        return lo, hi

    try:
        sym_hint = sizevars.optimization_hint(s)
    except Exception:
        return
    sym_lo, sym_hi = _declared(s)
    # A declared bound narrower than the cap wins. The cap is raised to the
    # traced hint when that is larger, so the walk can always reach the size
    # actually served.
    sym_ceiling = max(_TUNABLEOP_MAX_DYN_SYM, sym_hint)
    sym_hi = sym_ceiling if sym_hi is None else min(sym_hi, sym_ceiling)
    if sym_hi < 2:
        return

    def _memo_key() -> Any:
        """Identifies a ladder already recorded in this process.

        Rebuilt at the check and again at the record: submit() resolves and
        pins the results file, so its name is only known after one succeeds.
        The bounds are part of the key because a larger hint or a narrower
        declared range produces a different ladder for the same operands.
        """
        if not all(x is not None for x in sigs):
            return None
        scope = (
            (layout_device.type, layout_device.index),
            torch.cuda.tunable.get_filename(),
        )
        return (winner.name, tuple(sigs), scope, sym_hint, sym_lo, sym_hi)

    if (key := _memo_key()) is not None and key in _pretuned_gemms.seen:
        return

    def _eval_at(expr: Any, sym_val: int) -> int:
        """Value of a size expression with the dynamic symbol set to sym_val."""
        if isinstance(expr, sympy.Expr) and expr.free_symbols:
            return int(sizevars.optimization_hint(expr.subs(s, sym_val)))
        return int(expr)

    # The "too small to tune" threshold applies in DIM space, not symbol space:
    # for a dim like 200*s, s=2 is already a dim of 400, so starting the walk at
    # 8 would skip sizes runtime genuinely asks for.
    sym_start = max(2, sym_lo if sym_lo is not None else 2)
    # The traced hint is always a rung: it is the size actually served, so it
    # must be tuned even when it is not a power of two and neither endpoint.
    sym_vals = sorted(
        OrderedSet([sym_start, sym_hint, sym_hi])
        | OrderedSet([1 << k for k in range(1, 64) if sym_start < (1 << k) < sym_hi])
    )
    # Ceiling in DIM space, since the symbol cap cannot see a multiplier: with
    # the symbol alone bounded at 16384, 96*s reached 1572864. Anchored to the
    # served dim so it can never truncate below what the model actually runs --
    # a flat 16384 would have cut a GEMM served at 294912 down to 12288.
    try:
        served_dim = _eval_at(dyn_expr, sym_hint)
    except Exception:  # unevaluatable at the hint
        served_dim = 0
    dim_ceiling = max(_TUNABLEOP_MAX_DYN_SYM, served_dim)

    buckets: list[int] = []
    dim_vals: list[int] = []
    for sym_val in sym_vals:
        try:
            dim_val = _eval_at(dyn_expr, sym_val)
        except Exception:  # unevaluatable at this value
            continue
        if not (_TUNABLEOP_MIN_DYN_DIM <= dim_val <= dim_ceiling):
            continue
        if dim_val in dim_vals:
            continue
        buckets.append(sym_val)
        dim_vals.append(dim_val)
    # The traced hint first, then largest first. The budget below is shared and
    # spent in order: ascending order dropped the sizes nearest the shape
    # actually served, and pure largest-first can spend it all on the biggest
    # sizes -- the likeliest to fault -- before reaching the one size runtime is
    # known to ask for.
    order = sorted(
        range(len(buckets)), key=lambda i: (buckets[i] != sym_hint, -dim_vals[i])
    )
    buckets = [buckets[i] for i in order]
    dim_vals = [dim_vals[i] for i in order]
    log.info(
        "TunableOp range pre-tune: %s dim=%s expr=%s sym=%s sym_range=[%s, %s] "
        "sym_hint=%s buckets=%d dims=%s",
        winner.name,
        "/".join(dims),
        dyn_expr,
        s,
        sym_lo,
        sym_hi,
        sym_hint,
        len(buckets),
        dim_vals[:8],
    )
    if not buckets:
        return

    # Copy the winner's serializable bmreq and rebuild its tensor metas per
    # bucket, reusing Inductor's meta machinery rather than rebuilding the GEMM.
    from torch._inductor.autotune_process import TensorMeta
    from torch._inductor.select_algorithm import get_strides_with_layout_constraints

    def _as_buffer(node: Any) -> Any:
        return (
            ir.Buffer(name="fake", layout=node) if isinstance(node, ir.Layout) else node
        )

    def _is_densely_packed(node: Any) -> bool:
        """Traced strides are the dense strides for the traced sizes.

        _meta_at recomputes strides densely, so a padded or broadcast operand
        would be tuned at the wrong leading dimension -- [s, 256] strided
        [300, 1] becomes [256, 1]. TunableOp keys on ld, so such an entry can
        never match the runtime call, and the nearest matcher will not borrow
        across a padded ld either. Tuning it is pure waste; skip instead.
        """
        n = _as_buffer(node)
        real_strides = [
            int(x)
            for x in sizevars.optimization_hints(get_strides_with_layout_constraints(n))
        ]
        real_sizes = [int(x) for x in sizevars.optimization_hints(n.get_size())]
        order = ir.get_stride_order(real_strides)
        dense = [int(x) for x in ir.FlexibleLayout.stride_ordered(real_sizes, order)]
        return real_strides == dense

    try:
        operands = [*winner.input_nodes, winner.layout]
        if not all(_is_densely_packed(n) for n in operands):
            log.info(
                "TunableOp range pre-tune: skipping %s, an operand is not "
                "densely packed and would be tuned at the wrong ld",
                winner.name,
            )
            return
    except Exception as e:  # cannot prove density, do not guess
        log.info(
            "TunableOp range pre-tune: skipping %s, layout check failed: %s",
            winner.name,
            e,
        )
        return

    def _meta_at(node: Any, sym_val: int) -> TensorMeta:
        n = _as_buffer(node)
        # Substitute into the SIZE expressions only, then recompute dense strides:
        # a stride expression is not a size-product formula in the symbol, and a
        # bare-symbol batch stride would collapse to the bucket value instead of
        # value * inner_dims. The stride ORDER must come from the traced real-size
        # strides, since small-bucket strides can reorder dims and silently turn a
        # plain GEMM into a transposed one.
        sizes = [_eval_at(sz, sym_val) for sz in n.get_size()]
        order = ir.get_stride_order(
            sizevars.optimization_hints(get_strides_with_layout_constraints(n))
        )
        strides = [int(x) for x in ir.FlexibleLayout.stride_ordered(sizes, order)]
        return TensorMeta(
            device=n.get_device(),
            dtype=n.get_dtype(),
            sizes=sizes,
            strides=strides,
            offset=0,
            name=n.get_name(),
        )

    bmreqs: list[Any] = []
    try:
        for v in buckets:
            sized_bmreq: Any = copy.copy(winner.bmreq)
            sized_bmreq.input_tensor_meta = [_meta_at(n, v) for n in winner.input_nodes]
            sized_bmreq.output_tensor_meta = _meta_at(winner.layout, v)
            # TunableOp cannot tune under CUDA-graph capture. copy.copy keeps
            # both flags: benchmark_with_cudagraphs (set per-choice) and
            # config_cudagraph_benchmarking (snapshotted True on ROCm under
            # max-autotune), and benchmark_run_fn captures if either is set --
            # every bucket would then silently record nothing.
            sized_bmreq.benchmark_with_cudagraphs = False
            sized_bmreq.config_cudagraph_benchmarking = False
            bmreqs.append(sized_bmreq)
    except Exception as e:  # bmreq build must not fail the compile
        log.warning(
            "TunableOp range pre-tune: failed to build bmreqs for %s: %s",
            winner.name,
            e,
        )
        return

    if not bmreqs:
        return

    # Drop buckets no GPU could materialize. The dim ceiling bounds the dynamic
    # dim, not the problem size: a bmm with dynamic batch at dim 16384 and
    # K = N = 4096 fp16 is 512 GiB per operand, and the biggest sizes run early
    # -- each fault can cost a cold worker restart.
    def _meta_bytes(m: Any) -> float:
        # inf, not 0: a size we cannot compute must fail the cap, not pass it.
        try:
            numel = 1
            for d in m.sizes:
                numel *= int(d)
            return numel * m.dtype.itemsize
        except Exception:
            return float("inf")

    kept: list[Any] = []
    kept_dims: list[int] = []
    for bmreq, dim_val in zip(bmreqs, dim_vals):
        try:
            metas = [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]
            # Summed: the worker allocates every operand and the output at once.
            footprint = sum(_meta_bytes(m) for m in metas)
        except Exception:
            footprint = float("inf")
        if footprint > _TUNABLEOP_MAX_LADDER_BYTES:
            log.warning(
                "TunableOp range pre-tune for %s skipping dim %d: operands "
                "and output total %.1f GiB, over the %.1f GiB cap.",
                winner.name,
                dim_val,
                footprint / 1024**3,
                _TUNABLEOP_MAX_LADDER_BYTES / 1024**3,
            )
            continue
        kept.append(bmreq)
        kept_dims.append(dim_val)
    bmreqs, dim_vals = kept, kept_dims
    if not bmreqs:
        return

    # One job per bucket so a fault costs a single size, but one budget for the
    # whole GEMM so compile time does not scale with the bucket count.
    recorded = False
    deadline = time.monotonic() + config.cuda.autotune_tunableop_timeout
    for i, bmreq in enumerate(bmreqs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning(
                "TunableOp range pre-tune for %s exhausted its %.0fs budget after "
                "%d/%d sizes; skipping the rest.",
                winner.name,
                config.cuda.autotune_tunableop_timeout,
                i,
                len(bmreqs),
            )
            break
        bucket_start = time.monotonic()
        # The whole remaining budget, deliberately. TuningProcess.get() kills
        # the worker on timeout, so a per-bucket slot below the 15-40s
        # cold-start cost makes the restart cold, time out and be killed again.
        # Letting the first bucket have the budget absorbs the cold start
        # instead of fighting it.
        outcome = submit(
            BenchmarkJob(bmreq, tuning=True, measure=False), timeout=remaining
        )
        # The only record of what a bucket costs when it SUCCEEDS: the other
        # logs here report failures only, and that curve is what a per-bucket
        # budget would have to be sized against.
        log.debug(
            "TunableOp pre-tune bucket: op=%s dim=%s size=%d/%d wall=%.2fs status=%s",
            winner.name,
            dim_vals[i],
            i + 1,
            len(bmreqs),
            time.monotonic() - bucket_start,
            outcome.status,
        )
        if outcome.status == "ok":
            recorded = True
        else:
            # Solutions found before the fault are already flushed to the
            # results file, and the worker restarts on the next submission.
            log.warning(
                "TunableOp range pre-tune for %s hit %s on size %d/%d; continuing.",
                winner.name,
                outcome.status,
                i + 1,
                len(bmreqs),
            )

    if recorded and (key := _memo_key()) is not None:
        _pretuned_gemms.seen.add(key)
