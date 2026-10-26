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
import copy
import logging
import os
import threading
import time
from typing import Any, Literal, NamedTuple

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo

from . import config, ir
from .virtualized import V


log = logging.getLogger(__name__)


# Outcome of a worker submission: "ok", "timeout" (worker hung and was killed),
# "crash" (worker died, typically a GPU fault) or "error" (the job raised).
WorkerStatus = Literal["ok", "timeout", "crash", "error"]

_valid_modes = ("off", "tune", "reuse")
_mode_warned = False


def _mode() -> str:
    """The configured mode, warning once on values that silently mean "off".

    config.aten_tunableop_mode accepts any string from TORCHINDUCTOR_ATEN_TUNABLEOP_MODE,
    so a typo like "Tune" or "on" would otherwise disable both passes silently.
    """
    global _mode_warned
    mode = config.aten_tunableop_mode
    if mode not in _valid_modes and not _mode_warned:
        _mode_warned = True
        log.warning(
            "Unknown TORCHINDUCTOR_ATEN_TUNABLEOP_MODE=%r (expected one of %s); "
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
        # Stamped by submit() with the parent's tuning knobs (see
        # snapshot_settings): the worker is a long-lived fresh process that
        # would otherwise tune with defaults.
        self.settings: dict[str, Any] = {}
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
        # Re-enter the parent's tuning knobs (snapshot_settings): process-global
        # C++ state, not env vars, so the worker would otherwise silently use
        # defaults. Numerical-check tolerances have no getter and are not
        # propagated -- documented in snapshot_settings.
        for name, value in self.settings.items():
            getattr(tunable, name)(value)
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
# How long a graceful shutdown waits for the child before resorting to SIGKILL.
_SHUTDOWN_TIMEOUT_S = 10.0
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
    worker reused across GEMMs rather than one per GEMM, since startup dominates
    otherwise.

    Lifetime: module-global, torn down on a key change or at interpreter exit --
    not when a compile finishes, because nothing marks that boundary yet. So an
    idle child keeps its CUDA/HIP context, and whatever the caching allocator
    held for benchmarking, for the rest of the process. Callers that know where
    autotuning ends should call shutdown_worker() there.

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
        _shutdown_locked()
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
    """Shut down the worker. Callers holding _worker_lock use _shutdown_locked."""
    with _worker_lock:
        _shutdown_locked()


def _shutdown_locked(graceful: bool = True) -> None:
    """Assumes _worker_lock is held (get_worker calls it on key change).

    graceful=False skips the exit sentinel and kills the child outright. Use it
    when a GPU fault is suspected: the child may be stuck in device teardown, and
    once the reference is dropped there is no handle left to kill it with.

    Either way the process is reaped before the reference goes. shutdown(wait=False)
    on its own leaves the pipes, the selector fd and the Popen for GC, which is why
    TuningProcessPool.shutdown pairs it with wait().
    """
    global _worker, _worker_key
    if _worker is not None:
        try:
            if graceful:
                _worker.shutdown(wait=False)
                # Bounded: TuningProcess.wait() blocks on process.wait() with no
                # timeout, and this runs under _worker_lock on a key change and
                # again from atexit. A child wedged in device teardown would
                # otherwise stall every later submit(), or stop the compiling
                # process from exiting at all.
                try:
                    if _worker.alive():
                        _worker.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    log.debug(
                        "TunableOp worker did not exit in %ss; killing",
                        _SHUTDOWN_TIMEOUT_S,
                    )
                    _worker.kill()
                else:
                    _worker.close()
            else:
                _worker.kill()
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


# Tuning knobs propagated to the worker: (parent getter, worker setter) pairs.
#
# Not propagated, and not an exhaustive-by-accident list:
#   * numerical-check tolerances -- set_numerical_check_tolerances has no
#     getter, so there is nothing to snapshot;
#   * wildcard_fallback_enable, record_untuned_enable and
#     _record_borrowed_enable -- these do have getters, but they select what
#     the worker records rather than how it tunes, and the worker's recording
#     policy is set per job rather than inherited from the parent;
#   * the thread-local dynamic-dims mask, which is per call rather than per
#     process and so travels on the job itself, not in this snapshot.
# Anything read from the environment reaches the worker by inheritance.
_TUNABLE_SETTINGS: tuple[tuple[str, str], ...] = (
    ("get_max_tuning_duration", "set_max_tuning_duration"),
    ("get_max_tuning_iterations", "set_max_tuning_iterations"),
    ("get_rotating_buffer_size", "set_rotating_buffer_size"),
    ("get_cublaslt_requested_algo_count", "set_cublaslt_requested_algo_count"),
)

# Getters whose units differ from the matching setter's.
# get_rotating_buffer_size returns bytes (rotating_buffer_size_, or the L2 size
# when unset), while set_rotating_buffer_size takes MB and multiplies by
# 1024*1024 into a signed int. Stamping the raw value would square the scale:
# a 4 MiB L2 becomes 2**42, which truncates to 0 and silently tunes with the
# rotating buffer disabled. Only visible when the env var is unset, since that
# takes precedence inside the getter.
_SETTING_CONVERSIONS: dict[str, Any] = {
    "set_rotating_buffer_size": lambda v: v // (1024 * 1024),
}


def snapshot_settings() -> dict[str, Any]:
    """Read the parent's tuning knobs for stamping onto the job.

    The worker is long-lived and keyed only on (device, filename), so it never
    sees settings changed after it starts; stamping per job keeps it in sync.
    Best-effort per knob: a getter failure leaves that knob at the worker's
    default rather than failing the submission.
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
        # Stamp the parent's tuning knobs for the same reason: process-global
        # C++ state does not cross the pipe.
        job.settings = snapshot_settings()
    except Exception as e:  # resolve before the worker: no restart
        # Failed before the worker was involved (no device, no filename): report
        # without touching it. Shutting down here would kill a healthy worker
        # another thread is using, and -- worse -- write the None sentinel into
        # the pipe mid-stream after _worker_lock is released.
        log.debug("TunableOp benchmark setup failed: %s", e)
        return Outcome(None, "error")
    try:
        with _worker_lock:
            worker = get_worker((device, job.filename))
            try:
                worker.put(job)
                return Outcome(worker.get(timeout=timeout), "ok")
            except TimeoutError:
                return Outcome(None, "timeout")
            except EOFError:
                return Outcome(None, "crash")
            except Exception as e:  # a failed benchmark must not fail the compile
                log.debug("TunableOp benchmark failed: %s", e)
                # A catchable exception can leave a sticky fault behind (e.g. an
                # illegal address corrupting the GPU context): the worker stays
                # alive, so every later GEMM would fail the same way. Restart it
                # here, inside the lock, so no other thread observes the dead
                # worker or interleaves on the pipe; recorded solutions survive
                # via the results file. Kill rather than ask: a child wedged in
                # device teardown would never act on the sentinel, and would
                # hold the GPU for the rest of the job.
                _shutdown_locked(graceful=False)
                return Outcome(None, "error")
    except Exception as e:  # lock/restart machinery itself failed
        log.debug("TunableOp worker machinery failed: %s", e)
        return Outcome(None, "error")


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

# Largest single operand the ladder will materialize, in bytes. The dim ceiling
# bounds the dynamic dim, not the problem size: a bmm with dynamic batch at dim
# 16384 and K = N = 4096 fp16 is 512 GiB per operand. Largest-first order would
# submit those first, and each fault can cost a cold worker restart.
_TUNABLEOP_MAX_LADDER_BYTES = 8 * 1024**3

# Ceiling on the SYMBOL, not the dim. A declared range reports what a symbol MAY
# take, not what it does: every symbol here reports [2, 2**31-1], so the ladder ran
# to the old dim ceiling of 65536 for every GEMM -- 64x the traced hint on one model
# and 4096x on another. Capping in symbol space also leaves an expression dim like
# 200*s free to reach the size it is actually served at, which a dim-space ceiling
# truncated: one GEMM served a dim of 294912 while its ladder stopped at 49152.
# Expression dims are bounded instead by the served-dim ceiling below.
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
    if not config.aten_tunableop_ladder:
        # Traced shape only; the normal autotune path still tunes each aten
        # candidate at the shape it was traced at.
        return
    if not isinstance(winner, sa.ExternKernelCaller):
        return
    # Match on the underlying ExternKernelChoice name. ExternKernelCaller
    # always sets self.choice (and its own name from it). Compared by string
    # so a double import of the kernel modules cannot break the match. Covers
    # the _dtype out-variants and _scaled_mm_v2, which the old
    # op-name allowlist missed; _int_mm / sparse variants are not
    # hipBLASLt-tunable.
    choice_name = winner.choice.name
    if choice_name not in _TUNABLEOP_GEMM_CHOICES:
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
        try:
            scope = (torch.cuda.current_device(), resolve_filename(0))
        except Exception:
            scope = (None, "")
        memo_key: Any = (
            (winner.name, tuple(sigs), scope)
            if all(x is not None for x in sigs)
            else None
        )
    except Exception:
        memo_key = None
    if memo_key is not None:
        if memo_key in _pretuned_gemms.seen:
            return
        _pretuned_gemms.seen.add(memo_key)

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
    # Largest first: the budget below is shared across the ladder and spent in
    # order, so ascending order dropped the sizes nearest the shape actually
    # served while keeping 8, 16, 32.
    order = sorted(range(len(buckets)), key=lambda i: -dim_vals[i])
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
    # K = N = 4096 fp16 is 512 GiB per operand, and largest-first order submits
    # those first -- each fault can cost a cold worker restart.
    def _meta_bytes(m: Any) -> int:
        try:
            numel = 1
            for d in m.sizes:
                numel *= int(d)
            return numel * m.dtype.itemsize
        except Exception:
            return 0

    kept: list[Any] = []
    kept_dims: list[int] = []
    for bmreq, dim_val in zip(bmreqs, dim_vals):
        try:
            metas = [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]
            biggest = max(_meta_bytes(m) for m in metas)
        except Exception:
            biggest = 0
        if biggest > _TUNABLEOP_MAX_LADDER_BYTES:
            log.warning(
                "TunableOp range pre-tune for %s skipping dim %d: largest "
                "operand is %.1f GiB, over the %.1f GiB cap.",
                winner.name,
                dim_val,
                biggest / 1024**3,
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
    deadline = time.monotonic() + config.aten_tunableop_timeout
    for i, bmreq in enumerate(bmreqs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning(
                "TunableOp range pre-tune for %s exhausted its %.0fs budget after "
                "%d/%d sizes; skipping the rest.",
                winner.name,
                config.aten_tunableop_timeout,
                i,
                len(bmreqs),
            )
            break
        bucket_start = time.monotonic()
        # The whole remaining budget, deliberately. A per-bucket slot was tried
        # and is worse: TuningProcess.get() kills the worker on timeout, and a
        # slot below the cold-start cost of 15-40s makes the restart cold, time
        # out, and be killed again -- a flat 5s produced 334 timeouts and 334
        # kills where the shared budget produced 7. Letting the first bucket
        # have the budget absorbs the cold start instead of fighting it.
        outcome = submit(BenchmarkJob(bmreq, tuning=True), timeout=remaining)
        if config.aten_tunableop_pretune_debug:
            # Per-bucket cost, which nothing else records: only failures are
            # logged today, so the cost of the buckets that SUCCEED -- and
            # whether it falls as the worker warms up -- is invisible. That
            # curve is what a per-bucket budget has to be sized against. Info,
            # not debug: the flag alone must suffice, without also raising the
            # logger level.
            log.info(
                "TunableOp pre-tune bucket: op=%s dim=%s size=%d/%d "
                "wall=%.2fs status=%s",
                winner.name,
                dim_vals[i],
                i + 1,
                len(bmreqs),
                time.monotonic() - bucket_start,
                outcome.status,
            )
        if outcome.status != "ok":
            # Solutions found before the fault are already flushed to the
            # results file, and the worker restarts on the next submission.
            log.warning(
                "TunableOp range pre-tune for %s hit %s on size %d/%d; continuing.",
                winner.name,
                outcome.status,
                i + 1,
                len(bmreqs),
            )
