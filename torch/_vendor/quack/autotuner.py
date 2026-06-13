# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import os
import subprocess
import sys
import threading
import time
import inspect
import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property, partial
from typing import Dict, Tuple, List, Optional, Any
from .bench.bench_utils import (
    _bench_cuda_graph_l2_rotate,
    _clone_l2_rotate_inputs,
    _pick_l2_rotate_count,
)

import torch
from torch import Tensor

import triton

from . import __version__


PACKAGE_NAME = "torch_vendor_quack"
VERSION = __version__


_TENSOR_META_TAG = "__quack_tensor_meta__"


def _serialize_precompile_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            _TENSOR_META_TAG: True,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return tuple(_serialize_precompile_value(v) for v in value)
    if isinstance(value, list):
        return [_serialize_precompile_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_precompile_value(v) for k, v in value.items()}
    return value


def _get_current_cuda_device() -> str | None:
    """Return the physical CUDA device identifier for the current process.

    Maps the logical ``torch.cuda.current_device()`` index through
    ``CUDA_VISIBLE_DEVICES`` (if set) so the result is valid as a
    standalone ``CUDA_VISIBLE_DEVICES`` value (handles integer IDs,
    GPU UUIDs, and MIG IDs).

    Returns ``None`` if CUDA is not initialized or the device cannot
    be determined.
    """
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    logical_device = torch.cuda.current_device()
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible is not None:
        visible_devices = [d.strip() for d in parent_visible.split(",")]
        if logical_device < len(visible_devices):
            return visible_devices[logical_device]
        return None
    return str(logical_device)


def get_home_dir():
    return os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home())


def default_cache_dir():
    return os.path.join(get_home_dir(), f".{PACKAGE_NAME}", "cache")


class FileCacheManager(triton.runtime.cache.FileCacheManager):
    def __init__(self, key):
        super().__init__(key)
        self.cache_dir = (
            os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_DIR", "").strip() or default_cache_dir()
        )
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


def _gpu_warmup(duration_ms=200):
    """Saturate the GPU to reach thermal steady-state before benchmarking.

    Without this, the first autotuning config gets artificially good numbers
    because the GPU hasn't been power-throttled yet.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(100):
            a = a @ a
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Async precompile handle (gaps 2 + 3 from the autotune-pool hardening pass)
#
# The legacy ``_precompile`` was a blocking barrier: spawn N workers, dispatch
# all configs round-robin, collect-all-results, then return. Two consequences:
#
#   * If a worker died mid-compile (e.g. a config that trips an internal
#     cute.compile assert), the configs it had not yet acknowledged were
#     silently dropped. The autotuner then ran the eager body for those
#     configs without a warm .o, paying ~500 ms of compile inside the bench
#     measurement; the artificially-slow timing made that config never get
#     picked even when it was the actual best one.
#   * Total wall = parallel_compile + serial_bench. With ~10 configs at
#     ~500 ms compile and ~100 ms bench each, that's (5s/pool) + 1s. With
#     async overlap it drops to max(5s/pool, 1s) — a ~40 % win on cold
#     autotune.
#
# :class:`_PrecompileHandle` returns immediately after spawning workers and
# dispatching their initial round of tasks. Per-config :class:`threading.Event`
# completion signals let the bench loop wait on the specific config it's
# about to benchmark. A background reader thread per worker drains stdout
# replies and sets the corresponding events; on worker crash (None from
# ``_recv`` = EOF) every still-pending task assigned to that worker is
# marked failed (and the event set) so callers don't deadlock.
# ---------------------------------------------------------------------------


@dataclass
class _PrecompileHandle:
    """Per-config completion handle returned by :meth:`Autotuner._precompile`.

    Empty / no-op when the subprocess pool was skipped (cache hit, disabled,
    only 1 config). Calling ``wait_for(i)`` on a missing index returns
    immediately as a no-op, so the bench loop can iterate uniformly.
    """

    events: Dict[int, threading.Event] = field(default_factory=dict)
    failures: Dict[int, str] = field(default_factory=dict)
    _workers: List[Any] = field(default_factory=list)  # subprocess.Popen
    _reader_threads: List[threading.Thread] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def wait_for(self, config_idx: int, timeout: Optional[float] = None) -> None:
        """Block until ``config_idx``'s compile completes (or failed).

        No-op for indices that were never registered (e.g. the entire handle
        is empty because precompile was skipped). The bench loop relies on
        this to call ``wait_for`` unconditionally.
        """
        evt = self.events.get(config_idx)
        if evt is not None:
            evt.wait(timeout=timeout)

    def is_failed(self, config_idx: int) -> bool:
        """True if the subprocess pool failed to compile this config.

        Use after :meth:`wait_for` returns. Failed configs need an in-process
        jit_cache warm before benchmarking to avoid contaminating the timing
        measurement with compile cost (see :meth:`Autotuner._bench_with_warm`).
        """
        with self._lock:
            return config_idx in self.failures

    def shutdown(self) -> None:
        """Close stdins, wait for workers and reader threads to exit.

        Idempotent. Safe to call even if ``__init__`` failed midway through
        spawning workers.
        """
        for w in self._workers:
            try:
                if w.poll() is None and w.stdin is not None:
                    w.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        for w in self._workers:
            try:
                w.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                w.kill()
                try:
                    w.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass  # process really stuck; give up rather than hang the bench
        for t in self._reader_threads:
            t.join(timeout=1.0)


def _reader_thread_main(handle, worker, config_indices):
    """Drain replies from one worker's stdout and signal per-config events.

    Replies arrive in dispatch order (length-prefixed pickled strings:
    ``"OK"`` or ``"ERR:..."``). On EOF (= worker crashed), every config
    this worker still had pending is marked failed and its event set so
    waiters don't deadlock.

    Runs in its own daemon thread so the bench loop can interleave waiting
    with kernel launches.
    """
    for pos, config_idx in enumerate(config_indices):
        try:
            r = _recv_from_worker(worker.stdout)
        except (BrokenPipeError, EOFError, ConnectionResetError):
            r = None
        with handle._lock:
            if r is None:
                # Worker died; mark this config + all remaining ones failed.
                # ``setdefault`` so we don't clobber a real ERR: reply that
                # somehow raced (unlikely, but cheap to be safe).
                for remaining_idx in config_indices[pos:]:
                    handle.failures.setdefault(remaining_idx, "worker crashed during compile")
                    handle.events[remaining_idx].set()
                return
            if isinstance(r, str) and r.startswith("ERR:"):
                handle.failures[config_idx] = r
            handle.events[config_idx].set()


def _recv_from_worker(stream):
    """Read a length-prefixed pickled message. Returns None on EOF or partial body.

    Mirrors ``_compile_worker._recv`` so the parent and worker speak the
    same protocol. Kept private to this module rather than re-imported from
    the worker because the worker module isn't safe to import in the parent
    (it pushes ``_COMPILE_ONLY_DEPTH`` at module load).

    Returns ``None`` on both:

    * clean EOF (header shorter than 4 bytes — pipe write side closed before
      writing the header);
    * truncated body (header read OK, but the body read returns fewer bytes
      than the declared length — the worker was SIGKILL'd by its parent-death
      watchdog, or by OOM-killer, *between* writing the header and writing
      the body).

    Without the truncated-body check, ``pickle.loads`` raises ``UnpicklingError``
    on a short body. That exception would propagate out of the reader thread,
    leaving the per-config events unset and deadlocking every ``wait_for(i)``
    waiter in :meth:`Autotuner.__call__`'s ``benchmark()`` closure.
    """
    import pickle
    import struct

    header = stream.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack("<I", header)[0]
    if length == 0:
        return None
    body = stream.read(length)
    if len(body) < length:
        # Pipe closed mid-message. Treat as EOF; the reader thread's caller
        # ``_reader_thread_main`` will mark the rest of this worker's queue
        # as failed and set the events so waiters wake up.
        return None
    return pickle.loads(body)


class Autotuner:
    def __init__(
        self,
        fn,
        key,
        configs,
        restore_value=None,
        prune_configs_by: Optional[Dict] = None,
        do_bench=None,
        cache_results=False,
        precompile_configs=True,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [AutotuneConfig()]
        else:
            self.configs = configs
        signature = inspect.signature(fn)
        self.keys = key
        self.cache: Dict[Tuple, AutotuneConfig] = {}
        self.arg_names = list(signature.parameters.keys())
        self.cache_results = (
            cache_results or os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_AUTOTUNING", None) == "1"
        )
        self.precompile_configs = precompile_configs

        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        if len(self.restore_value) > 0:

            def _pre_hook(kwargs):
                self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook
        else:
            self.pre_hook = None

        if len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook
        else:
            self.post_hook = None

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get(
                "early_config_prune", self.early_config_prune
            )

        self.fn = fn
        self._do_bench = do_bench

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return partial(triton.testing.do_bench, warmup=5, rep=25)
        return self._do_bench

    def _precompile(self, *args, configs, **kwargs) -> _PrecompileHandle:
        """Pre-compile all configs in parallel subprocesses to populate .o cache.

        cute.compile() is not thread-safe (MLIR thread-local state) and fork after
        CUDA init causes segfaults. So we spawn persistent subprocess workers: each
        has its own CUDA context, creates FakeTensors matching the parent's tensor
        metadata, and compiles with COMPILE_ONLY=True. Workers stay alive to amortize
        import overhead across multiple configs. The parent then loads instantly from
        the .o cache during benchmarking.

        Returns a :class:`_PrecompileHandle` whose ``wait_for(i)`` blocks until
        config ``i``'s compile completes; the bench loop interleaves
        ``wait_for`` + benchmark to overlap the remaining compiles with GPU
        work. Workers are equipped with a parent-death watchdog
        (:func:`quack._compile_worker._install_parent_watchdog`) so an
        orphaned worker self-terminates within ~60 s instead of lingering on
        long-lived self-hosted CI runners.

        The returned handle is also valid (empty but functional) when
        precompilation is skipped — cache disabled, only 1 config, the
        first-config warm-cache shortcut hit. Callers always call
        ``wait_for`` + ``shutdown``; the no-op handle makes both fast.
        """
        from .cache import CACHE_ENABLED

        if not self.precompile_configs or not CACHE_ENABLED:
            return _PrecompileHandle()

        max_workers = min(len(configs), int(os.getenv("QUACK_COMPILE_WORKERS", "8")))
        if max_workers <= 1:
            return _PrecompileHandle()

        # Quick check: compile first config in-process. If it loads from .o cache
        # (<0.5s), the rest are likely cached too — skip spawning workers.
        t_check = time.time()
        try:
            current = dict(kwargs, **configs[0].all_kwargs())
            self.fn(*args, **current)
        except Exception:
            pass
        if time.time() - t_check < 0.5:
            return _PrecompileHandle()

        verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        if verbose:
            print(f"Pre-compiling {len(configs)} configs with {max_workers} workers")

        import pickle
        import struct

        def _send(stream, msg):
            data = pickle.dumps(msg)
            stream.write(struct.pack("<I", len(data)))
            stream.write(data)
            stream.flush()

        serialized_args = [_serialize_precompile_value(arg) for arg in args]
        serialized_kwargs = _serialize_precompile_value(kwargs)

        fn_module = self.fn.__module__
        fn_qualname = self.fn.__qualname__

        # Restrict worker subprocesses to the parent's current CUDA device.
        # Without this, all workers default to cuda:0 and their CUDA context
        # initialization can OOM when many ranks share a node. The parent-PID
        # env arms each worker's watchdog so an orphaned worker self-SIGKILLs
        # instead of lingering on long-lived self-hosted CI runners.
        worker_env = os.environ.copy()
        current_device = _get_current_cuda_device()
        if current_device is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = current_device
        worker_env["QUACK_COMPILE_WORKER_PARENT_PID"] = str(os.getpid())

        # Launch persistent worker pool
        workers = []
        for _ in range(max_workers):
            p = subprocess.Popen(
                [sys.executable, "-m", "torch._vendor.quack._compile_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if not verbose else None,
                env=worker_env,
            )
            ready = _recv_from_worker(p.stdout)
            if ready != "READY":
                p.kill()
                continue
            workers.append(p)

        if not workers:
            return _PrecompileHandle()

        # Round-robin dispatch + register per-config events. We do all sends
        # up front (pipe buffers absorb the small pickled payloads) and let
        # the per-worker reader threads drain replies asynchronously — the
        # bench loop calls handle.wait_for(i) for each config it's about to
        # benchmark, overlapping the still-pending compiles with GPU work.
        handle = _PrecompileHandle()
        handle._workers = workers
        assignments: List[List[int]] = [[] for _ in workers]
        for i, config in enumerate(configs):
            wi = i % len(workers)
            try:
                _send(
                    workers[wi].stdin,
                    {
                        "fn_module": fn_module,
                        "fn_qualname": fn_qualname,
                        "args": serialized_args,
                        "kwargs": serialized_kwargs,
                        "config_kwargs": config.all_kwargs(),
                    },
                )
            except (BrokenPipeError, OSError):
                # Worker died before we even finished dispatching. Mark this
                # config failed and continue to the next worker.
                handle.events[i] = threading.Event()
                handle.events[i].set()
                handle.failures[i] = "worker died before dispatch"
                continue
            assignments[wi].append(i)
            handle.events[i] = threading.Event()

        # Spawn one reader thread per worker. Each thread iterates the
        # config_indices it was assigned in dispatch order and signals events
        # as replies come back. ``daemon=True`` so a stuck reader doesn't
        # block process exit.
        for wi, w in enumerate(workers):
            if not assignments[wi]:
                continue
            t = threading.Thread(
                target=_reader_thread_main,
                name=f"quack-precompile-reader-{wi}",
                args=(handle, w, assignments[wi]),
                daemon=True,
            )
            t.start()
            handle._reader_threads.append(t)

        return handle

    def _bench(self, *args, config, **meta):
        verbose = os.environ.get(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        if verbose:
            print(f"Autotuning kernel {self.fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        # Default path: L2-cold CUDA-graph round-robin bench. ``__call__``
        # sets ``self._l2_cold_arg_sets`` / ``self._l2_cold_kwarg_sets`` to
        # pre-cloned (args, kwargs) sets once per shape (reused across all
        # configs). Round-robin over fresh sets keeps the kernel measured
        # under the cache-cold conditions that match production access
        # patterns, so the autotuner picks configs that win at the same
        # workload the user actually runs.
        l2_cold_arg_sets = getattr(self, "_l2_cold_arg_sets", None)
        l2_cold_kwarg_sets = getattr(self, "_l2_cold_kwarg_sets", None)
        has_hooks = self.pre_hook is not None or self.post_hook is not None
        use_l2_cold = (
            self._do_bench is None
            and l2_cold_arg_sets is not None
            and l2_cold_kwarg_sets is not None
            and not has_hooks
        )

        if use_l2_cold:
            try:
                return _bench_cuda_graph_l2_rotate(
                    self.fn,
                    l2_cold_arg_sets,
                    l2_cold_kwarg_sets,
                    extra_kwargs=config.all_kwargs(),
                    quantiles=(0.5, 0.2, 0.8),
                )
            except (RuntimeError, MemoryError) as e:
                # Narrow catch: only swallow GPU-side failures (smem
                # overflow, kernel launch errors, OOM). Programming errors
                # (TypeError, AssertionError, ValueError from conflict check
                # above) propagate so the user sees them.
                if verbose:
                    print(f"Autotuning failed with {type(e).__name__}: {e}")
                return [float("inf"), float("inf"), float("inf")]

        # Legacy path: triton.testing.do_bench or user-supplied do_bench.
        # Used when (a) a custom do_bench was passed via the decorator's
        # ``do_bench=`` arg, or (b) pre/post hooks are configured (the
        # clone/restore inside hooks doesn't work under CUDA graph capture).
        def kernel_call():
            if self.pre_hook is not None:
                self.pre_hook(full_nargs)
            try:
                self.fn.__call__(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    if self.post_hook is not None:
                        self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            if self.post_hook is not None:
                self.post_hook(full_nargs, exception=None)

        try:
            return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except Exception as e:
            if verbose:
                print(f"Autotuning failed with {e}")
            return [float("inf"), float("inf"), float("inf")]

    @torch.compiler.disable
    def check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key:
            bench_fn()
            return

        fn = self.fn
        config_str_list = [str(c) for c in configs]
        assert len(config_str_list) == len(set(config_str_list)), "Config strings must be unique"
        cache_key = [VERSION, str(tuning_key)] + config_str_list
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = FileCacheManager(_base32(cache_key))
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        # There's an environment variable to force cache update
        if path and not os.environ.get(f"{PACKAGE_NAME.upper()}_FORCE_CACHE_UPDATE", False):
            str2config = {s: c for s, c in zip(config_str_list, configs)}
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {str2config[config]: timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
                self.bench_time = 0
            return

        bench_fn()
        cache.put(
            json.dumps(
                {
                    "key": tuning_key,
                    "configs_timings": [
                        (str(config), timings) for config, timings in self.configs_timings.items()
                    ],
                }
            ),
            file_name,
            binary=False,
        )

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            # Need "str" to make it json-serializable
            key = [str(_args[key]) for key in self.keys if key in _args]
            for _, arg in _args.items():
                if isinstance(arg, Tensor):
                    key.append(str(arg.shape))
                    # If stride != 0, 1, we just cache it as 2
                    key.append(str([s if s in {0, 1} else 2 for s in arg.stride()]))
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                @torch.compiler.disable  # Don't want any tracing here
                def benchmark():
                    # Two new mechanisms cooperate here:
                    #   (a) ``_precompile`` returns a ``_PrecompileHandle``
                    #       whose ``wait_for(i)`` blocks until config i is
                    #       done compiling in the subprocess pool; remaining
                    #       configs continue compiling in background reader
                    #       threads, overlapping with this loop's GPU work.
                    #       Total wall = max(parallel_compile, serial_bench).
                    #   (b) The L2-cold ``_bench`` path needs pre-cloned
                    #       (args, kwargs) sets attached to ``self`` once per
                    #       shape so all configs share the same buffer set.
                    # Both share one try/finally so handle.shutdown() and the
                    # L2-cold set cleanup run even on exception (incl. OOM
                    # inside ``_gpu_warmup``).
                    handle = self._precompile(*args, configs=pruned_configs, **kwargs)
                    bench_start = time.time()
                    verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
                    has_hooks = self.pre_hook is not None or self.post_hook is not None
                    timings = {}
                    try:
                        _gpu_warmup()
                        # Pre-allocate cloned (args, kwargs) sets once per
                        # shape; the same sets are reused across all configs
                        # to avoid ~400x re-cloning. Skipped when hooks are
                        # present or a custom do_bench was supplied (legacy
                        # fallback in _bench).
                        if self._do_bench is None and not has_hooks:
                            try:
                                n_buffers = _pick_l2_rotate_count(args, kwargs)
                                arg_sets, kwarg_sets = _clone_l2_rotate_inputs(
                                    args, kwargs, n_buffers
                                )
                                self._l2_cold_arg_sets = arg_sets
                                self._l2_cold_kwarg_sets = kwarg_sets
                            except (RuntimeError, MemoryError):
                                # Cloning failed (likely OOM at extreme N);
                                # legacy do_bench path will be used by _bench.
                                self._l2_cold_arg_sets = None
                                self._l2_cold_kwarg_sets = None
                        else:
                            self._l2_cold_arg_sets = None
                            self._l2_cold_kwarg_sets = None

                        for i, config in enumerate(pruned_configs):
                            # Block until this config's compile has finished
                            # in the subprocess pool. The other configs keep
                            # compiling in their reader threads while we
                            # benchmark, giving us the parallel/serial overlap.
                            handle.wait_for(i)
                            # If the subprocess pool failed to compile this
                            # config (worker crashed, ERR: reply, etc.), warm
                            # jit_cache in-process FIRST so the bench time
                            # below excludes the compile cost — otherwise the
                            # artificially-slow timing would mask a config
                            # that's actually the best.
                            if handle.is_failed(i):
                                if verbose:
                                    print(
                                        f"[autotune] config {i} subprocess "
                                        f"compile failed ({handle.failures[i]}); "
                                        f"falling back to in-process compile "
                                        f"before benchmarking"
                                    )
                                try:
                                    current = dict(kwargs, **config.all_kwargs())
                                    self.fn(*args, **current)
                                except Exception:
                                    # _bench below will record float('inf')
                                    # if the kernel raises during the run.
                                    pass
                            timings[config] = self._bench(*args, config=config, **kwargs)
                    finally:
                        # Free L2-cold sets before persisting the cache so the
                        # user's subsequent .fn(...) call has full HBM.
                        self._l2_cold_arg_sets = None
                        self._l2_cold_kwarg_sets = None
                        # Always shutdown to avoid orphan workers / dangling
                        # reader threads, even if a bench raised.
                        handle.shutdown()
                    bench_end = time.time()
                    if verbose:
                        for config, time_ in timings.items():
                            print(f"[{config}] -> {time_[0]:.3f}ms")
                    # Surface bench failures (configs returning inf timings)
                    # so smem-overflow / launch errors aren't silently masked.
                    n_failed = sum(1 for t in timings.values() if t[0] == float("inf"))
                    if n_failed:
                        print(
                            f"quack autotune: {n_failed}/{len(timings)} configs "
                            f"failed for {self.fn.__name__}{key}; "
                            f"set {PACKAGE_NAME.upper()}_PRINT_AUTOTUNING=1 for details",
                            file=sys.stderr,
                        )
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    self.configs_timings = timings

                if self.cache_results:
                    self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if (
            os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
            and not used_cached_result
        ):
            print(
                f"{PACKAGE_NAME} autotuning for function {self.fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        ret = self.fn.__call__(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs: Dict) -> List[Any]:
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError(
                    "Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int"
                )

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs


class AutotuneConfig:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})

    def all_kwargs(self):
        return self.kwargs

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        return ", ".join(res)

    def __hash__(self):
        return hash(tuple(self.all_kwargs().items()))

    def __eq__(self, other):
        self_tuple = tuple(self.all_kwargs().items())
        other_tuple = tuple(other.all_kwargs().items())
        return self_tuple == other_tuple


def autotune(
    configs,
    key=None,
    prune_configs_by=None,
    restore_value=None,
    do_bench=None,
    cache_results=True,
    precompile_configs=True,
):
    f"""
    Decorator for auto-tuning a function function.

    .. highlight:: python

    If the environment variable :code:`{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING` is set to
    :code:`"1"`, we will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`AutotuneConfig` objects
    :type configs: list[AutotuneConfig]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    :param precompile_configs: whether to warm config compile caches in worker subprocesses.
    "type cache_results: bool
    """

    if key is None:
        key = []

    def decorator(fn):
        return Autotuner(
            fn,
            key,
            configs,
            restore_value=restore_value,
            prune_configs_by=prune_configs_by,
            do_bench=do_bench,
            cache_results=cache_results,
            precompile_configs=precompile_configs,
        )

    return decorator
