# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import os
import sys
import time
import inspect
import base64
import hashlib
import json
from pathlib import Path
from functools import cached_property, partial
from typing import Dict, Tuple, List, Optional, Any
from torch._vendor.quack.bench.bench_utils import (
    _bench_cuda_graph_l2_rotate,
    _clone_l2_rotate_inputs,
    _pick_l2_rotate_count,
)
from torch._vendor.quack.cache import get_cache_dir

import torch
from torch import Tensor

import triton

from . import __version__


PACKAGE_NAME = "torch_vendor_quack"
VERSION = __version__


def get_home_dir():
    return os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home())


def default_cache_dir():
    return os.path.join(get_home_dir(), f".{PACKAGE_NAME}", "cache")


class FileCacheManager(triton.runtime.cache.FileCacheManager):
    def __init__(self, key):
        super().__init__(key)
        self.cache_dir = get_cache_dir() or default_cache_dir()
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


#: How long a deferred config may wait on its pool compile before the bench
#: loop stops trusting the pool and benches it with the pool suppressed
#: (in-process compile). Guards against a wedged worker / a foreign flock
#: holder that never produces the .o; without it a permanently-"pending"
#: sha would rotate forever. Tests override this.
_POOL_WEDGE_TIMEOUT_S = 300.0


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
# Candidate-config compilation
#
# There is no separate precompile phase: the bench loop in ``benchmark()``
# (inside ``Autotuner.__call__``) runs under ``pool_scope()`` from
# quack.cache.async_compile. A config whose kernel misses the .o cache
# raises ``CompilePending`` from jit_cache after shipping the pickled
# ``_compile_*`` key to a CPU worker; the loop rotates that config to the
# back and benches whichever config is ready. Total wall stays
# max(parallel_compile, serial_bench), key discovery uses the real tensors
# in-process, and workers never launch kernels (they call the tensor-free
# ``_compile_*`` functions directly).
# ---------------------------------------------------------------------------


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
        self._arg_name_set = frozenset(self.arg_names)
        self.cache_results = (
            cache_results or os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_AUTOTUNING", None) == "1"
        )

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
                    # str(): the key tuple holds torch dtypes/shape tuples, and
                    # this field is informational (only configs_timings is read back)
                    "key": str(tuning_key),
                    "configs_timings": [
                        (str(config), timings) for config, timings in self.configs_timings.items()
                    ],
                }
            ),
            file_name,
            binary=False,
        )

    def __call__(self, *args, **kwargs):
        used_cached_result = True
        if len(self.configs) > 1:
            # Cache-hit fast path: build the key straight from args/kwargs —
            # this runs on EVERY tuned call, so no dict merges and no str()
            # formatting (together measured ~26us/call for a medium GEMM's arg
            # list). Plain-value tuples; stringified only when persisted to
            # the disk cache (see check_disk_cache / cache.put). The merged
            # named-args view is materialized only on a miss, for pruning.
            key = [kwargs[k] for k in self.keys if k in kwargs]
            arg_name_set = self._arg_name_set
            for arg in args:
                if isinstance(arg, Tensor):
                    # tuple(arg.shape), not arg.shape: torch.Size would change the
                    # str() of the key and invalidate on-disk autotune caches.
                    key.append(tuple(arg.shape))
                    # If stride != 0, 1, we just cache it as 2 (strides are never negative)
                    key.append(tuple([s if s < 2 else 2 for s in arg.stride()]))
                    key.append(arg.dtype)
            for name, arg in kwargs.items():
                if isinstance(arg, Tensor) and name in arg_name_set:
                    key.append(tuple(arg.shape))
                    key.append(tuple([s if s < 2 else 2 for s in arg.stride()]))
                    key.append(arg.dtype)
            key = tuple(key)
            if key not in self.cache:
                self.nargs = dict(zip(self.arg_names, args))
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                @torch.compiler.disable  # Don't want any tracing here
                def benchmark():
                    # Compile/bench overlap via the async compile pool
                    # (quack.cache.async_compile): the bench loop runs inside
                    # pool_scope(). A config whose kernel isn't compiled yet
                    # raises CompilePending from jit_cache (after shipping the
                    # key to a CPU worker); the loop rotates it to the back
                    # and benches whichever config is ready, retrying once its
                    # .o lands. Discovery happens in-process with the real
                    # tensors (no fake-tensor reconstruction), and the pool
                    # workers replay the pickled _compile_* key directly --
                    # which never launches, by construction.
                    #
                    # CompilePending can only fire OUTSIDE CUDA graph capture:
                    # the L2-cold bench does priming launches before capture,
                    # and the legacy do_bench path warms up first, so a cold
                    # key raises at the first plain launch.
                    from collections import deque

                    from torch._vendor.quack.cache.async_compile import (
                        CompilePending,
                        pool_scope,
                        suppress_pool,
                    )

                    bench_start = time.time()
                    verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
                    has_hooks = self.pre_hook is not None or self.post_hook is not None
                    timings = {}
                    _MAX_ATTEMPTS = 20
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

                        with pool_scope() as pool:
                            queue = deque(pruned_configs)
                            awaiting = {}  # id(config) -> sha
                            attempts = {}  # id(config) -> int
                            deadline = {}  # id(config) -> wedge deadline
                            spins = 0
                            while queue:
                                config = queue.popleft()
                                sha = awaiting.get(id(config))
                                wedged = sha is not None and time.monotonic() > deadline[id(config)]
                                if sha is not None and not wedged:
                                    state, _ = pool.poll(sha)
                                    if state == "pending":
                                        queue.append(config)
                                        spins += 1
                                        if spins >= len(queue):
                                            time.sleep(0.05)
                                            spins = 0
                                        continue
                                spins = 0
                                n = attempts.get(id(config), 0) + 1
                                attempts[id(config)] = n
                                try:
                                    if wedged or n > _MAX_ATTEMPTS:
                                        # Wedged pool: compile in-process so
                                        # the sweep always terminates.
                                        with suppress_pool():
                                            timings[config] = self._bench(
                                                *args, config=config, **kwargs
                                            )
                                    else:
                                        timings[config] = self._bench(
                                            *args, config=config, **kwargs
                                        )
                                except CompilePending as e:
                                    awaiting[id(config)] = e.sha
                                    deadline.setdefault(
                                        id(config),
                                        time.monotonic() + _POOL_WEDGE_TIMEOUT_S,
                                    )
                                    queue.append(config)
                    finally:
                        # Free L2-cold sets before persisting the cache so the
                        # user's subsequent .fn(...) call has full HBM.
                        self._l2_cold_arg_sets = None
                        self._l2_cold_kwarg_sets = None
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
        # Cheap flag first: the getenv (plus its f-string) is measurable on the
        # per-call cache-hit path.
        if not used_cached_result and (
            os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        ):
            print(
                f"{PACKAGE_NAME} autotuning for function {self.fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        ret = self.fn(*args, **kwargs, **config.all_kwargs())
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
    configs, key=None, prune_configs_by=None, restore_value=None, do_bench=None, cache_results=True
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
        )

    return decorator
