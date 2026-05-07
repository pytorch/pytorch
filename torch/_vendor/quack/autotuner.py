# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import os
import time
import inspect
import base64
import hashlib
import json
from pathlib import Path
from functools import cached_property, partial
from typing import Dict, Tuple, List, Optional, Any

import torch
from torch import Tensor

import triton

from . import __version__


PACKAGE_NAME = "quack"
VERSION = __version__


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

    def _precompile(self, *args, configs, **kwargs):
        """Pre-compile all configs in parallel subprocesses to populate .o cache.

        cute.compile() is not thread-safe (MLIR thread-local state) and fork after
        CUDA init causes segfaults. So we spawn persistent subprocess workers: each
        has its own CUDA context, creates FakeTensors matching the parent's tensor
        metadata, and compiles with COMPILE_ONLY=True. Workers stay alive to amortize
        import overhead across multiple configs. The parent then loads instantly from
        the .o cache during benchmarking.
        """
        from .cache_utils import CACHE_ENABLED

        if not CACHE_ENABLED:
            return

        max_workers = min(len(configs), int(os.getenv("QUACK_COMPILE_WORKERS", "8")))
        if max_workers <= 1:
            return

        # Quick check: compile first config in-process. If it loads from .o cache
        # (<0.5s), the rest are likely cached too — skip spawning workers.
        t_check = time.time()
        try:
            current = dict(kwargs, **configs[0].all_kwargs())
            self.fn(*args, **current)
        except Exception:
            pass
        if time.time() - t_check < 0.5:
            return

        verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        if verbose:
            print(f"Pre-compiling {len(configs)} configs with {max_workers} workers")
        t0 = time.time()

        import pickle
        import struct
        import subprocess
        import sys

        def _send(stream, msg):
            data = pickle.dumps(msg)
            stream.write(struct.pack("<I", len(data)))
            stream.write(data)
            stream.flush()

        def _recv(stream):
            header = stream.read(4)
            if len(header) < 4:
                return None
            length = struct.unpack("<I", header)[0]
            return pickle.loads(stream.read(length)) if length else None

        # Serialize tensor metadata
        tensor_meta = []
        for arg in args:
            if isinstance(arg, Tensor):
                tensor_meta.append(
                    {
                        "shape": list(arg.shape),
                        "stride": list(arg.stride()),
                        "dtype": str(arg.dtype),
                    }
                )
            else:
                tensor_meta.append(arg)

        fn_module = self.fn.__module__
        fn_qualname = self.fn.__qualname__

        # Restrict worker subprocesses to the parent's current CUDA device.
        # Without this, all workers default to cuda:0 and their CUDA context
        # initialization can OOM when many ranks share a node.
        worker_env = os.environ.copy()
        current_device = _get_current_cuda_device()
        if current_device is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = current_device

        # Launch persistent worker pool
        workers = []
        for _ in range(max_workers):
            p = subprocess.Popen(
                [sys.executable, "-m", "quack._compile_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if not verbose else None,
                env=worker_env,
            )
            ready = _recv(p.stdout)
            if ready != "READY":
                p.kill()
                continue
            workers.append(p)

        if not workers:
            return

        # Round-robin dispatch configs to workers
        pending = [0] * len(workers)
        for i, config in enumerate(configs):
            w = workers[i % len(workers)]
            _send(
                w.stdin,
                {
                    "fn_module": fn_module,
                    "fn_qualname": fn_qualname,
                    "tensor_meta": tensor_meta,
                    "kwargs": kwargs,
                    "config_kwargs": config.all_kwargs(),
                },
            )
            pending[i % len(workers)] += 1

        # Collect all results
        for wi, w in enumerate(workers):
            for _ in range(pending[wi]):
                _recv(w.stdout)

        # Shutdown workers (close stdin → worker exits)
        for w in workers:
            w.stdin.close()
            w.wait()

        if verbose:
            print(f"Pre-compilation done in {time.time() - t0:.1f}s")

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
                    self._precompile(*args, configs=pruned_configs, **kwargs)
                    _gpu_warmup()
                    bench_start = time.time()
                    timings = {
                        config: self._bench(*args, config=config, **kwargs)
                        for config in pruned_configs
                    }
                    bench_end = time.time()
                    if os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1":
                        for config, time_ in timings.items():
                            print(f"[{config}] -> {time_[0]:.3f}ms")
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
