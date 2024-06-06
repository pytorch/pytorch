from __future__ import annotations

import functools
import logging
import multiprocessing
import os
import sys
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from time import time
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._inductor import config
from torch._inductor.codecache import (
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    TritonCodeCache,
    TritonFuture,
)
from torch._inductor.compile_worker.subproc_pool import (
    _warm_process_pool,
    AnyPool,
    SubprocPool,
)
from torch._inductor.compile_worker.watchdog import _async_compile_initializer

from torch._inductor.runtime.compile_tasks import (
    _set_triton_ptxas_path,
    _worker_compile_triton,
)

from torch.hub import _Faketqdm, tqdm

if TYPE_CHECKING:
    from torch._inductor.runtime.hints import HalideMeta

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: Optional[float] = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end() -> None:
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)


# Used to keep track of all process pools invoked so far.
_pool_set: Set[AnyPool] = set()


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    after_fork()


def after_fork():
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile.process_pool.cache_clear()


try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # register_at_fork does not exists on windows


class AsyncCompile:
    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        assert config.compile_threads > 1
        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(config.compile_threads)
        else:
            # ensure properties have been calculated before processes
            # are forked
            caching_device_properties()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = ProcessPoolExecutor(
                config.compile_threads,
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )
            # when this pool is created in a subprocess object, the normal exit handler
            # doesn't run, and we need to register our own handler.
            # exitpriority has to be high, because another one of the finalizers will
            # kill the worker thread that sends the shutdown message to the workers...
            multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if config.compile_threads <= 1:
            return
        _compile_start()
        _warm_process_pool(cls.process_pool(), config.compile_threads)
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()
        _set_triton_ptxas_path()

        kernel = TritonCodeCache.load(kernel_name, source_code)
        if config.compile_threads > 1:
            return TritonFuture(
                kernel,
                self.process_pool().submit(
                    _worker_compile_triton,
                    kernel._reload_in_subproc,
                ),
            )
        else:
            kernel.precompile()
            return kernel

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if config.compile_threads <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: List[str], source_code: str):
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if config.compile_threads <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cuda(self, source_code, dst_file_ext):
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)

        def task():
            return CUDACodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if config.compile_threads <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def wait(self, scope: Dict[str, Any]) -> None:
        num_kernels = len(
            [
                value
                for key, value in scope.items()
                if isinstance(value, (Future, CodeCacheFuture))
            ]
        )
        pbar = tqdm(
            total=num_kernels,
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        if config.compile_threads > 1:
            for key, result in scope.items():
                if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                    pbar.set_postfix_str(key)
                if isinstance(result, (Future, CodeCacheFuture)):
                    scope[key] = result.result()
                    pbar.update(1)

        _compile_end()


if (
    os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
    or os.environ.get("TORCH_WARM_POOL", "1") != "1"
):
    pass
else:
    AsyncCompile.warm_pool()
