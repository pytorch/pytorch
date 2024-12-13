# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import functools
import logging
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from time import time
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.utils import dynamo_timed, set_feature_use
from torch._inductor import config
from torch._inductor.codecache import (
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    TritonCodeCache,
    TritonFuture,
)
from torch._inductor.compile_worker.subproc_pool import SubprocPool
from torch._inductor.runtime.compile_tasks import (
    _set_triton_ptxas_path,
    _worker_compile_triton,
)
from torch.hub import _Faketqdm, tqdm
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from torch._inductor.runtime.hints import HalideMeta

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: Optional[float] = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")

log = logging.getLogger(__name__)


def pre_fork_setup():
    """
    Setup that must be done prior to forking with a process pool.
    """
    # ensure properties have been calculated before processes
    # are forked
    caching_device_properties()

    # Computing the triton key can be slow. If we call it before fork,
    # it will be cached for the forked subprocesses.
    try:
        from triton.compiler.compiler import triton_key

        triton_key()
    except ImportError:
        # Triton might not be installed or might be an old version.
        pass


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
_pool_set: Set[SubprocPool] = set()


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


def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


class AsyncCompile:
    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert get_compile_threads() > 1
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> SubprocPool:
        assert get_compile_threads() > 1
        # Wrapper around ProcessPoolExecutor forks in a new process we control
        log.info("Creating subprocess pool with %d workers", get_compile_threads())
        pool = SubprocPool(get_compile_threads())

        # Set an attribute we can check to see if the pool is ready.
        pool.ready_future = pool.submit(AsyncCompile._get_ready)  # type: ignore[attr-defined]
        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if get_compile_threads() <= 1:
            return
        _compile_start()
        # Pool is initialized on first access
        cls.process_pool()
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    def _use_process_pool(self):
        return (
            get_compile_threads() > 1
            and self.process_pool().ready_future.done()  # type: ignore[attr-defined]
        )

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()
        _set_triton_ptxas_path()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return getattr(
                torch._inductor.codecache.PyCodeCache.load(source_code), kernel_name
            )

        kernel = TritonCodeCache.load(kernel_name, source_code)
        if self._use_process_pool():
            set_feature_use(
                "pytorch/inductor:enable_parallel_compile_version (post_warmup)", True
            )
            # We want to support changing these env vars after (and while) the
            # process pool is running, so pass them to the subprocess to reset.
            env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
            extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}
            return TritonFuture(
                kernel,
                self.process_pool().submit(
                    _worker_compile_triton,
                    kernel._reload_in_subproc,
                    extra_env,
                ),
            )
        else:
            set_feature_use(
                "pytorch/inductor:enable_parallel_compile_version (post_warmup)", False
            )
            with dynamo_timed(
                "async_compile.precompile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
            ):
                kernel.precompile()
            return kernel

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: List[str], source_code: str):
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cuda(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                CUDACodeCache.compile(source_code, "o")
            return CUDACodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def rocm(
        self,
        source_code,
        dst_file_ext,
        aot_compile=False,
    ):
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="o")
            if config.rocm.generate_test_runner:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="exe")
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if get_compile_threads() <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def wait(self, scope: Dict[str, Any]) -> None:
        with dynamo_timed(
            "async_compile.wait",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="triton_compile_time_us",
            log_waitcounter=True,
        ):
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
            if get_compile_threads() > 1:
                for key, result in scope.items():
                    if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                        pbar.set_postfix_str(key)
                    if isinstance(result, (Future, CodeCacheFuture)):
                        try:
                            scope[key] = result.result()
                        except BrokenProcessPool as e:
                            raise RuntimeError(
                                "A compilation subprocess exited unexpectedly. This "
                                "is likely due to a crash. To facilitate debugging, "
                                "you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 "
                                "to cause compilation to occur in the main process."
                            ) from e
                        pbar.update(1)

            _compile_end()


if (
    os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
    or os.environ.get("TORCH_WARM_POOL", "1") != "1"
    # The subprocess pool is only used for the Triton backend
    or not has_triton_package()
    # Skip for fbcode. We have internal reports of usages inside multiprocessing
    # pools that lead a multiplicative number of compile subprocesses.
    or config.is_fbcode()
):
    pass
else:
    AsyncCompile.warm_pool()

# On exit give the workers a chance to clean themselves up. Without this the
# resource_tracker can complain about leaked semaphores coming from the
# ProcessPoolExecutor:
#   UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects
#   to clean up at shutdown
atexit.register(shutdown_compile_workers)
