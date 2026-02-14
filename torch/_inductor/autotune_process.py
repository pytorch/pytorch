# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import ctypes
import dataclasses
import functools
import logging
import multiprocessing as mp
import os
import pickle
import queue
import selectors
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p, CDLL
from typing import Any, IO, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import (
    CppCodeCache,
    CUDACodeCache,
    DLLWrapper,
    get_hash,
    PyCodeCache,
)
from torch._inductor.utils import (
    do_bench_using_profiling,
    get_gpu_type,
    get_ld_library_path,
    is_gpu,
    python_subprocess_env,
)
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from types import ModuleType

    from torch._inductor.select_algorithm import (
        ChoiceCaller,
        PartialRender,
        TritonTemplateCaller,
    )

from . import config
from .runtime.benchmarking import benchmarker
from .virtualized import V


CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

autotuning_log = getArtifactLogger(__name__, "autotuning")


class NonzeroWorkspaceNotSupportedError(Exception):
    pass


class TuningProcess:
    """
    Class to launch and interact with a benchmarking subprocess.
    """

    @staticmethod
    def process_main(read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None:
        """
        Entry point for the child process.
        """
        autotuning_log.debug(
            "Started autotune subprocess %s. Visible devices: %s",
            os.getpid(),
            os.environ.get(CUDA_VISIBLE_DEVICES),
        )

        def workloop():
            while True:
                job, extra_env = TuningProcess.recv(read_pipe)
                if job is None:
                    # None is a sentinel for the child to shut down
                    break
                try:
                    if extra_env:
                        os.environ.update(extra_env)
                    result = job()
                except Exception as e:
                    result = e
                TuningProcess.send(result, write_pipe)

        try:
            workloop()
        except EOFError:
            # The parent closed the pipe
            pass

    @staticmethod
    def send(
        obj: Any, write_pipe: IO[bytes], extra_env: dict[str, str] | None = None
    ) -> None:
        pickle.dump((obj, extra_env), write_pipe)
        write_pipe.flush()

    @staticmethod
    def recv(read_pipe: IO[bytes]) -> Any:
        return pickle.load(read_pipe)

    def __init__(self, device: Optional[int]):
        self.device = device
        self.start()

    def start(self):
        """
        Start the benchmarking subprocess.
        """
        entry = os.path.join(os.path.dirname(__file__), "__autotune_main__.py")

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")

        self.selector = selectors.DefaultSelector()
        self.selector.register(self.read_pipe, selectors.EVENT_READ)

        cmd = [
            sys.executable,
            entry,
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
        ]
        env = {
            **python_subprocess_env(),
            # We shouldn't be using the Triton async compile subprocess pool,
            # but as a precaution set the env var that disables its creation.
            "TORCH_WARM_POOL": "0",
            # Some internal usages need a modified LD_LIBRARY_PATH.
            "LD_LIBRARY_PATH": get_ld_library_path(),
            # This will cause the subprocs to profile using the profiler.
            "TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING": "1"
            if config.profile_bandwidth_with_do_bench_using_profiling
            else "0",
        }
        if self.device is not None:
            env[CUDA_VISIBLE_DEVICES] = str(self.device)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            pass_fds=(subproc_read_fd, subproc_write_fd),
        )
        os.close(subproc_read_fd)
        os.close(subproc_write_fd)

        self.running = True

    def alive(self) -> bool:
        """
        True if the subprocess is still running.
        """
        return self.running and self.process.poll() is None

    def put(self, req: Any, extra_env: dict[str, str] | None = None) -> None:
        """
        Push a work item to the child process.
        """
        if not self.alive():
            self.start()
        TuningProcess.send(req, self.write_pipe, extra_env=extra_env)

    def get(self, timeout: float = 120.0) -> Any:
        """
        Get a response from the child process. Raises TimeoutError on timeout;
        raises EOFError if the subprocess crashes.
        """
        try:
            if not self.selector.select(timeout):
                raise TimeoutError(f"Timeout in autotune subprocess {self.process.pid}")
            result, _ = TuningProcess.recv(self.read_pipe)
        except TimeoutError:
            self.kill()
            raise
        except EOFError:
            # The subprocess crashed
            self.close()
            raise
        except Exception:
            autotuning_log.exception(
                "Unexpected exception in autotune subprocess %s", self.process.pid
            )
            self.kill()
            raise

        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self, wait: bool = True) -> None:
        """
        Signal the child process to shut down gracefully.
        """
        if self.alive():
            TuningProcess.send(None, self.write_pipe)
        if wait:
            self.wait()

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.alive():
            self.process.wait()
        self.close()

    def close(self) -> None:
        """
        Close resources.
        """
        self.selector.close()
        self.read_pipe.close()
        self.write_pipe.close()
        self.running = False

    def kill(self) -> None:
        """
        Send a SIGKILL to the child process.
        """
        if self.alive():
            autotuning_log.error(
                "Sending SIGKILL to autotune subprocess %d",
                self.process.pid,
            )
            self.process.kill()
        self.close()

    def restart(self) -> None:
        """
        Gracefully restarts the child process.
        """
        self.shutdown(wait=True)
        self.start()


class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    def __init__(self) -> None:
        """
        Start the child processes.
        """
        devices = self.get_device_list()
        autotuning_log.debug("Sub-process autotune device list: %s", devices)

        # Launch the child processes.
        self.processes = [TuningProcess(device=device) for device in devices]

        self.process_queue: queue.Queue[TuningProcess] = queue.Queue()
        for p in self.processes:
            self.process_queue.put(p)

        # Use a thread pool to manage distributing work to the subprocesses.
        # Threads block on an available process, so it makes sense to match
        # the number of threads with the number of devices.
        self.executor = ThreadPoolExecutor(max_workers=len(devices))

    @staticmethod
    def get_device_list() -> Sequence[Optional[int]]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # Don't use multiple devices
            return [None]

        gpu_type = get_gpu_type()
        device_interface = get_interface_for_device(gpu_type)
        count = device_interface.device_count()

        # If the user specified the visible devices in the env, use those.
        if CUDA_VISIBLE_DEVICES in os.environ:
            devices = [int(d) for d in os.environ[CUDA_VISIBLE_DEVICES].split(",")]
            assert len(devices) <= count
            return devices

        return list(range(count))

    def shutdown(self) -> None:
        """
        Signal all child processes to exit.
        """
        self.executor.shutdown()

        for p in self.processes:
            p.shutdown(wait=False)
        for p in self.processes:
            p.wait()

    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        assert choice.bmreq is not None

        env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}
        process = self.process_queue.get()
        process.put(choice.bmreq.benchmark, extra_env=extra_env)
        try:
            return process.get(
                config.max_autotune_subproc_result_timeout_seconds,
            )
        except TimeoutError:
            warnings.warn(
                f"Timed out benchmarking choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # Set to INF so this choice will be ignored
            return float("inf")
        except Exception as process_exception:
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # An unspecified launch failure (cudaErrorLaunchFailure) corrupts the
            # CUDA context, making it unrecoverable. All subsequent CUDA calls will
            # fail as well. The process must be restarted to restore CUDA functionality.
            if "cudaErrorLaunchFailure" in str(process_exception):
                process.restart()
            # Set to INF so this choice will be ignored
            return float("inf")
        finally:
            self.process_queue.put(process)

    def benchmark(
        self,
        choices: list[TritonTemplateCaller],
    ) -> dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """

        # Use a ThreadExecutorPool to spread the work across the subprocesses and
        # to grab subprocesses as soon as they're free.
        results = dict(zip(choices, self.executor.map(self.target, choices)))

        return results


LayoutOrBuffer = Union[ir.Layout, ir.Buffer]


@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: Optional[str] = None

    @classmethod
    def from_irnodes(
        cls, irnodes: Union[LayoutOrBuffer, Sequence[LayoutOrBuffer]]
    ) -> Union[TensorMeta, list[TensorMeta]]:
        if isinstance(irnodes, Sequence):
            result: list[Any] = [cls.from_irnodes(x) for x in irnodes]
            assert all(isinstance(x, TensorMeta) for x in result)
            return result

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)

        dtype = node.get_dtype()
        assert dtype is not None
        device = node.get_device()
        assert device is not None

        return TensorMeta(
            device=device,
            dtype=dtype,
            sizes=V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            strides=V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            offset=V.graph.sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
            name=node.get_name(),
        )

    def to_tensor(self) -> torch.Tensor:
        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: Iterable[Any],
    ) -> None:
        # the kernel name defined in the module
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            self.input_tensor_meta: list[TensorMeta] = [input_tensor_meta]
        else:
            self.input_tensor_meta: list[TensorMeta] = input_tensor_meta

        if output_tensor_meta and isinstance(output_tensor_meta, (tuple, list)):
            if len(output_tensor_meta) > 1:
                # Each output with same meta for Grouped GEMM
                assert all(
                    getattr(output_tensor_meta[0], attr) == getattr(x, attr)
                    for x in output_tensor_meta
                    for attr in ["device", "dtype", "sizes", "strides", "offset"]
                )
            self.output_tensor_meta = output_tensor_meta[0]
        else:
            self.output_tensor_meta: TensorMeta = output_tensor_meta

        self.extra_args = extra_args

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        raise NotImplementedError

    def cleanup_run_fn(self) -> None:
        pass

    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        raise NotImplementedError

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        debug = autotuning_log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        # create args and out tensor
        if out is None:
            assert self.input_tensor_meta and self.output_tensor_meta, (
                "Input and output tensor meta must be populated when input_tensors is empty"
            )
            assert len(input_tensors) == 0
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            out = self.output_tensor_meta.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()
        try:
            fn = self.make_run_fn(*input_tensors, out=out)
        except NonzeroWorkspaceNotSupportedError:
            # Skipping all ops with nonzero workspace requirements
            autotuning_log.info("Skipping op due to nonzero workspace requirement")
            return float("inf")

        if debug:
            load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()

        res = self.do_bench(fn, *input_tensors, out)

        if debug:
            bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            autotuning_log.debug(
                "InChildProcess %s: load %f, create tensor %f, bench %f",
                str(self),
                load_elapse,  # type: ignore[possibly-undefined]
                create_tensor_elapse,  # type: ignore[possibly-undefined]
                bench_elapse,
            )
        self.cleanup_run_fn()
        return res


class _TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file instead of the test file so the
    TuningProcess sub-process can unpickle these objects.
    """

    def __init__(
        self,
        result: float = 0.0,
        device: Optional[int] = None,
        sleep: Optional[float] = None,
        exc: Optional[Exception] = None,
        crash: bool = False,
    ):
        self.result = result
        self.device = device
        self.sleep = sleep
        self.exc = exc
        self.crash = crash

    def benchmark(
        self, *input_tensors: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> float:
        if self.device is not None:
            assert os.environ.get(CUDA_VISIBLE_DEVICES, None) == str(self.device)
        if self.sleep:
            time.sleep(self.sleep)
        if self.exc:
            raise self.exc
        if self.crash:
            sys.exit(1)
        return self.result


class GPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        device_idx_set = OrderedSet(
            tensor.device.index
            for tensor in [*input_tensors, out]
            if isinstance(tensor, torch.Tensor)
            and is_gpu(tensor.device.type)
            and tensor.device.index is not None
        )
        assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
        device_type = next(
            (
                tensor.device.type
                for tensor in input_tensors
                if is_gpu(tensor.device.type)
            ),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = device_interface.current_device()
        with device_interface.device(device_idx):  # type: ignore[attr-defined]
            res = benchmarker.benchmark(fn, device=device_type)
            device_interface.synchronize()  # shake out any CUDA errors

        return res


class CPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        return benchmarker.benchmark_cpu(fn)


class TritonBenchmarkRequest(BenchmarkRequest):
    """
    Represents a standalone benchmark request for a Triton Template.

    Important: Instances of this class have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: Iterable[Any],
        module_path: str,  # the path of the module defining the triton kernel
        module_cache_key: str,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        matrix_instr_nonkdim: int = 0,  # only used for hip to choose the shape of mfma instruction.
        waves_per_eu: int = 0,  # only used for hip to schedule waves per execution unit
        kpack: int = 0,  # ROCm specific gemm parameter
        workspace_size: Optional[int] = None,  # size of workspace buffer in bytes
        workspace_zero_fill: bool = False,  # whether to zero-fill workspace
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.matrix_instr_nonkdim = matrix_instr_nonkdim
        self.waves_per_eu = waves_per_eu
        self.kpack = kpack
        self.workspace_size = workspace_size
        self.workspace_zero_fill = workspace_zero_fill

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        autotuning_log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )

        run_method = getattr(mod, self.kernel_name).run
        extra_args = list(self.extra_args)

        # Recreate workspace tensor if needed (for TMA templates)
        # The workspace tensor couldn't be pickled, so we recreate it here
        # It should be inserted before the grid values (last 3 elements)
        if self.workspace_size is not None:
            from torch._inductor.select_algorithm import WORKSPACE_ARG_PLACEHOLDER

            workspace_tensor = torch.empty(
                (self.workspace_size,),
                dtype=torch.uint8,
                device=out.device,
            )
            if self.workspace_zero_fill:
                workspace_tensor.zero_()
            workspace_index = extra_args.index(WORKSPACE_ARG_PLACEHOLDER)
            extra_args[workspace_index] = workspace_tensor

        run_method.__self__.with_bandwidth_info = False

        # Newer version of triton add warmup argument to JITFunction.run.
        # This code handles backward-compatibility.
        warmup_arg = {}
        import inspect

        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False

        if out.device.type == "cpu":
            stream = 0
        else:
            device_type = out.device.type
            device_interface = get_interface_for_device(device_type)
            stream = device_interface.get_raw_stream(
                self.output_tensor_meta.device.index
            )

        if isinstance(
            getattr(mod, self.kernel_name),
            torch._inductor.runtime.triton_heuristics.DebugAutotuner,
        ):
            return functools.partial(
                run_method,
                *input_tensors,
                out,
                *extra_args,
                **warmup_arg,
                stream=stream,
            )
        else:
            return functools.partial(
                run_method,
                *input_tensors,
                out,
                *extra_args,
                **warmup_arg,
                stream=stream,
                benchmark_run=True,
            )

    def precompile(self):
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        getattr(mod, self.kernel_name).precompile()

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"


class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    pass


class TritonCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    pass


class ExternKernelBenchmarkRequest(BenchmarkRequest):
    """
    A class to handle extern kernel benchmark requests. This allows extern kernels
    (like aten::mm) to be benchmarked in a subprocess, similar to Triton kernels.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: Iterable[Any],
        callable_path: str,  # Module path to the callable (e.g., "extern_kernels.mm")
        kwargs: Optional[dict[str, Any]] = None,
        has_out_variant: bool = True,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.callable_path = callable_path
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        fn = self.to_callable()
        if self.has_out_variant:
            # For out=variant, pass output as keyword arg
            return functools.partial(fn, *input_tensors, out=out)
        else:
            # For non-out variant, just call with inputs
            return functools.partial(fn, *input_tensors)

    def benchmark(
        self, *input_tensors: torch.Tensor, out: Optional[torch.Tensor] = None
    ):
        if out is not None and out.numel() == 0:
            # no need to run the kernel of do benchmarking
            return 0.0
        if self.has_out_variant or len(input_tensors) == 0:
            return super().benchmark(*input_tensors, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*input_tensors)
            if out is not None:
                torch._C._dynamo.guards.assert_size_stride(
                    out_new, tuple(out.size()), tuple(out.stride())
                )
                out.copy_(out_new)  # for correctness checking
            if config.profile_bandwidth_with_do_bench_using_profiling:
                return do_bench_using_profiling(lambda: algo(*input_tensors))
            return benchmarker.benchmark(algo, input_tensors, {})

    def precompile(self) -> None:
        # Extern kernels don't need precompilation - they're already compiled
        pass

    def to_callable(self):
        # While ExternKernelChoice also has a to_callable method,
        # we avoid calling the ExternKernelChoice version here to make sure
        # this is picklable
        from torch._inductor.select_algorithm import extern_kernels

        fn = getattr(extern_kernels, self.kernel_name)
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)

        return fn

    def __str__(self) -> str:
        return f"ExternKernelBenchmarkRequest({self.callable_path})"


class ExternKernelGPUBenchmarkRequest(
    GPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    pass


class ExternKernelCPUBenchmarkRequest(
    CPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    pass


class CUDABenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    A class to handle CUDA (CUTLASS) benchmark requests. This class is for
    managing the lifecycle of a CUDA kernel benchmark, including compiling
    the source code, managing workspace memory, and executing the kernel.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.DLL: Optional[DLLWrapper] = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def precompile(self):
        """
        Precompile the CUDA source code to populate the CUDACodeCache.
        This may happen in a separate thread pool.
        """
        autotuning_log.debug("Precompiling %s", self)
        CUDACodeCache.compile(self.source_code, "so")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        """
        Create a function to run the CUDA kernel with the given input and output tensors.
        """

        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=out.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        ret = functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

        # sanity check to make sure we cleanup run fn properly
        try:
            ret()
        except RuntimeError as e:
            err_msg = str(e)

            def raise_runtime_error():
                raise RuntimeError(err_msg)

            self.cleanup_run_fn()
            return raise_runtime_error

        return ret

    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len(
            dict.fromkeys(meta.name for meta in self.input_tensor_meta)
        )
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        torch.cuda.synchronize()  # shake out any CUDA errors
        self.workspace_size = c_workspace_size.value
        autotuning_log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",  # noqa: B950
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
            self.DLL = None
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"


class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.hash_key = get_hash(source_code)
        self.DLL: Optional[Union[CDLL, ModuleType]] = None

    def precompile(self):
        # Prepopulate CppCodeCache
        # may happen in separate Threadpool
        autotuning_log.debug("Precompiling %s", self)
        CppCodeCache.load(self.source_code, device_type="cpu")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # TODO(jgong5): use CppPythonBindingsCodeCache for better binding perf
        self.DLL = CppCodeCache.load(self.source_code, device_type="cpu")
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        # Assume only size with type ctypes.c_ulonglong in extra_args
        assert all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args)
        run_method.argtypes = [ctypes.c_ulonglong] * (
            len(args) + len(list(self.extra_args))
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}"


class CuteDSLBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for CuteDSL (CUTLASS Python DSL) kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: tuple[Any, ...],
        source_code: PartialRender,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)

        finalized_code = source_code.finalize_all()
        self.module_cache_key, self.module_path = PyCodeCache.write(finalized_code)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        """
        Create a function to run the CuteDSL kernel with the given input and output tensors.
        Similar to TritonBenchmarkRequest.make_run_fn but for CuteDSL kernels.
        """
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)

        # Logic replicated async_compile
        from .codegen.cutedsl.cutedsl_kernel import MAIN_SUFFIX

        main_func_name = f"{self.kernel_name}_{MAIN_SUFFIX}"

        if not hasattr(mod, main_func_name):
            available = [name for name in dir(mod) if callable(getattr(mod, name))]
            raise RuntimeError(
                f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
            )

        kernel_func = getattr(mod, main_func_name)

        def run_kernel():
            device_interface = get_interface_for_device("cuda")
            stream = device_interface.get_raw_stream(out.device.index)
            return kernel_func(*input_tensors, out, stream=stream)

        return run_kernel


@functools.cache
def get_tuning_process_pool() -> TuningProcessPool:
    pool = TuningProcessPool()
    atexit.register(pool.shutdown)
    return pool


def benchmark_in_sub_process(
    choices: list[TritonTemplateCaller],
) -> dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    return get_tuning_process_pool().benchmark(choices)


class AutotuneProcessPool:
    """
    Singleton pool manager for running autotuning (precompilation + benchmarking)
    in a separate process.
    """

    _instance: Optional[AutotuneProcessPool] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._pool: ProcessPoolExecutor | None = self._init_pool()
        self._warmup_future: Future[Any] | None = None
        self._warmup_start_time: float | None = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # num_workers=1 to avoid GPU contention during benchmarking
                    cls._instance = cls()
        return cls._instance

    @property
    def pool(self):
        """Get the process pool."""
        if self._pool is None:
            self._pool = self._init_pool()
        return self._pool

    def _init_pool(self):
        """
        Get or create the process pool.

        Uses ProcessPoolExecutor with 'spawn' context for CUDA safety.
        ProcessPoolExecutor is lazily initialized - workers are not spawned
        until the first submit() call, making this property non-blocking.
        """
        # Use 'spawn' context to avoid CUDA fork issues
        # Workers are spawned lazily on first submit(), not here
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
        )
        atexit.register(self._shutdown)
        autotuning_log.info("AutotuneProcessPool created (workers spawn lazily)")

        return pool

    def warm_up(self) -> Future[Any]:
        """
        Submit a warmup job to eagerly spawn workers and initialize CUDA.

        This is optional - call it early to hide spawn latency.
        Returns the warmup future which can be ignored or awaited.
        """
        if self._warmup_future is None:
            with self._lock:
                if self._warmup_future is None:
                    self._warmup_start_time = time.perf_counter()
                    self._warmup_future = self.pool.submit(_warmup_autotune_subprocess)
                    self._warmup_future.add_done_callback(self._on_warmup_complete)
                    autotuning_log.info("Warmup job submitted")
        # pyrefly: ignore[bad-return]
        return self._warmup_future

    def _on_warmup_complete(self, future: Future[Any]) -> None:
        """Callback invoked when the warmup job completes."""
        warmup_elapsed_time = None
        if self._warmup_start_time is not None:
            warmup_elapsed_time = time.perf_counter() - self._warmup_start_time

        try:
            result = future.result()
            autotuning_log.error(
                "AutotuneProcessPool warmup completed successfully in %.4f seconds: %s",
                warmup_elapsed_time,
                result,
            )
        except Exception as e:
            autotuning_log.error(
                "AutotuneProcessPool warmup failed after %.4f seconds",
                warmup_elapsed_time,
            )
            raise e

    def submit(self, fn, *args, **kwargs) -> Future[Any]:
        """Submit a job to the pool and return a Future."""
        return self.pool.submit(fn, *args, **kwargs)

    def _shutdown(self):
        """Shutdown the pool on exit."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    @classmethod
    def shutdown_instance(cls):
        """Explicitly shutdown the singleton instance."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown()
                    cls._instance = None


def _warmup_autotune_subprocess() -> bool:
    """
    Warmup function run in the autotune subprocess.
    """
    import torch

    # Initialize dummy tensor for CUDA context
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    return True


def run_autotune_in_subprocess(
    benchmark_request: BenchmarkRequest,
) -> float:
    """
    Run autotuning benchmarks in a subprocess.

    This function is submitted to AutotuneProcessPool and runs in isolation
    to prevent GPU contention with the main compilation process.

    Args:
        picklable_choices: List of picklable choice information

    Returns:
        timing
    """

    try:
        # Run the benchmark directly - bmreq is already a BenchmarkRequest
        timing = benchmark_request.benchmark()

        return timing

    except Exception:
        autotuning_log.error(
            "Failed to benchmark choice %s",
            benchmark_request,
        )
        # Use infinity for failed benchmarks so they're not selected
        return float("inf")


class PrecompileThreadPool:
    """
    Thread pool for running precompilation asynchronously.

    This allows the main compilation process to continue while
    precompilation happens in background threads.
    """

    _instance: Optional[PrecompileThreadPool] = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @classmethod
    def get_instance(cls) -> PrecompileThreadPool:
        from torch._inductor.select_algorithm import get_num_workers

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(get_num_workers())
        return cls._instance

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def _shutdown(self, wait: bool = False):
        return self._executor.shutdown(wait=wait)

    @classmethod
    def shutdown_instance(cls) -> None:
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown(wait=False)
                    cls._instance = None


class AsyncAutotuner:
    """
    Handles asynchronous autotuning of kernel choices in a separate process.

    This class manages the lifecycle of autotuning:
    1. Accepts precompiled choices from the main process
    2. Submits benchmarking work to AutotuneProcessPool
    3. Returns results via a Future

    Usage:
        autotuner = AsyncAutotuner(choices)
        autotuner.start()  # Kicks off async benchmarking
        timings = autotuner.get_results()  # Blocks until complete
    """

    choice_hash_to_future = {}

    @staticmethod
    def get_choice_hash(choice: ChoiceCaller, inputs_key: str) -> str:
        return choice.hash_key() + inputs_key

    @classmethod
    def start(cls, choices: list[ChoiceCaller], inputs_key: str):
        """
        Start asynchronous autotuning in a subprocess.

        This method:
        1. Extracts picklable benchmark requests from choices
        2. Submits benchmarking work to AutotuneProcessPool
        3. Returns immediately (non-blocking)
        """

        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)

            if choice_hash in AsyncAutotuner.choice_hash_to_future:
                continue

            assert getattr(choice, "bmreq", None) is not None, (
                "bmreq is None for choice"
            )

            autotune_future = AutotuneProcessPool.get_instance().submit(
                run_autotune_in_subprocess,
                choice.bmreq,
            )

            AsyncAutotuner.choice_hash_to_future[choice_hash] = autotune_future

    @classmethod
    def get_results(
        cls, choices: list[ChoiceCaller], inputs_key: str
    ) -> dict[ChoiceCaller, float]:
        """
        Get autotuning results, blocking until complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            Dict mapping ChoiceCaller to benchmark timing
        """

        timings = {}
        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)
            timings[choice] = AsyncAutotuner.choice_hash_to_future[choice_hash].result()
        return timings
