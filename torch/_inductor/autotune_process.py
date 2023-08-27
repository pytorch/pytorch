import dataclasses
import functools
import logging
import queue
import time
import warnings
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache

if TYPE_CHECKING:
    from torch._inductor.select_algorithm import TritonTemplateCaller

from .utils import do_bench_using_profiling
from .virtualized import V

log = logging.getLogger(__name__)

DEBUG = False
EXIT_HANDLER_REGISTERED = False


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


@dataclasses.dataclass
class TuningProcess:
    process: Optional[BaseProcess] = None
    request_queue: Optional["Queue[Any]"] = None
    response_queue: Optional["Queue[Any]"] = None

    @staticmethod
    def process_main(
        request_queue: "Queue[Any]",
        response_queue: "Queue[Any]",
    ) -> None:
        print("enter child process main")
        while True:
            obj = request_queue.get()

            if obj is None:
                break  # None is a sentinel for the child to terminate
            elif isinstance(obj, Ping):
                response_queue.put(Pong())
            elif isinstance(obj, BenchmarkRequest):
                response_queue.put(obj.benchmark())
            else:
                raise RuntimeError(f"Invalid request type {type(obj)}")

    def valid(self) -> bool:
        return (
            self.process is not None
            and self.request_queue is not None
            and self.response_queue is not None
        )

    def clear(self) -> None:
        self.process = self.request_queue = self.response_queue = None

    def initialize(self) -> None:
        """
        Create child process, request/response queues and do the warm up.
        """
        if self.valid():
            return

        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")
        request_queue = self.request_queue = ctx.Queue()
        response_queue = self.response_queue = ctx.Queue()

        process = self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
            ),
        )
        process.start()

        # register the exit handler for the parent process so it will terminate
        # the child processes
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(lambda: self.terminate())

        # wait for the initialization to be done
        request_queue.put(Ping())
        resp = response_queue.get()
        assert isinstance(resp, Pong)

    def terminate(self) -> None:
        if self.valid():
            request_queue = self.request_queue
            assert request_queue is not None
            request_queue.put(None)
            process = self.process
            assert process is not None
            process.join()


tuning_process = TuningProcess()


LayoutOrBuffer = Union[ir.Layout, ir.Buffer]


@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: List[int]
    strides: List[int]
    offset: int

    @classmethod
    def from_irnodes(
        cls, irnodes: Union[LayoutOrBuffer, Tuple[LayoutOrBuffer], List[LayoutOrBuffer]]
    ) -> Union["TensorMeta", List["TensorMeta"]]:
        if isinstance(irnodes, (tuple, list)):
            result: List[Any] = [cls.from_irnodes(x) for x in irnodes]
            assert all(isinstance(x, TensorMeta) for x in result)
            return result

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)

        dtype = node.get_dtype()
        assert dtype is not None

        return TensorMeta(
            device=node.get_device(),
            dtype=dtype,
            sizes=V.graph.sizevars.size_hints(node.get_size()),
            strides=V.graph.sizevars.size_hints(node.get_stride()),
            offset=V.graph.sizevars.size_hint(node.get_layout().offset),
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
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Dict[str, Any],
    ):
        # the kernel name defined in the module
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            input_tensor_meta = [input_tensor_meta]
        self.input_tensor_meta = input_tensor_meta

        if isinstance(output_tensor_meta, (tuple, list)):
            assert len(output_tensor_meta) == 1
            output_tensor_meta = output_tensor_meta[0]
        self.output_tensor_meta = output_tensor_meta

        self.extra_args = extra_args

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        raise NotImplementedError()

    def cleanup_run_fn(self) -> None:
        pass

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        if DEBUG:
            start_ts = time.time()

        if DEBUG:
            load_elapse = time.time() - start_ts
            start_ts = time.time()

        # create args and out tensor
        if output_tensor is None:
            assert len(input_tensors) == 0
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            output_tensor = self.output_tensor_meta.to_tensor()

        if DEBUG:
            create_tensor_elapse = time.time() - start_ts
            start_ts = time.time()

        out = do_bench_using_profiling(
            self.make_run_fn(*input_tensors, output_tensor=output_tensor)
        )
        torch.cuda.synchronize()  # shake out any CUDA errors

        if DEBUG:
            bench_elapse = time.time() - start_ts
            print(
                f"InChidProcess {str(self)}: load {load_elapse}, "
                + f"create tensor {create_tensor_elapse}, bench {bench_elapse}, "
                + f"collected time {out}"
            )
        self.cleanup_run_fn()
        return out


class TritonBenchmarkRequest(BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Dict[str, Any],
        module_path: str,  # the path of the module defining the triton kernel
        module_cache_key: str,
        grid: List[int],
        num_stages: int,
        num_warps: int,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.grid = grid
        self.num_stages = num_stages
        self.num_warps = num_warps

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        if DEBUG:
            print(
                f"benchmark module key: {self.module_cache_key}, path: {self.module_path}"
            )

        run_method = getattr(mod, self.kernel_name).run

        return functools.partial(
            run_method,
            *input_tensors,
            output_tensor,
            *self.extra_args,
            grid=self.grid,
            num_stages=self.num_stages,
            num_warps=self.num_warps,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"


class CUDABenchmarkRequest(BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Dict[str, Any],
        source_code: str,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.DLL: Optional[DLLWrapper] = None
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(
            self.source_code, "so"
        )
        args = [
            c_void_p(tensor.data_ptr())
            for tensor in list(input_tensors) + [output_tensor]
        ]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

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
        self.workspace_size = c_workspace_size.value
        assert (
            self.workspace_size == 0
        ), "Autotune cache needs to be updated to support non-zero workspace_size!"
        self.workspace = torch.empty(
            self.workspace_size, dtype=torch.uint8, device="cuda"
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            c_void_p(self.workspace.data_ptr()),  # set workspace ptr
            stream_ptr,
        )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"


def benchmark_in_sub_process(
    choice: "TritonTemplateCaller",
) -> float:
    """
    Do benchmarking in subprocess and return the perf number (latency).
    """
    assert choice.bmreq is not None
    tuning_process.initialize()
    assert tuning_process.valid()
    process, request_queue, response_queue = (
        tuning_process.process,
        tuning_process.request_queue,
        tuning_process.response_queue,
    )
    assert (
        process is not None and request_queue is not None and response_queue is not None
    )

    request_queue.put(choice.bmreq)
    while True:
        try:
            timing = response_queue.get(timeout=1.0)
        except queue.Empty:
            status = process.exitcode
            if status is None:
                # child process is still running
                continue
            # child process fail
            assert status != 0

            warnings.warn(
                f"Fail to benchmark choice '{choice}'. It will be ignored. Please debug the root cause in case the choice can bring perf gains."  # noqa: B950 line too long
            )

            tuning_process.clear()

            # return INF so this choice will be ignored
            return float("inf")

        return timing
