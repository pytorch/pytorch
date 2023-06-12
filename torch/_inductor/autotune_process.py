import dataclasses
import queue
import time
import warnings
from typing import Any, Dict, List

import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import PyCodeCache

from .utils import do_bench
from .virtualized import V

DEBUG = False

DEBUG = False
EXIT_HANDLER_REGISTERED = False


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


@dataclasses.dataclass
class TuningProcess:
    process: multiprocessing.Process = None
    request_queue: multiprocessing.Queue = None
    response_queue: multiprocessing.Queue = None

    @staticmethod
    def process_main(request_queue, response_queue):
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

    def valid(self):
        return (
            self.process is not None
            and self.request_queue is not None
            and self.response_queue is not None
        )

    def clear(self):
        self.process = self.request_queue = self.response_queue = None

    def initialize(self):
        """
        Create child process, request/response queues and do the warm up.
        """
        if self.valid():
            return

        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
            ),
        )
        self.process.start()

        # register the exit handler for the parent process so it will terminate
        # the child processes
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(lambda: self.terminate())

        # wait for the initialization to be done
        self.request_queue.put(Ping())
        resp = self.response_queue.get()
        assert isinstance(resp, Pong)

    def terminate(self):
        if self.valid():
            self.request_queue.put(None)
            self.process.join()


tuning_process = TuningProcess()


@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: List[int]
    strides: List[int]
    offset: int

    @classmethod
    def from_irnodes(cls, irnodes):
        if isinstance(irnodes, (tuple, list)):
            return [cls.from_irnodes(x) for x in irnodes]

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)

        return TensorMeta(
            device=node.get_device(),
            dtype=node.get_dtype(),
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

    module_path: str  # the path of the module defining the triton kernel
    module_cache_key: str
    kernel_name: str  # the kernel name defined in the module
    grid: List[int]
    extra_args: Dict[str, Any]
    num_stages: int
    num_warps: int

    input_tensors: List[TensorMeta]
    output_tensor: TensorMeta

    def benchmark(self, *input_tensors, output_tensor=None) -> float:
        if DEBUG:
            start_ts = time.time()

        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        run = getattr(mod, self.kernel_name).run

        if DEBUG:
            load_elapse = time.time() - start_ts
            start_ts = time.time()

        # create args and out tensor
        if output_tensor is None:
            assert len(input_tensors) == 0
            input_tensors = [x.to_tensor() for x in self.input_tensors]
            output_tensor = self.output_tensor.to_tensor()

        if DEBUG:
            create_tensor_elapse = time.time() - start_ts
            start_ts = time.time()

        def worker():
            return run(
                *input_tensors,
                output_tensor,
                *self.extra_args,
                grid=self.grid,
                num_stages=self.num_stages,
                num_warps=self.num_warps,
            )

        out = do_bench(worker)
        torch.cuda.synchronize()  # shake out any CUDA errors

        if DEBUG:
            bench_elapse = time.time() - start_ts
            print(
                f"InChidProcess {self.module_cache_key}: load {load_elapse}, "
                + f"create tensor {create_tensor_elapse}, bench {bench_elapse}"
            )
        return out


def benchmark_in_sub_process(
    choice: "ChoiceCaller",
) -> float:
    """
    Do benchmarking in subprocess and return the perf number (latency).
    """
    assert choice.bmreq is not None
    tuning_process.initialize()
    assert tuning_process.valid()

    tuning_process.request_queue.put(choice.bmreq)

    while True:
        try:
            timing = tuning_process.response_queue.get(timeout=1.0)
        except queue.Empty:
            status = tuning_process.process.exitcode
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
