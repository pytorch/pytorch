import atexit
import dataclasses
import queue
import time
import warnings
from typing import Any, Dict, List, Optional

import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided

from torch._inductor import config, ir
from torch._inductor.codecache import PyCodeCache

from .utils import do_bench
from .virtualized import V

DEBUG = False

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
    dev_id: Optional[int] = None

    @staticmethod
    def process_main(request_queue, response_queue, dev_id):
        print(f"enter child process main for dev {dev_id}")
        torch.cuda.set_device(dev_id)
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

    def initialize(self, dev_id=None):
        """
        Create child process, request/response queues and do the warm up.
        """
        if self.valid():
            return

        if dev_id is not None:
            self.dev_id = dev_id

        assert self.dev_id is not None and self.dev_id >= 0
        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
                self.dev_id,
            ),
        )
        self.process.start()

        # wait for the initialization to be done
        self.request_queue.put(Ping())
        resp = self.response_queue.get()
        assert isinstance(resp, Pong)

    def terminate(self):
        if self.valid():
            self.request_queue.put(None)
            self.process.join()

    def __hash__(self):
        return id(self)


class TuningProcessPool:
    """
    Tuning process pool maintaining one process for each GPU. Recreate crashed
    process.
    """

    def __init__(self):
        self.avail_procs = set()
        self.all_procs = []

        # register the exit handler for the parent process so it will terminate
        # the child processes
        atexit.register(lambda: self.teardown())

    def initialize(self):
        """
        Not putting this in __init__ so we don't need create subprocess when
        importing the module.
        """
        if len(self.all_procs) > 0:  # already initialized
            return

        ngpu = torch.cuda.device_count()
        assert ngpu > 0
        print(f"Createing {ngpu} autotuning sub process one for each GPU")
        self.all_procs = [TuningProcess() for _ in range(ngpu)]
        for dev_id in range(ngpu):
            self.all_procs[dev_id].initialize(dev_id)

        self.avail_procs = set(self.all_procs)

    def has_avail_proc(self):
        return len(self.avail_procs) > 0

    def allocate_proc(self):
        assert self.has_avail_proc()
        return self.avail_procs.pop()

    def return_proc(self, proc):
        self.avail_procs.add(proc)

    def teardown(self):
        for proc in self.all_procs:
            proc.terminate()


if config.autotune_in_subproc:
    tuning_process_pool = TuningProcessPool()


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
            # don't use self.device since we may benchmark on diferent GPU
            device="cuda",
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

    def benchmark(self) -> float:
        if DEBUG:
            start_ts = time.time()

        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        run = getattr(mod, self.kernel_name).run

        if DEBUG:
            load_elapse = time.time() - start_ts
            start_ts = time.time()

        # create args and out tensor
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

        out = do_bench(worker)[0]
        torch.cuda.synchronize()  # shake out any CUDA errors

        if DEBUG:
            bench_elapse = time.time() - start_ts
            print(
                f"InChidProcess-{torch.cuda.current_device()} {self.module_cache_key}: load {load_elapse}, "
                + f"create tensor {create_tensor_elapse}, bench {bench_elapse}"
            )
        return out


def benchmark_in_sub_process(choices):
    timings = {}
    if len(choices) == 0:
        return timings

    if DEBUG:
        print(f"Tuning {len(choices)} choices in sub processes")

    pending_tasks = {}  # map choice to proc
    reqlist = [choice.bmreq for choice in choices]
    nextreqidx = 0
    while nextreqidx < len(reqlist) or len(pending_tasks) > 0:
        while nextreqidx < len(reqlist) and tuning_process_pool.has_avail_proc():
            proc = tuning_process_pool.allocate_proc()
            bmreq = reqlist[nextreqidx]

            proc.request_queue.put(bmreq)
            pending_tasks[choices[nextreqidx]] = proc
            nextreqidx += 1

        for choice, proc in pending_tasks.items():
            try:
                # small timeout so the parent process does not stuck too long if
                # the child process is still busy doing its work.
                timing = proc.response_queue.get(timeout=0.001)
            except queue.Empty:
                status = proc.process.exitcode
                if status is None:
                    # still running
                    continue
                # otherwise a crash happens
                assert (
                    status != 0
                ), f"Child process should be crashed but get status code {status}"
                warnings.warn(
                    f"Fail to benchmark choice '{choice}'. It will be ignored. Please debug the root cause in case the choice can bring perf gains."  # noqa: B950 line too long
                )

                timing = float("inf")

                # must reinitialize proc
                proc.clear()
                proc.initialize()

                # fall through

            timings[choice] = timing
            tuning_process_pool.return_proc(proc)
        pending_tasks = {
            choice: proc
            for choice, proc in pending_tasks.items()
            if choice not in timings
        }

    return timings
