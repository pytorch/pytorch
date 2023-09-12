from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import PyCodeCache

if TYPE_CHECKING:
    from torch._inductor.select_algorithm import TritonTemplateCaller

from . import config
from .utils import do_bench
from .virtualized import V


EXIT_HANDLER_REGISTERED = False

log = logging.getLogger(__name__)


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


@dataclasses.dataclass
class TuningProcess:
    """
    Abstraction for launching a helper process to benchmark kernels. Rather
    than spawning the parent process, the approach Popens a new process with
    an entry point that we control. Avoiding the spawn means we do not re-enter
    the toplevel script. The subprocess communicates with the parent process
    via pickling requests/responses over stdin/stdout pipes.
    """

    device: Optional[int] = None
    process: Optional[subprocess.Popen[bytes]] = None

    @staticmethod
    def process_main() -> None:
        """
        Entry point for the child process.
        """
        log.debug("Entering TuningProcess child main")
        try:
            TuningProcess.workloop()
        except Exception:
            log.exception("Exception in TuningProcess")

    @staticmethod
    def workloop() -> None:
        """
        Work loop for the benchmarking subprocess.
        """

        def reply(obj):
            # Note this is subtly different than the put() method below.
            pickle.dump(obj, sys.stdout.buffer)
            sys.stdout.flush()

        while True:
            obj = pickle.load(sys.stdin.buffer)
            if obj is None:
                # None is a sentinel for the child to terminate
                break
            elif isinstance(obj, Ping):
                reply(Pong())
            elif isinstance(obj, BenchmarkRequest):
                reply(obj.benchmark())
            else:
                raise RuntimeError(f"Invalid request type {type(obj)}")

    def initialize(self) -> None:
        """
        Create child process. Set the environment to make only the provided
        GPU device visible to the process.
        """
        if self.process is not None:
            return

        env = os.environ.copy()
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "torch._inductor.autotune_process_entry"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

    def put(self, obj: Any) -> None:
        """
        Push a work item to the child process.
        """
        # In case of a prior crash, ensure the subprocess is running
        self.initialize()
        assert self.process is not None
        assert self.process.stdin is not None
        pickle.dump(obj, self.process.stdin)
        self.process.stdin.flush()

    def get(self) -> Any:
        """
        Get a response from the child process.
        """
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            return pickle.load(self.process.stdout)
        except EOFError:
            # Child crashed; clean up
            self.close()
            raise
        except pickle.UnpicklingError as ex:
            raise RuntimeError(
                "Error deserializing response from the benchmarking subprocess. "
                "Is the benchmark code path writing to stdout?"
            ) from ex

    def close(self) -> None:
        """
        Close the communication pipes from the child process.
        """
        if self.process is not None:
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def terminate(self) -> None:
        """
        Signal the child process to terminate.
        """
        if self.process is not None:
            self.put(None)

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.process is not None:
            self.process.wait()
            self.close()


@dataclasses.dataclass
class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    processes: Optional[Queue[TuningProcess]] = None
    executor: Optional[ThreadPoolExecutor] = None

    def initialize(self, count: Optional[int] = None) -> None:
        """
        Start the child processes.
        """
        assert (self.processes is None) == (self.executor is None)
        if self.processes is not None:
            return

        if config.autotune_multi_device:
            count = count or torch.cuda.device_count()
        else:
            count = 1
        assert count > 0 and count <= torch.cuda.device_count()

        # Launch the child processes and push a msg to "warm up"
        self.processes = Queue()
        for device in range(count):
            p = TuningProcess(device=device if config.autotune_multi_device else None)
            p.initialize()
            p.put(Ping())
            self.processes.put(p)

        # Wait for the initialization to finish
        for p in self.processes.queue:
            assert isinstance(p.get(), Pong)

        # Use a thread pool to manage distributing work to the subprocesses.
        # Threads block on an available process, so it makes sense to match
        # the number of threads with the number of devices.
        self.executor = ThreadPoolExecutor(max_workers=count)

        # Register the exit handler for the parent process so it will terminate
        # the child processes.
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(lambda: self.terminate())

    def terminate(self) -> None:
        """
        Signal all child processes to terminate.
        """
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

        if self.processes is not None:
            for p in self.processes.queue:
                p.terminate()
            for p in self.processes.queue:
                p.wait()
            self.processes = None

    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        assert choice.bmreq is not None
        assert self.processes is not None

        process = self.processes.get()
        process.put(choice.bmreq)
        try:
            return process.get()
        except EOFError:
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # set to INF so this choice will be ignored
            return float("inf")
        finally:
            self.processes.put(process)

    def benchmark(
        self,
        choices: List[TritonTemplateCaller],
    ) -> Dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """
        assert self.processes is not None, "Tuning process pool is not initialized"
        assert self.executor is not None

        results = {}

        # Use a ThreadExecutorPool to spread the work across the subproccesses and
        # to grab subprocesses as soon as they're free.
        for choice, result in zip(choices, self.executor.map(self.target, choices)):
            results[choice] = result

        return results


tuning_pool = TuningProcessPool()


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
    ) -> Union[TensorMeta, List[TensorMeta]]:
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

    module_path: str  # the path of the module defining the triton kernel
    module_cache_key: str
    kernel_name: str  # the kernel name defined in the module
    grid: List[int]
    extra_args: Dict[str, Any]
    num_stages: int
    num_warps: int

    input_tensors: Union[TensorMeta, List[TensorMeta]]
    output_tensor: Union[TensorMeta, List[TensorMeta]]

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        debug = log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )

        run = getattr(mod, self.kernel_name).run

        if debug:
            load_elapse = time.time() - start_ts
            start_ts = time.time()

        # create args and out tensor
        if output_tensor is None:
            assert len(input_tensors) == 0
            if isinstance(self.input_tensors, List):
                input_tensors = tuple(x.to_tensor() for x in self.input_tensors)
            if isinstance(self.input_tensors, TensorMeta):
                input_tensors = tuple(self.input_tensors.to_tensor())
            assert isinstance(self.output_tensor, TensorMeta)
            output_tensor = self.output_tensor.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts
            start_ts = time.time()

        def worker() -> float:
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

        if debug:
            bench_elapse = time.time() - start_ts
            log.debug(
                "InChildProcess %s: load %f, create tensor %f, bench %f",
                self.module_cache_key,
                load_elapse,
                create_tensor_elapse,
                bench_elapse,
            )

        return out


class TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file so that the TuningProcess
    sub-process knows how to unpickle these objects.
    """

    def __init__(self, value: Optional[float] = None) -> None:
        self.value = value

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        if self.value is None:
            raise Exception("Failed to run")
        return self.value


def benchmark_in_sub_process(
    choices: List[TritonTemplateCaller],
) -> Dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    return tuning_pool.benchmark(choices)
