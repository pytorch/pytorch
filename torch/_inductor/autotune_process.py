import dataclasses
import pickle
import subprocess
import sys
import time
import warnings

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import PyCodeCache

if TYPE_CHECKING:
    from torch._inductor.select_algorithm import TritonTemplateCaller

from .utils import do_bench
from .virtualized import V

DEBUG = False
EXIT_HANDLER_REGISTERED = False


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


@dataclasses.dataclass
class TuningProcess:
    process: Optional["subprocess.Popen[bytes]"] = None

    @staticmethod
    def process_main() -> None:
        """
        Work loop for the benchmarking subprocess.
        """
        if DEBUG:
            print("Entering TuningProcess child main", file=sys.stderr)

        def reply(obj):
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
        Create child process and do the warm up.
        """
        if self.process is not None:
            return

        self.process = subprocess.Popen(
            [sys.executable, "-m", "torch._inductor.autotune_process_entry"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # register the exit handler for the parent process so it will terminate
        # the child processes
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(lambda: self.terminate())

        # wait for the initialization to be done
        self.put(Ping())
        resp = self.get()
        assert isinstance(resp, Pong)

    def put(self, obj: Any) -> None:
        """
        Push a work item to the child process.
        """
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
            # Child crashed; recreate it
            self.process = None
            self.initialize()
            raise
        except pickle.UnpicklingError:
            raise RuntimeError(
                "Error deserializing response from the benchmarking subprocess. "
                "Is the benchmark code path writing to stdout?"
            )

    def terminate(self) -> None:
        """
        Signal the child process to terminate and wait for it to exit.
        """
        if self.process is not None:
            self.put(None)
            self.process.wait()
            self.process = None


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

    module_path: str  # the path of the module defining the triton kernel
    module_cache_key: str
    kernel_name: str  # the kernel name defined in the module
    grid: List[int]
    extra_args: Dict[str, Any]
    num_stages: int
    num_warps: int

    input_tensors: Union["TensorMeta", List["TensorMeta"]]
    output_tensor: Union["TensorMeta", List["TensorMeta"]]

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        if DEBUG:
            start_ts = time.time()

        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        if DEBUG:
            print(
                f"benchmark module key: {self.module_cache_key}, path: {self.module_path}",
                file=sys.stderr,
            )

        run = getattr(mod, self.kernel_name).run

        if DEBUG:
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

        if DEBUG:
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

        if DEBUG:
            bench_elapse = time.time() - start_ts
            print(
                f"InChidProcess {self.module_cache_key}: load {load_elapse}, "
                + f"create tensor {create_tensor_elapse}, bench {bench_elapse}",
                file=sys.stderr,
            )
        return out


def benchmark_in_sub_process(
    choice: "TritonTemplateCaller",
) -> float:
    """
    Do benchmarking in subprocess and return the perf number (latency).
    """
    assert choice.bmreq is not None
    tuning_process.initialize()

    tuning_process.put(choice.bmreq)
    try:
        return tuning_process.get()
    except EOFError:
        warnings.warn(
            f"Failed to benchmark choice '{choice}'. It will be ignored. "
            "Please debug the root cause in case the choice can bring perf gains."
        )
        # return INF so this choice will be ignored
        return float("inf")
