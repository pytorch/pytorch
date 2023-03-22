import dataclasses
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

        out = do_bench(worker)[0]
        torch.cuda.synchronize()  # shake out any CUDA errors

        if DEBUG:
            bench_elapse = time.time() - start_ts
            print(
                f"InChidProcess {self.module_cache_key}: load {load_elapse}, "
                + f"create tensor {create_tensor_elapse}, bench {bench_elapse}"
            )
        return out


def process_main(bmreq: BenchmarkRequest, timings: torch.Tensor):
    """
    The main function for the child process.
    """
    timings[0] = bmreq.benchmark()


def benchmark_in_sub_process(
    choice: "ChoiceCaller",
) -> float:
    assert choice.bmreq is not None

    # use a tensor since the mutation to a python list in a sub process
    # is not synced back to the parent process. While a tensor works well since
    # they are moved to shared memory.
    # TODO: can ue a Queue instead.
    timings = torch.zeros(1, dtype=torch.float32)

    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=process_main,
        args=(
            choice.bmreq,
            timings,
        ),
    )
    child.start()
    child.join()

    # child process fail
    if child.exitcode != 0:
        warnings.warn(
            f"Fail to benchmark choice '{choice}'. It will be ignored. Please debug the root cause in case the choice can bring perf gains."  # noqa: B950 line too long
        )
        # return a large value to this choice will be ignored
        return float("inf")

    return timings[0].item()
