# mypy: allow-untyped-defs
import itertools
import random
from typing import Any, Optional, Union

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    ShapeAsConstantBuffer,
    TensorBox,
)
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch
from torch._inductor.utils import ensure_cutlass_api_available
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")


class CutlassAPIBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for cutlass_api kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self._compiled_artifact = None

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """Create a function to run the cutlass_api kernel."""
        import cutlass_api

        a, b = input_tensors

        args = cutlass_api.arguments.GemmArguments(
            a,
            b,
            out,
            accumulator_type=self.accumulator_type,
        )

        if self._compiled_artifact is None:
            self._compiled_artifact = self.kernel.compile(args)

        artifact = self._compiled_artifact
        kernel = self.kernel

        def run_kernel():
            kernel.run(args, artifact, assume_supported_args=True)

        return run_kernel

    def cleanup_run_fn(self) -> None:
        pass


class CutlassAPIGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for cutlass_api GEMM kernels.

    Wraps a cutlass_api kernel and integrates with Inductor's autotuning.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"cutlass_api {kernel.metadata.kernel_name}",
        )
        self.kernel = kernel
        self.accumulator_type = accumulator_type

        output_buffer = Buffer(name="cutlass_api_out", layout=layout)

        self.bmreq = CutlassAPIBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel=kernel,
            accumulator_type=accumulator_type,
        )

    def __str__(self) -> str:
        return f"CutlassAPIGemmCaller({self.kernel.metadata.kernel_name})"

    def benchmark(self, *args, out) -> float:
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> Union[TensorBox, ShapeAsConstantBuffer]:
        from torch._inductor.ir import CutlassAPIGemmBuffer

        return TensorBox.create(
            CutlassAPIGemmBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                kernel=self.kernel,
                accumulator_type=self.accumulator_type,
            )
        )

    def call_name(self) -> str:
        return self.name

    def to_callable(self):
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        return f"cutlass_api_{self.kernel.metadata.kernel_name}"

    def info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": "cutlass_api",
            "kernel_name": self.kernel.metadata.kernel_name,
        }


def _create_dummy_tensor_from_layout(layout: Layout) -> Optional[torch.Tensor]:
    """
    Create a FakeTensor from a Layout for kernel filtering.

    Uses Layout.get_example() which creates FakeTensors within V.fake_mode,
    avoiding real CUDA memory allocation. cutlass_api only needs shape/stride/dtype
    metadata for its supports() checks.
    """
    try:
        return layout.get_example()
    except Exception:
        return None


def add_cutlass_api_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: Optional[torch.dtype] = None,
) -> None:
    """
    Add cutlass_api GEMM kernels to the autotune choices.

    Queries cutlass_api for compatible kernels and adds them as autotune choices.
    """
    if ensure_cutlass_api_available():
        import cutlass_api
    else:
        log.debug("cutlass_api not available, skipping cutlass_api choices")
        return

    if accumulator_type is None:
        accumulator_type = torch.float32

    a_node, b_node = input_nodes

    # Create dummy tensors for cutlass_api's supports() checks
    a_tensor = _create_dummy_tensor_from_layout(a_node.get_layout())
    b_tensor = _create_dummy_tensor_from_layout(b_node.get_layout())
    out_tensor = _create_dummy_tensor_from_layout(layout)

    if a_tensor is None or b_tensor is None or out_tensor is None:
        log.debug("Failed to create dummy tensors")
        return

    try:
        args = cutlass_api.arguments.GemmArguments(
            a_tensor,
            b_tensor,
            out_tensor,
            accumulator_type=accumulator_type,
        )
    except Exception:
        log.debug("Failed to create GemmArguments", exc_info=True)
        return

    cc = get_cuda_arch()
    if cc is None:
        log.debug("Failed to get CUDA arch")
        return
    cc_int = int(cc)

    try:
        kernels = cutlass_api.get_kernels(args=args, cc=cc_int)
    except Exception:
        log.debug("Failed to get cutlass_api kernels", exc_info=True)
        return

    if not kernels:
        log.debug("No compatible cutlass_api kernels found")
        return

    # Limit kernels to profile if configured
    if config.cuda.cutlass_api_max_profiling_configs:
        kernels = random.sample(
            kernels,
            min(len(kernels), config.cuda.cutlass_api_max_profiling_configs),
        )

    num_added = 0
    for kernel in kernels:
        name = f"cutlass_api_gemm_{next(CutlassAPIGemmCaller.index_counter)}"
        try:
            caller = CutlassAPIGemmCaller(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                kernel=kernel,
                accumulator_type=accumulator_type,
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            log.debug("Failed to create cutlass_api choice", exc_info=True)

    log.debug("Added %d cutlass_api GEMM choices", num_added)
