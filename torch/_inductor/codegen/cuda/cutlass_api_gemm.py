# mypy: allow-untyped-defs
"""
CutlassAPI GEMM backend for PyTorch Inductor.

This module provides integration with the cutlass_api library, which offers
pre-built CUTLASS kernels for GEMM operations. Unlike CuteDSL templates
(which generate Python code from Jinja templates), cutlass_api provides
pre-compiled kernel objects that can be directly benchmarked and executed.
"""

import itertools
import random
from typing import Any, Optional, Union

import torch
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.ir import Buffer, ChoiceCaller, Layout, TensorBox
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")

MAX_CUTLASS_API_PROFILING_CONFIGS = 5


class CutlassAPIBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    Benchmark request for cutlass_api kernels.

    This class handles benchmarking of cutlass_api kernels in potentially
    separate processes. Since kernel objects are not serializable, we store
    the kernel metadata and recreate the kernel when needed.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        kernel_metadata: dict[str, Any],
        accumulator_type: torch.dtype,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self._kernel = None
        self._compiled_artifact = None

    def _get_kernel(self):
        """
        Lazily get the kernel object.

        Kernel objects are not serializable, so we recreate them using the
        stored metadata when needed (e.g., in a subprocess).
        """
        if self._kernel is None:
            import cutlass_api

            # Find the matching kernel by name
            kernels = cutlass_api.get_kernels(
                metadata_filter=lambda m: m.kernel_name
                == self.kernel_metadata["kernel_name"]
            )
            if not kernels:
                raise RuntimeError(
                    f"Could not find cutlass_api kernel: {self.kernel_metadata['kernel_name']}"
                )
            self._kernel = kernels[0]
        return self._kernel

    def _get_accumulator_type(self):
        """Convert torch dtype to cutlass accumulator type."""
        import cutlass

        dtype_map = {
            torch.float32: cutlass.Float32,
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }
        return dtype_map.get(self.accumulator_type, cutlass.Float32)

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """
        Create a function to run the cutlass_api kernel with the given tensors.
        """
        import cutlass_api

        kernel = self._get_kernel()
        a, b = input_tensors

        # Create GemmArguments with actual tensors
        args = cutlass_api.arguments.GemmArguments(
            a,
            b,
            out,
            accumulator_type=self._get_accumulator_type(),
        )

        # Compile if not cached
        if self._compiled_artifact is None:
            self._compiled_artifact = kernel.compile(args)

        artifact = self._compiled_artifact

        def run_kernel():
            kernel.run(args, artifact, assume_supported_args=True)

        return run_kernel

    def cleanup_run_fn(self) -> None:
        """Clean up any resources used by the kernel."""


class CutlassAPIGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for cutlass_api GEMM kernels.

    This class wraps a cutlass_api kernel object and integrates it with
    Inductor's autotuning system.
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

        # Create a fake output buffer for TensorMeta
        output_buffer = Buffer(name="cutlass_api_out", layout=layout)

        # Create benchmark request with serializable metadata
        self.bmreq = CutlassAPIBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel_metadata=self._serialize_metadata(kernel.metadata),
            accumulator_type=accumulator_type,
        )

    def _serialize_metadata(self, metadata) -> dict[str, Any]:
        """
        Convert KernelMetadata to a serializable dict.

        Only include fields needed to recreate the kernel.
        """
        return {
            "kernel_name": metadata.kernel_name,
            "min_cc": metadata.min_cc,
        }

    def __str__(self) -> str:
        return f"CutlassAPIGemmCaller({self.kernel.metadata.kernel_name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        """Create the output node for this kernel choice."""
        # Import here to avoid circular imports
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
        """Return the kernel call name."""
        return self.name

    def to_callable(self):
        """Return callable that can execute this kernel."""
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        """Return unique hash key for this choice."""
        return f"cutlass_api_{self.kernel.metadata.kernel_name}"

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel."""
        return {
            "name": self.name,
            "backend": "cutlass_api",
            "kernel_name": self.kernel.metadata.kernel_name,
        }


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    """Convert torch dtype to cutlass dtype for comparison."""
    import cutlass

    dtype_map = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.int8: cutlass.Int8,
        torch.int32: cutlass.Int32,
    }
    return dtype_map.get(dtype)


def _create_metadata_filter(
    a_dtype: torch.dtype, b_dtype: torch.dtype, out_dtype: torch.dtype
):
    """
    Create a metadata filter function for cutlass_api.get_kernels().

    Filters kernels based on input/output dtype compatibility.
    """
    cutlass_a_dtype = _torch_dtype_to_cutlass(a_dtype)
    cutlass_b_dtype = _torch_dtype_to_cutlass(b_dtype)
    cutlass_out_dtype = _torch_dtype_to_cutlass(out_dtype)

    def metadata_filter(meta) -> bool:
        """Filter kernels by dtype compatibility."""
        try:
            ops = meta.operands
            # Check if the kernel's operand dtypes match our requirements
            if hasattr(ops, "A") and hasattr(ops, "B") and hasattr(ops, "out"):
                if cutlass_a_dtype is not None and ops.A.dtype != cutlass_a_dtype:
                    return False
                if cutlass_b_dtype is not None and ops.B.dtype != cutlass_b_dtype:
                    return False
                if cutlass_out_dtype is not None and ops.out.dtype != cutlass_out_dtype:
                    return False
            return True
        except Exception:
            # If we can't check metadata, include the kernel and let runtime check handle it
            return True

    return metadata_filter


def add_cutlass_api_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: Optional[torch.dtype] = None,
) -> None:
    """
    Add cutlass_api GEMM kernels to the autotune choices.

    This function queries cutlass_api for compatible GEMM kernels and adds
    them as choices for autotuning.

    Args:
        choices: List of ChoiceCaller objects to append to
        layout: Output layout
        input_nodes: Input buffer nodes [A, B]
        accumulator_type: Data type for accumulation (default: float32)
    """
    try:
        import cutlass_api
    except ImportError:
        log.debug("cutlass_api not available, skipping cutlass_api choices")
        return

    if accumulator_type is None:
        accumulator_type = torch.float32

    # Extract dtypes from input nodes
    a_node, b_node = input_nodes
    a_dtype = a_node.get_dtype()
    b_dtype = b_node.get_dtype()
    out_dtype = layout.dtype

    # Create metadata filter based on dtypes
    metadata_filter = _create_metadata_filter(a_dtype, b_dtype, out_dtype)

    # Get compatible kernels
    try:
        kernels = cutlass_api.get_kernels(metadata_filter=metadata_filter)
    except Exception as e:
        log.debug("Failed to get cutlass_api kernels: %s", e)
        return

    if not kernels:
        log.debug(
            "No compatible cutlass_api kernels found for dtypes: A=%s, B=%s, out=%s",
            a_dtype,
            b_dtype,
            out_dtype,
        )
        return

    if len(kernels) > MAX_CUTLASS_API_PROFILING_CONFIGS:
        kernels = random.sample(kernels, MAX_CUTLASS_API_PROFILING_CONFIGS)

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
        except Exception as e:
            log.debug("Failed to create cutlass_api choice for %s: %s", name, e)

    log.debug("Added %d cutlass_api GEMM choices", num_added)
